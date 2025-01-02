import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from datetime import datetime
from utils.data_processor import (
    preprocess_data,
    calculate_basic_stats,
    analyze_time_series,
    analyze_customer_segments,
)


def plotly_chart(fig, use_container_width=True):
    """
    Plotlyグラフを表示し、クリックイベントを処理する関数

    Args:
        fig: plotly.graph_objects.Figure - 表示するPlotlyグラフ
        use_container_width: bool - コンテナの幅を使用するかどうか
    """
    # クリックイベントの設定
    config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }

    # グラフの表示
    st.plotly_chart(fig, use_container_width=use_container_width, config=config)


# ページ設定
st.set_page_config(
    page_title="販売データ分析ダッシュボード", page_icon="📊", layout="wide"
)

# セッション状態の初期化
if "drilldown_filter" not in st.session_state:
    st.session_state.drilldown_filter = {"active": False, "type": None, "value": None}


# データ読み込み関数
@st.cache_data
def load_data():
    """
    CSVファイルからデータを読み込み、前処理を行う関数
    """
    try:
        # CSVファイルを読み込む
        df = pd.read_csv("data/sample_data.csv")

        # データの前処理
        df = preprocess_data(df)

        return df
    except Exception as e:
        st.error(f"データの読み込み中にエラーが発生しました: {str(e)}")
        return None


def apply_filters(
    df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    selected_categories: list,
    selected_regions: list,
) -> pd.DataFrame:
    """
    データフレームにフィルターを適用する関数
    """
    # 日付フィルター
    start_datetime = pd.Timestamp(start_date)
    end_datetime = pd.Timestamp(end_date)
    mask = (df["購入日"] >= start_datetime) & (df["購入日"] <= end_datetime)

    # カテゴリーフィルター
    if selected_categories:
        mask = mask & (df["購入カテゴリー"].isin(selected_categories))

    # 地域フィルター
    if selected_regions:
        mask = mask & (df["地域"].isin(selected_regions))

    # ドリルダウンフィルターの適用
    if st.session_state.drilldown_filter["active"]:
        filter_type = st.session_state.drilldown_filter["type"]
        filter_value = st.session_state.drilldown_filter["value"]
        if filter_type and filter_value:
            mask = mask & (df[filter_type] == filter_value)

    return df[mask]


def display_basic_metrics(stats: dict):
    """
    基本的な指標を表示する関数
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("総売上高", f"¥{stats['総売上高']:,.0f}")
    with col2:
        st.metric("平均購入金額", f"¥{stats['平均購入金額']:,.0f}")
    with col3:
        st.metric("取引件数", f"{stats['取引件数']:,}件")
    with col4:
        st.metric("ユニーク顧客数", f"{stats['ユニーク顧客数']:,}人")


def enhance_plotly_figure(fig, hover_data=None, click_handler=None):
    """
    Plotlyグラフにインタラクティブ機能を追加する関数
    """
    if hover_data:
        # ホバーテンプレートの修正
        hover_template = "<br>".join(
            [
                f"{key}: {value}" if not value.startswith("%{") else f"{key}: {value}"
                for key, value in hover_data.items()
            ]
        )
        hover_template += "<extra></extra>"  # 余分な凡例を非表示

        fig.update_traces(hovertemplate=hover_template)

    # クリックイベントの設定
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
        ),
        clickmode="event+select",
    )

    # クリックイベントのコールバック設定
    if click_handler:
        fig.data[0].on_click(click_handler)

    return fig


def handle_chart_click(chart_type, selected_data):
    """
    グラフのクリックイベントを処理する関数
    """
    if selected_data:
        if (
            st.session_state.drilldown_filter["active"]
            and st.session_state.drilldown_filter["type"] == chart_type
            and st.session_state.drilldown_filter["value"] == selected_data
        ):
            # 同じ項目をクリックした場合、フィルターを解除
            st.session_state.drilldown_filter = {
                "active": False,
                "type": None,
                "value": None,
            }
        else:
            # 新しい項目をクリックした場合、フィルターを設定
            st.session_state.drilldown_filter = {
                "active": True,
                "type": chart_type,
                "value": selected_data,
            }
        st.experimental_rerun()


def display_time_series_analysis(df: pd.DataFrame):
    """
    時系列分析の結果を表示する関数
    """
    st.header("時系列分析")

    # 分析の実行
    time_analysis = analyze_time_series(df)

    # 月次売上推移
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("月次売上推移")
        fig_monthly = px.line(
            time_analysis["monthly_sales"],
            x="購入日",
            y="購入金額",
            title="月次売上推移",
        )

        def monthly_click_handler(trace, points, state):
            if points.point_inds:
                handle_chart_click("購入日", points.xs[points.point_inds[0]])

        fig_monthly = enhance_plotly_figure(
            fig_monthly,
            {"日付": "%{x}", "売上高": "%{y:,.0f}円"},
            click_handler=monthly_click_handler,
        )
        plotly_chart(fig_monthly, use_container_width=True)

    # 曜日別売上傾向
    with col2:
        st.subheader("曜日別売上傾向")
        weekday_data = time_analysis["weekday_sales"].copy()
        weekday_data.columns = [
            "購入日",
            "売上合計",
            "平均売上",
            "取引件数",
        ]
        fig_weekday = px.bar(
            weekday_data,
            x="購入日",
            y="売上合計",
            title="曜日別売上高",
        )

        def weekday_click_handler(trace, points, state):
            if points.point_inds:
                handle_chart_click("購入日", points.xs[points.point_inds[0]])

        fig_weekday = enhance_plotly_figure(
            fig_weekday,
            {
                "曜日": "%{x}",
                "売上高": "%{y:,.0f}円",
                "取引件数": "%{customdata[0]:,}件",
            },
            click_handler=weekday_click_handler,
        )
        fig_weekday.update_traces(customdata=weekday_data[["取引件数"]])
        plotly_chart(fig_weekday, use_container_width=True)

    # 時系列トレンド
    st.subheader("売上トレンド（7日移動平均）")
    fig_trend = px.line(
        time_analysis["daily_sales"],
        x="購入日",
        y=["購入金額", "移動平均"],
        title="日次売上高と移動平均",
    )

    def trend_click_handler(trace, points, state):
        if points.point_inds:
            handle_chart_click("購入日", points.xs[points.point_inds[0]])

    fig_trend = enhance_plotly_figure(
        fig_trend,
        {"日付": "%{x}", "売上高": "%{y:,.0f}円"},
        click_handler=trend_click_handler,
    )
    plotly_chart(fig_trend, use_container_width=True)


def display_customer_analysis(df: pd.DataFrame):
    """
    顧客分析の結果を表示する関数
    """
    st.header("顧客分析")

    # 分析の実行
    customer_analysis = analyze_customer_segments(df)

    # 年齢層別分析
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("年齢層別購買傾向")
        age_data = customer_analysis["age_group_analysis"].copy()
        age_data.columns = ["年齢層", "売上合計", "平均売上", "取引件数"]
        fig_age = px.bar(
            age_data,
            x="年齢層",
            y="売上合計",
            title="年齢層別売上高",
            custom_data=["平均売上", "取引件数"],
        )

        def age_click_handler(trace, points, state):
            if points.point_inds:
                handle_chart_click("年齢層", points.xs[points.point_inds[0]])

        fig_age = enhance_plotly_figure(
            fig_age,
            {
                "年齢層": "%{x}",
                "売上高": "%{y:,.0f}円",
                "平均売上": "%{customdata[0]:,.0f}円",
                "取引件数": "%{customdata[1]:,}件",
            },
            click_handler=age_click_handler,
        )
        plotly_chart(fig_age, use_container_width=True)

    # 性別による分析
    with col2:
        st.subheader("性別購買傾向")
        gender_data = customer_analysis["gender_analysis"].copy()
        gender_data.columns = ["性別", "売上合計", "平均売上", "取引件数"]

        fig_gender = px.pie(
            gender_data,
            names="性別",
            values="売上合計",
            title="性別売上構成比",
        )

        def gender_click_handler(trace, points, state):
            if points.point_inds:
                handle_chart_click("性別", points.label[points.point_inds[0]])

        fig_gender.update_traces(
            textinfo="label+percent",
            hovertemplate=(
                "<b>%{label}</b><br>"
                + "売上高: ¥%{value:,.0f}<br>"
                + "構成比: %{percent}<br>"
                + "<extra></extra>"
            ),
            hoverlabel=dict(bgcolor="white", font_size=12),
        )

        fig_gender = enhance_plotly_figure(
            fig_gender, click_handler=gender_click_handler
        )
        plotly_chart(fig_gender, use_container_width=True)

    # 地域別分析
    st.subheader("地域別売上分布")
    region_data = customer_analysis["region_analysis"].copy()
    region_data.columns = ["地域", "売上合計", "平均売上", "取引件数"]
    fig_region = px.bar(
        region_data,
        x="地域",
        y="売上合計",
        title="地域別売上高",
        custom_data=["平均売上", "取引件数"],
    )

    def region_click_handler(trace, points, state):
        if points.point_inds:
            handle_chart_click("地域", points.xs[points.point_inds[0]])

    fig_region = enhance_plotly_figure(
        fig_region,
        {
            "地域": "%{x}",
            "売上高": "%{y:,.0f}円",
            "平均売上": "%{customdata[0]:,.0f}円",
            "取引件数": "%{customdata[1]:,}件",
        },
        click_handler=region_click_handler,
    )
    plotly_chart(fig_region, use_container_width=True)


def display_data_table(df: pd.DataFrame):
    """
    詳細データテーブルを表示する関数
    """
    st.header("取引データ詳細")

    # ソート機能の実装
    sort_column = st.selectbox(
        "ソート項目",
        ["購入日", "購入金額", "顧客ID", "年齢", "性別", "地域", "購入カテゴリー"],
    )
    sort_order = st.radio("ソート順", ["昇順", "降順"], horizontal=True)

    # データのソート
    ascending = sort_order == "昇順"
    sorted_df = df.sort_values(by=sort_column, ascending=ascending)

    # ページネーション
    rows_per_page = st.selectbox("表示件数", [10, 20, 50, 100])
    total_pages = len(sorted_df) // rows_per_page + (
        1 if len(sorted_df) % rows_per_page > 0 else 0
    )

    if "current_page" not in st.session_state:
        st.session_state.current_page = 0

    # ページ選択UI
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("前のページ") and st.session_state.current_page > 0:
            st.session_state.current_page -= 1
    with col2:
        st.write(f"ページ {st.session_state.current_page + 1} / {total_pages}")
    with col3:
        if st.button("次のページ") and st.session_state.current_page < total_pages - 1:
            st.session_state.current_page += 1

    # データの表示
    start_idx = st.session_state.current_page * rows_per_page
    end_idx = start_idx + rows_per_page

    # 表示するデータの整形
    display_df = sorted_df.iloc[start_idx:end_idx].copy()
    display_df["購入日"] = display_df["購入日"].dt.strftime("%Y-%m-%d")
    display_df["購入金額"] = display_df["購入金額"].apply(lambda x: f"¥{x:,.0f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )


def main():
    # タイトル
    st.title("販売データ分析ダッシュボード")

    # データの読み込み
    df = load_data()
    if df is None:
        return

    # サイドバーのフィルター
    st.sidebar.header("フィルター設定")

    # 日付範囲の選択
    min_date = df["購入日"].min().date()
    max_date = df["購入日"].max().date()
    start_date = st.sidebar.date_input("開始日", min_date)
    end_date = st.sidebar.date_input("終了日", max_date)

    # カテゴリーの選択
    categories = df["購入カテゴリー"].unique().tolist()
    selected_categories = st.sidebar.multiselect(
        "商品カテゴリー", categories, default=categories
    )

    # 地域の選択
    regions = df["地域"].unique().tolist()
    selected_regions = st.sidebar.multiselect("地域", regions, default=regions)

    # アクティブなドリルダウンフィルターの表示
    if st.session_state.drilldown_filter["active"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("アクティブなフィルター")
        st.sidebar.write(
            f"{st.session_state.drilldown_filter['type']}: "
            f"{st.session_state.drilldown_filter['value']}"
        )
        if st.sidebar.button("フィルターをクリア"):
            st.session_state.drilldown_filter = {
                "active": False,
                "type": None,
                "value": None,
            }
            st.experimental_rerun()

    # フィルターの適用
    filtered_df = apply_filters(
        df, start_date, end_date, selected_categories, selected_regions
    )

    # 基本統計情報の表示
    stats = calculate_basic_stats(filtered_df)
    display_basic_metrics(stats)

    # タブの作成
    tab1, tab2 = st.tabs(["分析ダッシュボード", "データテーブル"])

    with tab1:
        # 時系列分析の表示
        display_time_series_analysis(filtered_df)
        # 顧客分析の表示
        display_customer_analysis(filtered_df)

    with tab2:
        # データテーブルの表示
        display_data_table(filtered_df)


if __name__ == "__main__":
    main()
