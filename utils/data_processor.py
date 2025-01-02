import pandas as pd
import numpy as np
from datetime import datetime


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    データの前処理を行う関数

    Parameters:
    -----------
    df : pd.DataFrame
        処理前のデータフレーム

    Returns:
    --------
    pd.DataFrame
        処理後のデータフレーム
    """
    df = df.copy()

    # 日付データの型変換
    df["購入日"] = pd.to_datetime(df["購入日"])

    # 数値データの型変換
    df["購入金額"] = pd.to_numeric(df["購入金額"])
    df["年齢"] = pd.to_numeric(df["年齢"])
    df["顧客ID"] = pd.to_numeric(df["顧客ID"])

    # カテゴリカルデータの確認
    categorical_columns = ["性別", "地域", "購入カテゴリー", "支払方法"]
    for col in categorical_columns:
        df[col] = df[col].astype("category")

    return df


def calculate_basic_stats(df: pd.DataFrame) -> dict:
    """
    基本的な統計情報を計算する関数

    Parameters:
    -----------
    df : pd.DataFrame
        データフレーム

    Returns:
    --------
    dict
        基本統計情報を含む辞書
    """
    stats = {
        "総売上高": df["購入金額"].sum(),
        "平均購入金額": df["購入金額"].mean(),
        "取引件数": len(df),
        "ユニーク顧客数": df["顧客ID"].nunique(),
        "データ期間": {"開始日": df["購入日"].min(), "終了日": df["購入日"].max()},
    }
    return stats


def analyze_time_series(df: pd.DataFrame) -> dict:
    """
    時系列分析を行う関数

    Parameters:
    -----------
    df : pd.DataFrame
        データフレーム

    Returns:
    --------
    dict
        時系列分析結果を含む辞書
    """
    # 月次売上推移
    monthly_sales = (
        df.groupby(df["購入日"].dt.to_period("M"))
        .agg({"購入金額": "sum"})
        .reset_index()
    )
    monthly_sales["購入日"] = monthly_sales["購入日"].astype(str)

    # 曜日別売上傾向
    weekday_sales = (
        df.groupby(df["購入日"].dt.day_name())
        .agg({"購入金額": ["sum", "mean", "count"]})
        .reset_index()
    )

    # 時系列トレンド（7日移動平均）
    daily_sales = (
        df.groupby(df["購入日"].dt.date).agg({"購入金額": "sum"}).reset_index()
    )
    daily_sales["移動平均"] = daily_sales["購入金額"].rolling(window=7).mean()

    return {
        "monthly_sales": monthly_sales,
        "weekday_sales": weekday_sales,
        "daily_sales": daily_sales,
    }


def analyze_customer_segments(df: pd.DataFrame) -> dict:
    """
    顧客セグメント分析を行う関数

    Parameters:
    -----------
    df : pd.DataFrame
        データフレーム

    Returns:
    --------
    dict
        顧客分析結果を含む辞書
    """
    # 年齢層別分析
    df["年齢層"] = pd.cut(
        df["年齢"],
        bins=[0, 20, 30, 40, 50, 60, 100],
        labels=["20歳未満", "20代", "30代", "40代", "50代", "60歳以上"],
    )

    age_group_analysis = (
        df.groupby("年齢層").agg({"購入金額": ["sum", "mean", "count"]}).reset_index()
    )

    # 性別による分析
    gender_analysis = (
        df.groupby("性別").agg({"購入金額": ["sum", "mean", "count"]}).reset_index()
    )

    # 地域別分析
    region_analysis = (
        df.groupby("地域").agg({"購入金額": ["sum", "mean", "count"]}).reset_index()
    )

    return {
        "age_group_analysis": age_group_analysis,
        "gender_analysis": gender_analysis,
        "region_analysis": region_analysis,
    }
