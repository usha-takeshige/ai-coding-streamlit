# 販売データ分析ダッシュボード

このプロジェクトは、販売データを分析し、インタラクティブなダッシュボードを通じて可視化するWebアプリケーションです。

## 機能概要

- 総売上高、取引件数、平均購入金額などの基本統計情報の表示
- カテゴリー別売上構成比の分析
- 期間、カテゴリー、地域によるフィルタリング機能
- 時系列での売上推移分析
- 顧客属性（年齢層、性別）による購買傾向分析
- 地域別売上分布の可視化
- 商品カテゴリー分析
- インタラクティブなデータテーブル

## 技術スタック

- Python 3.8+
- Streamlit
- Pandas
- Plotly

## セットアップ方法

1. リポジトリのクローン
```bash
git clone [リポジトリURL]
cd [プロジェクトディレクトリ]
```

2. 仮想環境の作成と有効化
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

3. 依存パッケージのインストール
```bash
pip install -r requirements.txt
```

## 使用方法

1. アプリケーションの起動
```bash
streamlit run app.py
```

2. ブラウザでアクセス
- デフォルトで http://localhost:8501 が開きます

## データ形式

入力データは以下の形式のCSVファイルを想定しています：
- 取引日時
- 商品カテゴリー
- 売上金額
- 顧客情報（年齢層、性別）
- 地域情報
- 支払方法

詳細な形式については `data_description.md` を参照してください。

## フィルター機能

- 期間選択：特定の期間のデータを分析
- カテゴリーフィルター：特定の商品カテゴリーに絞った分析
- 地域フィルター：特定の地域のデータを分析

## グラフとビジュアライゼーション

1. 時系列分析
   - 月次売上推移
   - 曜日別売上傾向
   - トレンド分析

2. 顧客分析
   - 年齢層別購買傾向
   - 性別による購買パターン
   - 地域別売上分布

3. 商品分析
   - カテゴリー別売上比率
   - カテゴリー別平均購入金額
   - 支払方法の分布

## 注意事項

- データは日次で更新されることを想定しています
- 大規模なデータセットの場合、初回読み込みに時間がかかる場合があります
- フィルター適用時はリアルタイムで計算が行われるため、レスポンスに若干の遅延が生じる可能性があります

## サポート

問題や質問がある場合は、Issueを作成してください。
