# 天気からレンタサイクル需要を予測する

レンタサイクル需要予測動画で使った、分析・グラフ作成コードです。ソウルのレンタサイクル利用記録を使い、気温・湿度・雨などと利用回数の関係を調べ、**天気＋時刻・曜日・休日**から1時間あたりの利用回数を重回帰で予測します。

## 動画で扱う内容

- 天気とレンタサイクル利用回数の関係、平日・週末の時間帯パターン
- 非稼働時間の扱い、特徴量の作成、時系列での学習・評価分割
- 重回帰、予測結果と誤差、特徴量の置換重要度
- 損失関数の考え方（MSE / Weighted MSE）

公開コードの学習は `statsmodels` の通常の最小二乗法（OLS）です。Weighted MSEによる学習は実装していません。動画編集用のアニメーションや音声・映像素材はこのリポジトリに含めていません。

## データセット

[Seoul Bike Sharing Demand — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand)

- 期間：2017年12月1日〜2018年11月30日、8,760時間
- 目的変数：1時間あたりのレンタル回数（ユニーク利用者数ではありません）
- 稼働時間：8,465時間。非稼働295時間は学習・評価から除外
- 出典：*Seoul Bike Sharing Demand* [Dataset]. (2020). UCI Machine Learning Repository. [DOI: 10.24432/C5F62R](https://doi.org/10.24432/C5F62R)
- データのライセンス：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

CSVは同梱せず、下記のスクリプトでUCI公式配布ZIPから元のCSVを取得します。前処理では列名変更、日時・曜日・雨雪の有無などの派生列追加、非稼働時間の除外を行います。

## 実行方法

Python 3.11（元の環境は3.11.8）を使用します。まずリポジトリを取得し、そのフォルダに移動してください。

```bash
git clone https://github.com/Katucurry/seoul-bike-weather-demand.git
cd seoul-bike-weather-demand
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python src/analysis_pipeline.py
python src/generate_eda_gallery.py
```

Windowsでは仮想環境を `.venv\Scripts\activate` で有効化します。ダウンロードにはインターネット接続が必要です。手動取得する場合は、UCIの「Download」からZIPを展開し、`data/raw/SeoulBikeData.csv` に配置してください。CSVの文字コードはLatin-1です。

生成される主なファイル：

| 保存先 | 内容 |
|---|---|
| `reports/metrics.json` | 評価指標・データ件数・分割条件 |
| `reports/analysis_report.md` | 分析結果の説明 |
| `models/coefficients.csv` | 係数と信頼区間 |
| `models/model_summary.txt` | OLSの推定結果 |
| `data/processed/test_predictions.csv` | テスト期間の実測・予測・残差 |
| `gallery/index.html` | 動画用グラフ一覧 |
| `gallery/eda/index.html` | 探索的分析のグラフ一覧 |

`gallery/index.html` をブラウザで開いて図を確認できます。図の日本語はmacOSのHiraginoを自動利用します。他の環境では日本語フォントが必要です。文字が欠ける場合は、各スクリプトの `FONT_PATH` をインストール済み日本語フォントのパスに変更してください。

損失関数の説明用の等高線図だけを生成する場合：

```bash
python src/plot_loss_contour.py
```

出力は `assets/generated/loss_contour2d_contours.png` です。この図は説明用の人工的な二次関数で、レンタサイクルモデルの実測損失面ではありません。

## 使用ライブラリ

`requirements.txt` に主要ライブラリのバージョンを固定しています。検証環境の間接依存を含む全バージョンは `requirements-lock.txt` に保存しています。同じ構成を使う場合は `python -m pip install -r requirements-lock.txt` を実行してください。

- pandas / NumPy / SciPy：読み込み・前処理・数値計算
- statsmodels：重回帰（OLS）と係数の推定
- scikit-learn：MAE・MSE・R²の評価
- matplotlib / Pillow：グラフ描画・画像処理

## モデルと評価結果

時刻と曜日の交互作用、気温と気温の二乗、湿度、風速、日射、雨・雪、休日を使います。特徴量は元の単位で学習し、予測値は0以上にクリップします。露点温度は気温との相関が高いためモデルに含めていません。

**学習：2018年9月30日まで（7,152時間）、テスト：2018年10月1日以降（1,313時間）**。ランダム分割せず、将来期間で評価します。

| 方法 | MAE（回/時間） | RMSE（回/時間） | R² |
|---|---:|---:|---:|
| 学習期間の平均を予測 | 452.14 | 579.30 | -0.065 |
| 時刻×曜日の平均 | 276.01 | 386.12 | 0.527 |
| 天気＋時刻×曜日の重回帰 | 243.95 | 321.22 | 0.672 |

重回帰のMAEは全体平均のベースラインから約46.0%、時刻×曜日のベースラインから約11.6%改善しました。時刻の生活パターンが強く、天気が需要を補正する役割を持つ結果です。詳細は `reports/metrics.json` に保存しています。

観測された天気を使う検証であり、天気予報の誤差を含めた実運用の検証ではありません。係数は因果効果を示すものではなく、観測範囲外への外挿や他都市・他年への適用には別途検証が必要です。

## 公開対象

分析用ソース・依存関係・データ取得手順・評価指標のみを公開しています。CSV、生成グラフ、動画、音声、モデルのバイナリ、仮想環境、認証情報はGit管理から除外しています。データのCC BY 4.0はデータに対するライセンスであり、コードに対するライセンス指定ではありません。
