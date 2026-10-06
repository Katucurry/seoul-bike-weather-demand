#!/usr/bin/env python3
"""Train an interpretable Seoul bike-demand model and build a video graph gallery."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "SeoulBikeData.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
GALLERY_DIR = ROOT / "gallery"
MPL_DIR = ROOT / "build" / "matplotlib"

for directory in (PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, GALLERY_DIR, MPL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402


BG = "#FAF8F5"
PANEL = "#FFFFFF"
GRID = "#CBD5E1"
WHITE = "#0F172A"
MUTED = "#475569"
CYAN = "#0284C7"
BLUE = "#2563EB"
PURPLE = "#7C3AED"
MAGENTA = "#DB2777"
RED = "#E11D48"
ORANGE = "#EA580C"
GREEN = "#16A34A"
YELLOW = "#D97706"

FONT_PATH = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    FONT_NAME = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
else:
    FONT_NAME = "sans-serif"

plt.rcParams.update(
    {
        "font.family": FONT_NAME,
        "axes.unicode_minus": False,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "text.color": WHITE,
        "axes.labelcolor": WHITE,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.alpha": 0.5,
        "axes.titleweight": "bold",
    }
)

MODEL_FORMULA = (
    "count ~ C(hour) * C(dow) + temp_c + I(temp_c**2) + humidity_pct "
    "+ wind_ms + solar_mj + rain_any + np.log1p(rain_mm) "
    "+ snow_any + snow_cm + holiday"
)
SPLIT_DATE = pd.Timestamp("2018-10-01")
GALLERY_ITEMS: list[dict[str, Any]] = []


def load_data() -> pd.DataFrame:
    source = pd.read_csv(RAW_CSV, encoding="latin1")
    rename = {
        "Rented Bike Count": "count",
        "Hour": "hour",
        "Temperature(°C)": "temp_c",
        "Humidity(%)": "humidity_pct",
        "Wind speed (m/s)": "wind_ms",
        "Visibility (10m)": "visibility_10m",
        "Dew point temperature(°C)": "dew_point_c",
        "Solar Radiation (MJ/m2)": "solar_mj",
        "Rainfall(mm)": "rain_mm",
        "Snowfall (cm)": "snow_cm",
        "Seasons": "season",
        "Holiday": "holiday_label",
        "Functioning Day": "functioning_label",
    }
    df = source.rename(columns=rename)
    date = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["timestamp"] = date + pd.to_timedelta(df["hour"], unit="h")
    df["date"] = date
    df["dow"] = date.dt.dayofweek
    df["dow_label"] = df["dow"].map(
        {0: "月", 1: "火", 2: "水", 3: "木", 4: "金", 5: "土", 6: "日"}
    )
    df["day_type"] = np.where(df["dow"] < 5, "平日", "週末")
    df["holiday"] = (df["holiday_label"] != "No Holiday").astype(int)
    df["functioning"] = (df["functioning_label"] == "Yes").astype(int)
    df["rain_any"] = (df["rain_mm"] > 0).astype(int)
    df["snow_any"] = (df["snow_cm"] > 0).astype(int)
    df["is_test"] = (df["timestamp"] >= SPLIT_DATE).astype(int)
    return df.sort_values("timestamp").reset_index(drop=True)


def calendar_prediction(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    means = train.groupby(["hour", "dow"])["count"].mean()
    overall = float(train["count"].mean())
    return np.array([means.get((h, d), overall) for h, d in zip(test["hour"], test["dow"])])


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def grouped_permutation_importance(
    model: Any,
    test: pd.DataFrame,
    *,
    repeats: int = 12,
) -> dict[str, float]:
    """Measure unit-independent held-out importance by shuffling raw feature groups."""
    groups = {
        "時刻": ["hour"],
        "曜日": ["dow"],
        "気温": ["temp_c"],
        "湿度": ["humidity_pct"],
        "雨": ["rain_any", "rain_mm"],
        "日射": ["solar_mj"],
        "休日": ["holiday"],
        "風速": ["wind_ms"],
        "雪": ["snow_any", "snow_cm"],
    }
    baseline = mean_absolute_error(test["count"], np.clip(model.predict(test), 0, None))
    rng = np.random.default_rng(20260729)
    importance: dict[str, float] = {}
    for label, columns in groups.items():
        increases: list[float] = []
        for _ in range(repeats):
            shuffled = test.copy()
            order = rng.permutation(len(shuffled))
            shuffled.loc[:, columns] = test[columns].to_numpy()[order]
            prediction = np.clip(model.predict(shuffled), 0, None)
            increases.append(float(mean_absolute_error(test["count"], prediction) - baseline))
        importance[label] = float(np.mean(increases))
    return importance


TITLE_ENABLED = True
TEXTLESS_MODE = False


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    if not TITLE_ENABLED or TEXTLESS_MODE:
        return
    fig.text(0.06, 0.93, title, fontsize=26, fontweight="bold", color=WHITE)
    fig.text(0.06, 0.885, subtitle, fontsize=13, color=MUTED)


def strip_guide_text(fig: plt.Figure) -> None:
    for text in list(fig.texts):
        text.remove()
    for ax in fig.get_axes():
        while len(ax.texts) > 0:
            ax.texts[0].remove()


def save_figure(
    fig: plt.Figure,
    filename: str,
    title: str,
    purpose: str,
    chapter: str,
) -> None:
    if TEXTLESS_MODE:
        strip_guide_text(fig)
    path = GALLERY_DIR / filename
    fig.savefig(path, dpi=100, bbox_inches=None, facecolor=BG)
    plt.close(fig)
    GALLERY_ITEMS.append(
        {
            "file": filename,
            "title": title,
            "purpose": purpose,
            "chapter": chapter,
        }
    )


def plot_data_overview(df: pd.DataFrame) -> None:
    daily = df.groupby("date").agg(count=("count", "sum"), temp=("temp_c", "mean"))
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.08, right=0.92, top=0.82, bottom=0.12)
    title_block(fig, "8,760時間の「街の行動記録」", "2017年12月-2018年11月・ソウルの時間別レンタル実績")
    ax.plot(daily.index, daily["count"], color=CYAN, linewidth=2.2, label="1日の利用回数")
    ax.fill_between(daily.index, daily["count"], color=CYAN, alpha=0.13)
    ax.set_ylabel("1日の利用回数")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=25)
    ax2 = ax.twinx()
    ax2.plot(daily.index, daily["temp"], color=ORANGE, linewidth=1.6, alpha=0.82, label="日平均気温")
    ax2.set_ylabel("日平均気温（℃）", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    ax.text(
        0.01,
        0.96,
        "1行 = 1時間",
        transform=ax.transAxes,
        fontsize=15,
        color=BG,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": YELLOW, "edgecolor": "none"},
    )
    style_axis(ax)
    save_figure(fig, "01_data_overview.png", "8,760時間の行動記録", "データの期間と季節変動を一枚で示す", "データ紹介")


def plot_hour_weekday_heatmap(functioning: pd.DataFrame) -> None:
    pivot = functioning.pivot_table(index="dow", columns="hour", values="count", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.11, right=0.90, top=0.82, bottom=0.14)
    title_block(fig, "時刻と曜日だけで、街の生活リズムが見える", "色が明るいほど、1時間あたりの平均利用回数が多い")
    cmap = LinearSegmentedColormap.from_list("bike", [BG, BLUE, CYAN, YELLOW])
    image = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xticks(range(24))
    ax.set_xticklabels([str(h) for h in range(24)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(["月", "火", "水", "木", "金", "土", "日"])
    ax.set_xlabel("時刻")
    ax.set_ylabel("曜日")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("平均利用回数 / 時間")
    ax.add_patch(plt.Rectangle((17.5, -0.48), 1.0, 4.96, fill=False, edgecolor=RED, linewidth=3))
    ax.text(18, -0.78, "平日18時", color=RED, ha="center", fontsize=14, fontweight="bold")
    save_figure(fig, "02_hour_weekday_heatmap.png", "時刻×曜日ヒートマップ", "通勤ピークと週末型の違いを発見する", "時間帯の推理")


def plot_daytype_profile(functioning: pd.DataFrame) -> None:
    profile = functioning.groupby(["day_type", "hour"])["count"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.82, bottom=0.13)
    title_block(fig, "平日は二つの山、週末は午後に一つの山", "合計の差だけでは見えない「使われ方」の違い")
    ax.plot(profile.index, profile["平日"], color=CYAN, linewidth=4, marker="o", markersize=5, label="平日")
    ax.plot(profile.index, profile["週末"], color=MAGENTA, linewidth=4, marker="o", markersize=5, label="週末")
    ax.fill_between(profile.index, profile["平日"], color=CYAN, alpha=0.08)
    ax.fill_between(profile.index, profile["週末"], color=MAGENTA, alpha=0.07)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("時刻")
    ax.set_ylabel("平均利用回数 / 時間")
    ax.legend(frameon=False, fontsize=14)
    style_axis(ax)
    save_figure(fig, "03_weekday_weekend_profile.png", "平日と週末の時間帯パターン", "会員列がない実データで生活行動を推理する", "時間帯の推理")


def plot_temperature(functioning: pd.DataFrame) -> None:
    sample = functioning.sample(min(4200, len(functioning)), random_state=42)
    bins = np.arange(math.floor(functioning["temp_c"].min() / 3) * 3, functioning["temp_c"].max() + 3, 3)
    grouped = functioning.assign(temp_bin=pd.cut(functioning["temp_c"], bins=bins)).groupby(
        "temp_bin", observed=True
    )["count"].agg(["mean", "size"])
    centers = np.array([interval.mid for interval in grouped.index])
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.82, bottom=0.13)
    title_block(fig, "寒さを抜けるほど増える。ただし猛暑域では鈍る", "薄い点は各時間、太い線は3℃幅の平均。因果ではなく観測上の関係")
    ax.scatter(sample["temp_c"], sample["count"], s=13, color=BLUE, alpha=0.15, edgecolors="none")
    ax.plot(centers, grouped["mean"], color=YELLOW, linewidth=5, marker="o", markersize=7, label="3℃幅の平均")
    hottest = grouped[grouped["size"] >= 30]["mean"].idxmax()
    hot_x = hottest.mid
    hot_y = grouped.loc[hottest, "mean"]
    ax.annotate(
        f"平均が高い帯域\n約{hot_x:.0f}℃",
        (hot_x, hot_y),
        (hot_x - 10, hot_y + 560),
        color=YELLOW,
        fontsize=16,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": YELLOW, "lw": 2},
    )
    ax.set_xlabel("気温（℃）")
    ax.set_ylabel("利用回数 / 時間")
    ax.legend(frameon=False, fontsize=14)
    style_axis(ax)
    save_figure(fig, "04_temperature_relationship.png", "気温と利用回数", "暖かさと猛暑域の鈍化を見せる", "天気の推理")


def plot_rain_effect(functioning: pd.DataFrame) -> None:
    rain_profile = functioning.groupby(["hour", "rain_any"])["count"].mean().unstack()
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.82, bottom=0.13)
    title_block(fig, "雨は、需要の山を一気に削る", "単純平均。青=雨なし、赤=1時間に0mm超の雨")
    ax.plot(rain_profile.index, rain_profile[0], color=CYAN, linewidth=4, label="雨なし")
    ax.plot(rain_profile.index, rain_profile[1], color=RED, linewidth=4, label="雨あり")
    ax.fill_between(rain_profile.index, rain_profile[1], rain_profile[0], color=RED, alpha=0.16)
    ax.text(
        0.72,
        0.77,
        f"全時間平均\n雨なし {functioning.loc[functioning.rain_any.eq(0), 'count'].mean():.0f}回\n雨あり {functioning.loc[functioning.rain_any.eq(1), 'count'].mean():.0f}回",
        transform=ax.transAxes,
        fontsize=17,
        color=WHITE,
        bbox={"boxstyle": "round,pad=0.7", "facecolor": PANEL, "edgecolor": RED},
    )
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("時刻")
    ax.set_ylabel("平均利用回数 / 時間")
    ax.legend(frameon=False, fontsize=14)
    style_axis(ax)
    save_figure(fig, "05_rain_effect_by_hour.png", "雨あり・なしの時間帯比較", "雨の影響を直感的に見せる", "天気の推理")


def plot_functioning_day(df: pd.DataFrame) -> None:
    counts = df.groupby("functioning_label").agg(hours=("count", "size"), average=("count", "mean"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.08, right=0.94, top=0.80, bottom=0.16, wspace=0.26)
    title_block(fig, "ゼロにも、理由がある", "非稼働295時間は「需要ゼロ」ではなく「サービス停止」の記録")
    axes[0].bar(["稼働", "非稼働"], [counts.loc["Yes", "hours"], counts.loc["No", "hours"]], color=[CYAN, RED], width=0.58)
    axes[0].set_ylabel("記録時間数")
    for i, value in enumerate([counts.loc["Yes", "hours"], counts.loc["No", "hours"]]):
        axes[0].text(i, value + 130, f"{int(value):,}時間", ha="center", color=WHITE, fontsize=17, fontweight="bold")
    style_axis(axes[0])
    axes[1].bar(["稼働", "非稼働"], [counts.loc["Yes", "average"], counts.loc["No", "average"]], color=[CYAN, RED], width=0.58)
    axes[1].set_ylabel("平均利用回数 / 時間")
    axes[1].text(0, counts.loc["Yes", "average"] + 35, f"{counts.loc['Yes', 'average']:.0f}回", ha="center", color=WHITE, fontsize=17, fontweight="bold")
    axes[1].text(1, 35, "0回", ha="center", color=WHITE, fontsize=17, fontweight="bold")
    axes[1].text(0.5, 0.82, "学習から除外し、\n運用ルールで0と扱う", transform=axes[1].transAxes, ha="center", color=YELLOW, fontsize=18, fontweight="bold")
    style_axis(axes[1])
    save_figure(fig, "06_functioning_day_caveat.png", "非稼働時間の扱い", "ゼロの意味を確認するデータ品質場面", "データの罠")


def plot_weather_effects(full_model: Any) -> None:
    p = full_model.params
    effects = {
        "気温 +1℃\n（15℃付近）": p["temp_c"] + 2 * 15 * p["I(temp_c ** 2)"],
        "湿度 +10%": p["humidity_pct"] * 10,
        "風速 +1m/s": p["wind_ms"],
        "日射 +1MJ/m2": p["solar_mj"],
        "雨 0→1mm": p["rain_any"] + p["np.log1p(rain_mm)"] * np.log(2),
        "休日": p["holiday"],
    }
    labels = list(effects)
    values = np.array(list(effects.values()))
    colors = [GREEN if value >= 0 else RED for value in values]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.24, right=0.94, top=0.82, bottom=0.13)
    title_block(fig, "天気を「何回増減するか」へ翻訳する", "時刻・曜日・他の天気条件を固定した重回帰の推定値")
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62, alpha=0.92)
    ax.axvline(0, color=WHITE, linewidth=1.3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("利用回数の変化 / 時間")
    for yi, value in zip(y, values):
        ax.text(value + (12 if value >= 0 else -12), yi, f"{value:+.0f}回", va="center", ha="left" if value >= 0 else "right", color=WHITE, fontsize=15, fontweight="bold")
    ax.text(0.02, 0.02, "※相関の記述であり、因果効果ではない", transform=ax.transAxes, color=MUTED, fontsize=12)
    style_axis(ax, grid_axis="x")
    save_figure(fig, "07_weather_marginal_effects.png", "天気の推定効果", "重回帰係数を人間の言葉へ翻訳する", "モデルの推理")


def plot_scenario(full_model: Any) -> None:
    dry = pd.DataFrame(
        [
            {
                "hour": 18,
                "dow": 1,
                "temp_c": 27,
                "humidity_pct": 55,
                "wind_ms": 1.5,
                "solar_mj": 0.2,
                "rain_any": 0,
                "rain_mm": 0,
                "snow_any": 0,
                "snow_cm": 0,
                "holiday": 0,
            }
        ]
    )
    wet = dry.copy()
    wet.loc[0, ["rain_any", "rain_mm"]] = [1, 1]
    values = [float(full_model.predict(dry).iloc[0]), float(full_model.predict(wet).iloc[0])]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.12, right=0.94, top=0.80, bottom=0.15)
    title_block(fig, "火曜18時・27℃。雨だけ変えたら？", "湿度55%、風速1.5m/s、日射0.2MJ/m2、通常営業という同じ条件")
    bars = ax.bar(["雨なし", "雨 1mm"], values, color=[CYAN, RED], width=0.48)
    ax.set_ylabel("予測利用回数 / 時間")
    ax.set_ylim(0, max(values) * 1.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 55, f"{value:,.0f}回", ha="center", color=WHITE, fontsize=27, fontweight="bold")
    diff = values[1] - values[0]
    ax.annotate(f"雨で {diff:,.0f}回", xy=(1, values[1]), xytext=(0.55, max(values) * 1.08), ha="center", color=YELLOW, fontsize=22, fontweight="bold", arrowprops={"arrowstyle": "->", "color": YELLOW, "lw": 2.5})
    style_axis(ax)
    save_figure(fig, "08_prediction_scenario.png", "27℃・18時の予測シナリオ", "答えだけでなく雨の寄与を見せる", "予測ゲーム")


def plot_model_comparison(
    mean_scores: dict[str, float],
    calendar_scores: dict[str, float],
    model_scores: dict[str, float],
) -> None:
    names = ["毎回平均値", "時刻×曜日", "天気も追加"]
    values = [mean_scores["mae"], calendar_scores["mae"], model_scores["mae"]]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.11, right=0.94, top=0.80, bottom=0.16)
    title_block(fig, "予測は、何も考えない答えより賢いか？", "2018年10-11月を未来として評価。低いほど良いMAE")
    bars = ax.bar(names, values, color=[MUTED, PURPLE, CYAN], width=0.56)
    ax.set_ylabel("平均絶対誤差（回 / 時間）")
    ax.set_ylim(0, max(values) * 1.22)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 12, f"{value:.0f}回", ha="center", color=WHITE, fontsize=23, fontweight="bold")
    improvement = 100 * (1 - model_scores["mae"] / mean_scores["mae"])
    ax.text(0.64, 0.79, f"平均値より\n誤差 {improvement:.0f}%減", transform=ax.transAxes, color=YELLOW, fontsize=22, fontweight="bold", ha="center", bbox={"boxstyle": "round,pad=0.55", "facecolor": PANEL, "edgecolor": YELLOW})
    style_axis(ax)
    save_figure(fig, "09_model_comparison.png", "ベースライン比較", "モデルが何もしない予測より役立つか確認する", "予測の採点")


def plot_predictions(test: pd.DataFrame) -> None:
    focus = test[test["timestamp"] >= test["timestamp"].max() - pd.Timedelta(days=13)]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.82, bottom=0.15)
    title_block(fig, "未来2か月を隠して、あとから答え合わせ", "代表として最後の14日間を表示")
    ax.plot(focus["timestamp"], focus["count"], color=WHITE, linewidth=2.3, label="実測")
    ax.plot(focus["timestamp"], focus["prediction"], color=CYAN, linewidth=2.3, label="予測")
    ax.fill_between(focus["timestamp"], focus["count"], focus["prediction"], color=PURPLE, alpha=0.10)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax.set_ylabel("利用回数 / 時間")
    ax.legend(frameon=False, fontsize=14)
    style_axis(ax)
    save_figure(fig, "10_actual_vs_predicted.png", "実測と予測の時系列", "モデルが波を追う様子と外れを同時に見せる", "予測の採点")


def plot_prediction_scatter(test: pd.DataFrame, model_scores: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.10, right=0.94, top=0.82, bottom=0.13)
    title_block(fig, "当たる時間も、大きく外れる時間もある", "対角線に近いほど予測が正確")
    ax.scatter(test["prediction"], test["count"], s=24, color=CYAN, alpha=0.28, edgecolors="none")
    limit = max(test["prediction"].max(), test["count"].max())
    ax.plot([0, limit], [0, limit], color=YELLOW, linestyle="--", linewidth=2.2)
    ax.set_xlim(0, limit * 1.03)
    ax.set_ylim(0, limit * 1.03)
    ax.set_xlabel("予測")
    ax.set_ylabel("実測")
    ax.text(
        0.05,
        0.76,
        (
            f"MAE  {model_scores['mae']:.0f}回\n"
            f"RMSE {model_scores['rmse']:.0f}回\n"
            f"R2   {model_scores['r2']:.3f}"
        ),
        transform=ax.transAxes,
        fontsize=19,
        color=WHITE,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": PANEL,
            "edgecolor": CYAN,
        },
    )
    style_axis(ax)
    save_figure(fig, "11_prediction_scatter.png", "予測対実測", "精度を一枚で検査する", "予測の採点")


def plot_worst_misses(test: pd.DataFrame) -> None:
    worst = test.loc[test["residual"].abs().nlargest(8).index].sort_values("residual")
    labels = [ts.strftime("%m/%d %H時") for ts in worst["timestamp"]]
    colors = [RED if value < 0 else GREEN for value in worst["residual"]]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.17, right=0.94, top=0.82, bottom=0.14)
    title_block(fig, "大外れは、次に調べるべき手がかり", "残差 = 実測 - 予測。赤は予測しすぎ、緑は予測不足")
    y = np.arange(len(worst))
    ax.barh(y, worst["residual"], color=colors, height=0.62)
    ax.axvline(0, color=WHITE, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel("残差（回 / 時間）")
    for yi, value in zip(y, worst["residual"]):
        ax.text(value + (18 if value >= 0 else -18), yi, f"{value:+.0f}", va="center", ha="left" if value >= 0 else "right", color=WHITE, fontsize=13, fontweight="bold")
    style_axis(ax, grid_axis="x")
    save_figure(fig, "12_worst_misses.png", "最大誤差の時間", "外れを事件候補として調べる場面", "残差の捜査")


def plot_collinearity(df: pd.DataFrame) -> None:
    corr = float(df[["temp_c", "dew_point_c"]].corr().iloc[0, 1])
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.10, right=0.94, top=0.82, bottom=0.13)
    title_block(fig, "似すぎた情報を、同じ式へ入れない", "気温と露点温度は強く連動。係数の役割分担が不安定になる")
    sample = df.sample(min(4500, len(df)), random_state=7)
    ax.scatter(sample["temp_c"], sample["dew_point_c"], s=18, color=PURPLE, alpha=0.25, edgecolors="none")
    line = np.linspace(df["temp_c"].min(), df["temp_c"].max(), 100)
    z = np.polyfit(df["temp_c"], df["dew_point_c"], 1)
    ax.plot(line, z[0] * line + z[1], color=YELLOW, linewidth=3)
    ax.text(0.06, 0.79, f"相関係数\n{corr:.3f}", transform=ax.transAxes, fontsize=29, fontweight="bold", color=YELLOW, bbox={"boxstyle": "round,pad=0.55", "facecolor": PANEL, "edgecolor": YELLOW})
    ax.set_xlabel("気温（℃）")
    ax.set_ylabel("露点温度（℃）")
    style_axis(ax)
    save_figure(fig, "13_collinearity_warning.png", "気温と露点温度", "似た変数を同時投入する危険を示す", "モデルの注意")


def plot_temperature_range(df: pd.DataFrame) -> None:
    observed_min = float(df["temp_c"].min())
    observed_max = float(df["temp_c"].max())
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.10, right=0.94, top=0.82, bottom=0.15)
    title_block(fig, "モデルは、経験した範囲の外を知らない", "45℃は学習データ外。式が数字を返しても、信頼できるとは限らない")
    ax.hist(df["temp_c"], bins=42, color=CYAN, alpha=0.78, edgecolor=BG)
    ax.axvspan(observed_min, observed_max, color=GREEN, alpha=0.08, label="観測範囲")
    ax.axvspan(observed_max, 47, color=RED, alpha=0.15, label="外挿")
    ax.axvline(observed_max, color=YELLOW, linewidth=3, linestyle="--")
    ax.text(observed_max - 0.5, ax.get_ylim()[1] * 0.88, f"最高 {observed_max:.1f}℃", ha="right", color=YELLOW, fontsize=17, fontweight="bold")
    ax.text(44, ax.get_ylim()[1] * 0.55, "45℃を入力？", ha="center", color=RED, fontsize=22, fontweight="bold")
    ax.set_xlim(min(-20, observed_min - 2), 47)
    ax.set_xlabel("気温（℃）")
    ax.set_ylabel("記録時間数")
    ax.legend(frameon=False, fontsize=14)
    style_axis(ax)
    save_figure(fig, "14_extrapolation_warning.png", "観測温度と外挿", "データ範囲外予測の危険を見せる", "モデルの注意")


def plot_regression_formula(model: Any) -> None:
    p = model.params
    temp_15 = p["temp_c"] + 2 * 15 * p["I(temp_c ** 2)"]
    humidity_10 = p["humidity_pct"] * 10
    rain_1 = p["rain_any"] + p["np.log1p(rain_mm)"] * np.log(2)
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    title_block(
        fig,
        "回帰式は、条件ごとの増減を足し合わせる",
        "学習は元の単位のまま。係数の絶対値を特徴量重要度として比べない",
    )
    ax.text(0.08, 0.71, "予測利用回数 ＝", fontsize=31, fontweight="bold", color=WHITE)
    ax.text(
        0.30,
        0.71,
        "β0 ＋ β[時刻×曜日] ＋ β1×気温 ＋ β2×気温^2",
        fontsize=24,
        fontweight="bold",
        color=CYAN,
    )
    ax.text(
        0.30,
        0.62,
        "＋ β3×湿度 ＋ β4×風速 ＋ β5×日射 ＋ β6×雨 ＋ β7×雪 ＋ β8×休日",
        fontsize=20,
        fontweight="bold",
        color=CYAN,
    )
    cards = [
        ("15℃付近で気温 +1℃", f"{temp_15:+.0f} 回/時間", ORANGE),
        ("湿度 +10ポイント", f"{humidity_10:+.0f} 回/時間", PURPLE),
        ("雨 0mm → 1mm", f"{rain_1:+.0f} 回/時間", RED),
    ]
    for index, (label, value, color) in enumerate(cards):
        x = 0.08 + index * 0.30
        ax.add_patch(
            plt.Rectangle(
                (x, 0.25),
                0.25,
                0.22,
                facecolor=PANEL,
                edgecolor=color,
                linewidth=2.5,
                transform=ax.transAxes,
            )
        )
        ax.text(x + 0.02, 0.40, label, transform=ax.transAxes, fontsize=16, color=MUTED)
        ax.text(
            x + 0.02,
            0.30,
            value,
            transform=ax.transAxes,
            fontsize=25,
            fontweight="bold",
            color=color,
        )
    ax.text(
        0.08,
        0.13,
        "※係数は、ほかの条件を同じにしたときの推定上の差。因果関係の断定ではありません。",
        fontsize=14,
        color=MUTED,
    )
    save_figure(
        fig,
        "15_regression_formula.png",
        "回帰式と主な係数",
        "式の構造・元単位の係数・標準化の有無を説明する",
        "モデルの組み立て",
    )


def plot_feature_importance(importance: dict[str, float]) -> None:
    ordered = sorted(importance.items(), key=lambda item: item[1])
    labels = [item[0] for item in ordered]
    values = np.array([item[1] for item in ordered])
    display_values = np.maximum(values, 0)
    colors = [CYAN if label in {"時刻", "曜日"} else ORANGE for label in labels]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.18, right=0.94, top=0.82, bottom=0.14)
    title_block(
        fig,
        "どの情報を崩すと、予測が最も悪化するか",
        "未来2か月で各特徴量をシャッフル。MAEの増加が大きいほど重要",
    )
    y = np.arange(len(labels))
    ax.barh(y, display_values, color=colors, height=0.64)
    ax.axvline(0, color=WHITE, linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15)
    ax.set_xlabel("シャッフル後のMAE増加（回 / 時間）")
    for yi, (value, display_value) in enumerate(zip(values, display_values)):
        ax.text(
            display_value + 2,
            yi,
            "ほぼ0" if value <= 0 else f"+{value:.0f}",
            va="center",
            color=WHITE,
            fontsize=14,
            fontweight="bold",
        )
    ax.text(
        0.60,
        0.04,
        "単位が違っても比較できる「予測への寄与」",
        transform=ax.transAxes,
        va="bottom",
        fontsize=14,
        color=YELLOW,
    )
    style_axis(ax, grid_axis="x")
    save_figure(
        fig,
        "16_feature_importance.png",
        "テスト期間での特徴量重要度",
        "生の係数ではなく、単位に依存しない重要度を示す",
        "モデルの組み立て",
    )


def write_gallery_index() -> None:
    cards = []
    for item in GALLERY_ITEMS:
        cards.append(
            f"""
            <article class="card">
              <img src="{item['file']}" alt="{item['title']}">
              <div class="copy">
                <span>{item['chapter']}</span>
                <h2>{item['title']}</h2>
                <p>{item['purpose']}</p>
              </div>
            </article>
            """
        )
    html = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>レンタサイクル需要予測・動画グラフギャラリー</title>
<style>
:root {{ color-scheme: dark; --bg:#07111F; --panel:#10263D; --text:#F4F8FC; --muted:#A9BAC9; --cyan:#35D3FF; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans",sans-serif; }}
header {{ padding:56px max(6vw,24px) 26px; max-width:1600px; margin:auto; }}
h1 {{ font-size:clamp(30px,4vw,58px); margin:0 0 14px; }}
header p {{ color:var(--muted); font-size:18px; max-width:900px; line-height:1.8; }}
main {{ max-width:1600px; margin:auto; padding:20px max(4vw,20px) 80px; display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:28px; }}
.card {{ overflow:hidden; border:1px solid #23445E; border-radius:22px; background:var(--panel); box-shadow:0 24px 60px #0005; }}
img {{ width:100%; display:block; aspect-ratio:16/9; object-fit:cover; }}
.copy {{ padding:22px 24px 28px; }}
.copy span {{ color:var(--cyan); font-size:12px; letter-spacing:.14em; }}
h2 {{ margin:8px 0 10px; font-size:23px; }}
.copy p {{ color:var(--muted); line-height:1.7; margin:0; }}
@media (max-width:600px) {{ main {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>動画グラフギャラリー</h1>
  <p>「天気だけで自転車の利用者数は当てられるのか？」を、ソウルの8,760時間の実データで推理するための図版です。全グラフは分析パイプラインから再生成できます。</p>
</header>
<main>{''.join(cards)}</main>
</body>
</html>"""
    (GALLERY_DIR / "index.html").write_text(html, encoding="utf-8")
    (GALLERY_DIR / "catalog.json").write_text(
        json.dumps(GALLERY_ITEMS, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_report(
    df: pd.DataFrame,
    functioning: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    simple_model: Any,
    full_model: Any,
    mean_scores: dict[str, float],
    calendar_scores: dict[str, float],
    model_scores: dict[str, float],
    importance: dict[str, float],
) -> None:
    params = full_model.params
    corr = df[["temp_c", "dew_point_c"]].corr().iloc[0, 1]
    top = test.loc[test["residual"].abs().nlargest(5).index, ["timestamp", "count", "prediction", "residual", "rain_mm", "holiday_label"]]
    worst_lines = "\n".join(
        f"- {row.timestamp:%Y-%m-%d %H:%M}: 実測 {row['count']:.0f}、予測 {row.prediction:.0f}、残差 {row.residual:+.0f}、雨 {row.rain_mm:g}mm、{row.holiday_label}"
        for _, row in top.iterrows()
    )
    report = f"""# 分析レポート

## 結論

ソウルの時間別8,760行から、天気・時刻・曜日を使って1時間あたりの利用回数を予測します。

- 稼働時間: {len(functioning):,}時間
- 非稼働時間: {len(df) - len(functioning):,}時間（すべて利用0）
- 単回帰: 気温1℃上昇と利用 **{simple_model.params['temp_c']:.1f}回/時間増**が関連（R² {simple_model.rsquared:.3f}）
- 単純平均: 雨なし {functioning.loc[functioning.rain_any.eq(0), 'count'].mean():.0f}回/時間、雨あり {functioning.loc[functioning.rain_any.eq(1), 'count'].mean():.0f}回/時間
- 条件固定の重回帰: 0mmから1mmの雨で **{params['rain_any'] + params['np.log1p(rain_mm)'] * np.log(2):.0f}回/時間**
- 条件固定の重回帰: 湿度10%上昇で **{params['humidity_pct'] * 10:.0f}回/時間**
- 平日18時の平均: {functioning.loc[(functioning.dow < 5) & functioning.hour.eq(18), 'count'].mean():.0f}回/時間
- 週末17時の平均: {functioning.loc[(functioning.dow >= 5) & functioning.hour.eq(17), 'count'].mean():.0f}回/時間

## 未来データでの評価

学習を2018-09-30まで、テストを2018-10-01以降に固定しました。非稼働時間は除外しています。

| 方法 | MAE | RMSE | R² |
|---|---:|---:|---:|
| 毎回、学習期間の平均 | {mean_scores['mae']:.0f} | {mean_scores['rmse']:.0f} | {mean_scores['r2']:.3f} |
| 時刻×曜日の平均 | {calendar_scores['mae']:.0f} | {calendar_scores['rmse']:.0f} | {calendar_scores['r2']:.3f} |
| 天気＋時刻×曜日の重回帰 | {model_scores['mae']:.0f} | {model_scores['rmse']:.0f} | {model_scores['r2']:.3f} |

重回帰は平均値ベースラインに対してMAEを {100 * (1 - model_scores['mae'] / mean_scores['mae']):.1f}%削減しました。カレンダーベースラインとの差は小さい一方、R²は {calendar_scores['r2']:.3f} から {model_scores['r2']:.3f} へ改善しています。つまり、時刻の生活パターンが非常に強く、天気はそこから需要を補正する役割です。

## 特徴量の尺度と重要度

- 回帰学習時の特徴量は標準化していません。気温は℃、湿度はパーセントポイント、雨量はmmのままです。
- そのため、生の係数の絶対値を特徴量間で比較して「重要」とは判断しません。
- 重要度は、テスト期間で特徴量群をシャッフルしたときのMAE増加で比較します。

{chr(10).join(f"- {name}: MAE {value:+.1f}回/時間" for name, value in sorted(importance.items(), key=lambda item: item[1], reverse=True))}

## 最大誤差

{worst_lines}

最大誤差には祝日や降雨時間が含まれます。外部イベントを断定できる列はないため、動画では「事件の確定」ではなく「次に調べるべき手がかり」として扱います。

## 解釈上の注意

- 気温と露点温度の相関は {corr:.3f}。両方を同時に投入しません。
- 係数は相関の記述であり、気温や雨の因果効果ではありません。
- 観測気温は {df.temp_c.min():.1f}〜{df.temp_c.max():.1f}℃。45℃の予測は外挿です。
- 雪の係数は降雪時間が限られ、時刻・冬季需要との交絡もあるため、単純な物語にしません。
- 非稼働時間の0は需要の自然発生ではなく、サービス停止を含む運用上の0です。

## データ出典

UCI Machine Learning Repository, “Seoul Bike Sharing Demand”, DOI: https://doi.org/10.24432/C5F62R, CC BY 4.0.
"""
    (REPORTS_DIR / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    df = load_data()
    functioning = df[df["functioning"].eq(1)].copy()
    train = functioning[functioning["timestamp"] < SPLIT_DATE].copy()
    test = functioning[functioning["timestamp"] >= SPLIT_DATE].copy()

    simple_model = smf.ols("count ~ temp_c", data=train).fit()
    eval_model = smf.ols(MODEL_FORMULA, data=train).fit()
    importance = grouped_permutation_importance(eval_model, test)

    test["prediction"] = np.clip(eval_model.predict(test), 0, None)
    test["residual"] = test["count"] - test["prediction"]
    mean_pred = np.repeat(train["count"].mean(), len(test))
    calendar_pred = calendar_prediction(train, test)
    mean_scores = metrics(test["count"], mean_pred)
    calendar_scores = metrics(test["count"], calendar_pred)
    model_scores = metrics(test["count"], test["prediction"].to_numpy())

    export_columns = [
        "timestamp",
        "count",
        "hour",
        "dow",
        "dow_label",
        "day_type",
        "temp_c",
        "humidity_pct",
        "wind_ms",
        "solar_mj",
        "rain_mm",
        "snow_cm",
        "holiday",
        "functioning",
        "is_test",
    ]
    df[export_columns].to_csv(PROCESSED_DIR / "seoul_bike_model_data.csv", index=False)
    test[export_columns + ["prediction", "residual"]].to_csv(
        PROCESSED_DIR / "test_predictions.csv", index=False
    )

    confidence = eval_model.conf_int()
    coef = pd.DataFrame(
        {
            "term": eval_model.params.index,
            "coefficient": eval_model.params.values,
            "std_error": eval_model.bse.values,
            "p_value": eval_model.pvalues.values,
            "ci_low": confidence[0].values,
            "ci_high": confidence[1].values,
        }
    )
    coef.to_csv(MODELS_DIR / "coefficients.csv", index=False)
    (MODELS_DIR / "model_summary.txt").write_text(
        eval_model.summary().as_text(), encoding="utf-8"
    )

    output_metrics = {
        "dataset": {
            "rows": len(df),
            "functioning_rows": len(functioning),
            "non_functioning_rows": len(df) - len(functioning),
            "start": str(df["timestamp"].min()),
            "end": str(df["timestamp"].max()),
        },
        "split": {
            "cutoff": str(SPLIT_DATE),
            "train_rows": len(train),
            "test_rows": len(test),
        },
        "mean_baseline": mean_scores,
        "calendar_baseline": calendar_scores,
        "weather_regression": model_scores,
        "train_r2": float(eval_model.rsquared),
        "simple_temp_coefficient": float(simple_model.params["temp_c"]),
        "temperature_dewpoint_correlation": float(
            df[["temp_c", "dew_point_c"]].corr().iloc[0, 1]
        ),
        "standardized_features": False,
        "importance_method": "grouped permutation importance on held-out future data; MAE increase",
        "permutation_importance_mae_increase": importance,
        "formula": MODEL_FORMULA,
    }
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(output_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def generate_all_plots(output_dir: Path, enable_titles: bool, textless: bool = False) -> None:
        global GALLERY_DIR, TITLE_ENABLED, TEXTLESS_MODE
        GALLERY_DIR = output_dir
        GALLERY_DIR.mkdir(parents=True, exist_ok=True)
        TITLE_ENABLED = enable_titles
        TEXTLESS_MODE = textless

        plot_data_overview(df)
        plot_hour_weekday_heatmap(train)
        plot_daytype_profile(train)
        plot_temperature(train)
        plot_rain_effect(train)
        plot_functioning_day(df)
        plot_collinearity(train)
        plot_weather_effects(eval_model)
        plot_regression_formula(eval_model)
        plot_feature_importance(importance)
        plot_scenario(eval_model)
        plot_model_comparison(mean_scores, calendar_scores, model_scores)
        plot_predictions(test)
        plot_prediction_scatter(test, model_scores)
        plot_worst_misses(test)
        plot_temperature_range(train)

    # Standard gallery plots with titles
    generate_all_plots(ROOT / "gallery", enable_titles=True, textless=False)
    write_gallery_index()

    # Simple clean plots without title blocks
    generate_all_plots(ROOT / "gallery" / "simple", enable_titles=False, textless=False)

    # Textless plots without any text/labels/legends
    generate_all_plots(ROOT / "gallery" / "textless", enable_titles=False, textless=True)

    write_report(
        df,
        functioning,
        train,
        test,
        simple_model,
        eval_model,
        mean_scores,
        calendar_scores,
        model_scores,
        importance,
    )
    print(json.dumps(output_metrics, ensure_ascii=False, indent=2))
    print(f"gallery: {ROOT / 'gallery' / 'index.html'}")
    print(f"simple gallery: {ROOT / 'gallery' / 'simple'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
