#!/usr/bin/env python3
"""Generate an exploratory-data-analysis gallery from the original bike data."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "SeoulBikeData.csv"
OUT_DIR = ROOT / "gallery" / "eda"
MPL_DIR = ROOT / "build" / "matplotlib"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402


BG = "#FAF8F5"
PANEL = "#FFFFFF"
TEXT = "#0F172A"
MUTED = "#475569"
GRID = "#CBD5E1"
BLUE = "#2563EB"
CYAN = "#0284C7"
ORANGE = "#EA580C"
RED = "#E11D48"
GREEN = "#16A34A"
PURPLE = "#7C3AED"
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
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": GRID,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.alpha": 0.55,
    }
)

ITEMS: list[dict[str, str | int]] = []


def load_original_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_CSV, encoding="latin1")
    df = df.rename(
        columns={
            "Rented Bike Count": "count",
            "Hour": "hour",
            "Temperature(°C)": "temperature",
            "Humidity(%)": "humidity",
            "Wind speed (m/s)": "wind_speed",
            "Visibility (10m)": "visibility",
            "Dew point temperature(°C)": "dew_point",
            "Solar Radiation (MJ/m2)": "solar_radiation",
            "Rainfall(mm)": "rainfall",
            "Snowfall (cm)": "snowfall",
            "Seasons": "season",
            "Holiday": "holiday",
            "Functioning Day": "functioning_day",
        }
    )
    date = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["timestamp"] = date + pd.to_timedelta(df["hour"], unit="h")
    return df.sort_values("timestamp").reset_index(drop=True)


def functioning_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["functioning_day"].eq("Yes")].copy()


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)


def title_block(fig: plt.Figure, title: str, subtitle: str) -> None:
    fig.text(0.06, 0.94, title, fontsize=24, fontweight="bold", color=TEXT)
    fig.text(0.06, 0.895, subtitle, fontsize=12.5, color=MUTED)


def save(fig: plt.Figure, priority: int, slug: str, title: str, purpose: str) -> None:
    filename = f"{priority:02d}_{slug}.png"
    fig.savefig(OUT_DIR / filename, dpi=120, facecolor=BG)
    plt.close(fig)
    ITEMS.append({"priority": priority, "file": filename, "title": title, "purpose": purpose})


def scatter_with_binned_mean(ax: plt.Axes, df: pd.DataFrame, x: str, color: str) -> None:
    sample = df.sample(min(4200, len(df)), random_state=20260809)
    ax.scatter(sample[x], sample["count"], s=11, alpha=0.12, color=color, edgecolors="none")
    try:
        bins = pd.qcut(df[x], q=14, duplicates="drop")
    except ValueError:
        bins = pd.cut(df[x], bins=10, duplicates="drop")
    means = df.assign(_bin=bins).groupby("_bin", observed=True).agg(
        x_mean=(x, "mean"), count_mean=("count", "mean")
    )
    ax.plot(means["x_mean"], means["count_mean"], color=YELLOW, linewidth=3, marker="o", markersize=4)
    style_axis(ax)


def plot_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.82, bottom=0.13)
    title_block(fig, "利用回数の分布", "元データ全8,760時間。赤は利用0回、破線は99パーセンタイル")
    upper = float(df["count"].quantile(0.995))
    _, _, patches = ax.hist(df["count"].clip(upper=upper), bins=55, color=CYAN, alpha=0.82, edgecolor=BG)
    if patches:
        patches[0].set_facecolor(RED)
    zero_count = int(df["count"].eq(0).sum())
    p99 = float(df["count"].quantile(0.99))
    ax.axvline(p99, color=RED, linewidth=2.5, linestyle="--", label=f"99%点: {p99:,.0f}回")
    ax.text(0.68, 0.72, f"0回の記録: {zero_count:,}時間\n最大: {df['count'].max():,}回", transform=ax.transAxes,
            fontsize=17, color=TEXT, bbox={"boxstyle": "round,pad=0.7", "facecolor": PANEL, "edgecolor": GRID})
    ax.set_xlabel("1時間あたり利用回数（上位0.5%は表示範囲でクリップ）")
    ax.set_ylabel("時間数")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, 1, "target_histogram", "利用回数のヒストグラム", "分布・ゼロ件・外れ値候補を確認")


def plot_target_timeseries(df: pd.DataFrame) -> None:
    daily = df.set_index("timestamp")["count"].resample("D").sum()
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.82, bottom=0.14)
    title_block(fig, "利用回数の時系列", "日合計と7日移動平均。年間の季節性と短期変動を見る")
    ax.plot(daily.index, daily, color=CYAN, alpha=0.38, linewidth=1.4, label="日合計")
    ax.plot(daily.index, daily.rolling(7, center=True).mean(), color=BLUE, linewidth=3, label="7日移動平均")
    ax.fill_between(daily.index, daily, color=CYAN, alpha=0.08)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("1日の利用回数")
    ax.legend(frameon=False)
    style_axis(ax)
    save(fig, 1, "target_timeseries", "利用回数の時系列", "周期性・トレンド・急変日を確認")


def plot_hour(df: pd.DataFrame) -> None:
    means = functioning_rows(df).groupby("hour")["count"].mean()
    peak = int(means.idxmax())
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.82, bottom=0.13)
    title_block(fig, "時刻別の平均利用回数", "稼働時間のみ。通勤・通学時間帯のピークを確認")
    colors = [RED if hour == peak else CYAN for hour in means.index]
    ax.bar(means.index, means, color=colors, width=0.78)
    ax.annotate(f"最大: {peak}時\n{means.loc[peak]:,.0f}回", (peak, means.loc[peak]), (peak - 4, means.max() * 1.08),
                fontsize=16, color=RED, arrowprops={"arrowstyle": "->", "color": RED, "lw": 2})
    ax.set_xticks(range(24))
    ax.set_xlabel("時刻")
    ax.set_ylabel("平均利用回数 / 時間")
    style_axis(ax)
    save(fig, 2, "hourly_mean", "時刻別平均利用回数", "通勤・通学時間帯のピークを確認")


def plot_temperature(df: pd.DataFrame) -> None:
    active = functioning_rows(df)
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.82, bottom=0.13)
    title_block(fig, "気温と利用回数", "稼働時間のみ。点は各時間、黄線は気温帯ごとの平均")
    scatter_with_binned_mean(ax, active, "temperature", BLUE)
    ax.set_xlabel("気温（℃）")
    ax.set_ylabel("利用回数 / 時間")
    save(fig, 3, "temperature_scatter", "気温の散布図", "気温と需要の非線形な関係を確認")


def plot_rainfall(df: pd.DataFrame) -> None:
    active = functioning_rows(df)
    labels = ["0", "0超–1", "1超–5", "5超–10", "10超"]
    active["rain_band"] = pd.cut(active["rainfall"], bins=[-0.001, 0, 1, 5, 10, np.inf], labels=labels)
    summary = active.groupby("rain_band", observed=False)["count"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.82, bottom=0.15)
    title_block(fig, "降雨量別の平均利用回数", "稼働時間のみ。棒上の n は各降雨帯の記録時間数")
    bars = ax.bar(labels, summary["mean"], color=[CYAN, BLUE, PURPLE, ORANGE, RED], width=0.68)
    for bar, n in zip(bars, summary["size"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 18, f"n={int(n):,}", ha="center", fontsize=12, color=MUTED)
    ax.set_xlabel("1時間の降雨量（mm）")
    ax.set_ylabel("平均利用回数 / 時間")
    style_axis(ax)
    save(fig, 4, "rainfall_bands", "降雨量別利用回数", "雨量が増えると需要が急減するか確認")


def plot_box(df: pd.DataFrame, column: str, order: list[str], labels: list[str], priority: int, slug: str, title: str, purpose: str) -> None:
    active = functioning_rows(df)
    values = [active.loc[active[column].eq(category), "count"].to_numpy() for category in order]
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.09, right=0.95, top=0.82, bottom=0.14)
    title_block(fig, title, "稼働時間のみ。箱は中央50%、中央線は中央値、点は外れ値候補")
    box = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=True,
                     flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.18, "markeredgecolor": MUTED})
    for patch, color in zip(box["boxes"], [CYAN, BLUE, ORANGE, RED][: len(order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.62)
    for median in box["medians"]:
        median.set_color(TEXT)
        median.set_linewidth(2.5)
    ax.set_ylabel("利用回数 / 時間")
    style_axis(ax)
    save(fig, priority, slug, title, purpose)


def plot_functioning(df: pd.DataFrame) -> None:
    summary = df.groupby("functioning_day")["count"].agg(["mean", "size"])
    zero_rate = df.groupby("functioning_day")["count"].apply(lambda s: s.eq(0).mean() * 100)
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.subplots_adjust(left=0.08, right=0.95, top=0.82, bottom=0.15, wspace=0.27)
    title_block(fig, "稼働日と非稼働日の違い", "元データ全件。非稼働日の0件が需要ではなく運用ルールかを確認")
    order = ["Yes", "No"]
    labels = ["稼働日", "非稼働日"]
    axes[0].bar(labels, summary.loc[order, "mean"], color=[CYAN, RED], width=0.62)
    axes[0].set_ylabel("平均利用回数 / 時間")
    axes[0].set_title("平均利用回数", color=TEXT)
    axes[1].bar(labels, zero_rate.loc[order], color=[CYAN, RED], width=0.62)
    axes[1].set_ylabel("利用0回の割合（%）")
    axes[1].set_title("0回の記録割合", color=TEXT)
    for ax in axes:
        style_axis(ax)
    save(fig, 7, "functioning_day", "稼働日別の利用回数", "非稼働日の0件記録を確認")


def plot_small_multiples(df: pd.DataFrame, specs: list[tuple[str, str, str]], priority: int, slug: str, title: str, purpose: str) -> None:
    active = functioning_rows(df)
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.80, bottom=0.15, wspace=0.25)
    title_block(fig, title, "稼働時間のみ。点は各時間、黄線は各特徴量帯の平均")
    colors = [BLUE, PURPLE, GREEN]
    for ax, (column, label, unit), color in zip(axes, specs, colors):
        scatter_with_binned_mean(ax, active, column, color)
        ax.set_xlabel(f"{label}（{unit}）" if unit else label)
        ax.set_title(label, color=TEXT, fontsize=15)
    axes[0].set_ylabel("利用回数 / 時間")
    save(fig, priority, slug, title, purpose)


def plot_correlation(df: pd.DataFrame) -> None:
    active = functioning_rows(df)
    columns = ["count", "hour", "temperature", "humidity", "wind_speed", "visibility", "dew_point", "solar_radiation", "rainfall", "snowfall"]
    labels = ["利用回数", "時刻", "気温", "湿度", "風速", "視程", "露点", "日射", "雨量", "降雪"]
    corr = active[columns].corr()
    fig, ax = plt.subplots(figsize=(16, 9))
    # Keep the square matrix and its colorbar centered as one visual group.
    ax.set_position([0.30, 0.18, 0.36, 0.64])
    title_block(fig, "数値特徴量の相関行列", "稼働時間のみ。絶対値が1に近いほど線形関係が強い")
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=42, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.55 else TEXT)
    cbar_ax = fig.add_axes([0.69, 0.18, 0.022, 0.64])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Pearson相関係数")
    save(fig, 10, "correlation_matrix", "数値特徴量の相関行列", "特徴量間の重複と多重共線性候補を確認")


def write_gallery() -> None:
    (OUT_DIR / "catalog.json").write_text(json.dumps(ITEMS, ensure_ascii=False, indent=2), encoding="utf-8")
    cards = "\n".join(
        f'''<article class="card"><img src="{item["file"]}" alt="{item["title"]}"><div class="copy"><span>優先 {item["priority"]}</span><h2>{item["title"]}</h2><p>{item["purpose"]}</p></div></article>'''
        for item in ITEMS
    )
    html = f'''<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>レンタサイクル需要予測・EDAギャラリー</title><style>
:root{{--bg:#f6f4f1;--panel:#fff;--text:#0f172a;--muted:#475569;--accent:#2563eb;--border:#dbe2ea}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans",sans-serif}}
header{{max-width:1600px;margin:auto;padding:48px max(4vw,24px) 20px}}h1{{font-size:clamp(28px,4vw,52px);margin:0 0 12px}}header p{{color:var(--muted)}}
main{{max-width:1600px;margin:auto;padding:20px max(4vw,20px) 72px;display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:26px}}
.card{{overflow:hidden;border:1px solid var(--border);border-radius:18px;background:var(--panel);box-shadow:0 14px 35px #0f172a12}}img{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}}
.copy{{padding:18px 22px 24px}}span{{color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.08em}}h2{{margin:7px 0 8px;font-size:21px}}p{{margin:0;color:var(--muted);line-height:1.6}}@media(max-width:600px){{main{{grid-template-columns:1fr}}}}
</style></head><body><header><h1>元データ EDAギャラリー</h1><p>Seoul Bike Sharing Demand・8,760時間。需要分布から相関行列まで、優先順に確認できます。</p></header><main>{cards}</main></body></html>'''
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    df = load_original_data()
    plot_target_distribution(df)
    plot_target_timeseries(df)
    plot_hour(df)
    plot_temperature(df)
    plot_rainfall(df)
    plot_box(df, "season", ["Winter", "Spring", "Summer", "Autumn"], ["冬", "春", "夏", "秋"], 5,
             "season_boxplot", "季節別の利用回数", "季節による需要分布の違いを確認")
    plot_box(df, "holiday", ["No Holiday", "Holiday"], ["通常日", "休日"], 6,
             "holiday_boxplot", "休日別の利用回数", "通常日と休日の需要差を確認")
    plot_functioning(df)
    plot_small_multiples(df, [("humidity", "湿度", "%"), ("wind_speed", "風速", "m/s"), ("solar_radiation", "日射量", "MJ/m2")],
                         8, "weather_supplement", "湿度・風速・日射量", "補助的な天候影響を確認")
    plot_small_multiples(df, [("dew_point", "露点温度", "℃"), ("visibility", "視程", "10m単位"), ("snowfall", "降雪量", "cm")],
                         9, "additional_weather", "露点・視程・降雪", "追加的な説明力を確認")
    plot_correlation(df)
    write_gallery()
    print(f"Generated {len(ITEMS)} EDA figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
