#!/usr/bin/env python3
"""Reproduce the contour-only still from the Manim LossContour2D scene.

The original scene (model_optimization_manim/optimization_explainer.py:
class LossContour2D) draws axes + labels, then finishes drawing the level-set
contours before any sampling dots / trail / markers are introduced.  This
image captures exactly that moment: axes, w1/w2 labels, and the five
elliptical level sets -- no circles.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "assets" / "generated" / "loss_contour2d_contours.png"
MPL_DIR = ROOT / "build" / "matplotlib"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Manim LossContour2D art direction (optimization_explainer.py)
BACKGROUND_COLOR = "#FAF8F5"
AXIS_COLOR = "#53657A"
LOSS_COLOR = "#6657D9"
FONT_SIZE_MEDIUM = 30

# Shared math from the Manim scene
LOSS_CENTER = np.array([0.6, -0.4], dtype=float)
LOSS_MATRIX = np.array([[0.55, 0.18], [0.18, 0.90]], dtype=float)
LEVELS = [0.30, 0.75, 1.40, 2.40, 3.70]

FONT_PATH = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
FONT_NAME = "sans-serif"
if FONT_PATH.exists():
    font_manager.fontManager.addfont(str(FONT_PATH))
    FONT_NAME = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()  # type: ignore[arg-type]

plt.rcParams.update(
    {
        "font.family": FONT_NAME,
        "axes.unicode_minus": False,
        "figure.facecolor": BACKGROUND_COLOR,
        "axes.facecolor": BACKGROUND_COLOR,
        "savefig.facecolor": BACKGROUND_COLOR,
    }
)

# Exact level sets for the quadratic synthetic_loss function, matching
# create_loss_contours(): parameterize each ellipse via eigendecomposition.
eigenvalues, eigenvectors = np.linalg.eigh(LOSS_MATRIX)

fig, ax = plt.subplots(figsize=(16, 9))
fig.subplots_adjust(left=0.24, right=0.76, top=0.84, bottom=0.16)
ax.set_facecolor(BACKGROUND_COLOR)

# Parameter axes: x_range/y_range [-3.3, 3.3], tips=False, include_numbers=False
ax.set_xlim(-3.3, 3.3)
ax.set_ylim(-3.3, 3.3)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# Center cross axes as in Manim's Axes (no arrow tips)
ax.axhline(0, color=AXIS_COLOR, linewidth=1.8, zorder=1)
ax.axvline(0, color=AXIS_COLOR, linewidth=1.8, zorder=1)

# Five elliptical level sets (contours completed, no dots/trail yet)
for index, level in enumerate(LEVELS):
    theta = np.linspace(0, 2 * np.pi, 121)
    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])
    radii = np.sqrt(level / eigenvalues)[:, None]
    parameter_points = LOSS_CENTER[:, None] + eigenvectors @ (radii * unit_circle)
    ax.plot(
        parameter_points[0],
        parameter_points[1],
        color=LOSS_COLOR,
        linewidth=2.2,
        alpha=0.92 - 0.10 * index,
        zorder=2,
    )

# Axis labels placed at the axis ends, like Manim get_axis_labels()
ax.text(3.45, 0.0, "$w_1$", fontsize=FONT_SIZE_MEDIUM, color=AXIS_COLOR,
        ha="left", va="center", zorder=3)
ax.text(0.0, 3.45, "$w_2$", fontsize=FONT_SIZE_MEDIUM, color=AXIS_COLOR,
        ha="center", va="bottom", zorder=3)

# Manim create_parameter_axes(): x_length=6.2, y_length=5.8 with equal data
# ranges, so the display is slightly compressed vertically.
ax.set_aspect(5.8 / 6.2)

fig.savefig(str(OUT_PATH), dpi=120, facecolor=BACKGROUND_COLOR)
print(f"Saved: {OUT_PATH}")