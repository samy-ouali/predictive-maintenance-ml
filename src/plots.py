"""Charts saved to reports/figures and shown in the README."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"

# A small, colorblind-safe palette: one accent colour plus neutral greys.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREY = "#8a8985"
TEXT = "#0b0b0b"
MUTED = "#52514e"
SURFACE = "#fcfcfb"


def _style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=TEXT, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, color=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#d6d5d0")
    ax.tick_params(colors=MUTED)
    ax.grid(axis="both", color="#ecebe8", linewidth=0.8)
    ax.set_axisbelow(True)


def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)


def plot_failure_rate(df):
    """Failure rate per product quality type (L = low, M = medium, H = high)."""
    rates = df.groupby("product_type")["failure"].mean().reindex(["L", "M", "H"]) * 100
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(rates.index, rates.values, color=BLUE, width=0.55)
    ax.bar_label(bars, labels=[f"{v:.1f}%" for v in rates.values], color=TEXT, padding=3)
    _style(ax, "Failure rate by product quality type", "Product type", "Failure rate (%)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, rates.max() * 1.3)
    _save(fig, "failure_rate_by_type.png")


def plot_failure_map(df):
    """Torque vs rotational speed: failures cluster at the extremes."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ok = df[df["failure"] == 0]
    ko = df[df["failure"] == 1]
    ax.scatter(ok["rpm"], ok["torque_nm"], s=6, color=GREY, alpha=0.25, label="No failure")
    ax.scatter(ko["rpm"], ko["torque_nm"], s=14, color=ORANGE, edgecolor=SURFACE,
               linewidth=0.6, label="Failure")
    _style(ax, "Failures happen at extreme torque / speed combinations",
           "Rotational speed (rpm)", "Torque (Nm)")
    ax.legend(frameon=False, labelcolor=TEXT)
    _save(fig, "failure_map.png")


def plot_model_comparison(scores):
    """Cross-validated average precision (PR-AUC) for each model."""
    names = list(scores.keys())
    means = [scores[n]["mean"] for n in names]
    stds = [scores[n]["std"] for n in names]
    order = np.argsort(means)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    y = np.arange(len(names))
    ax.barh(y, np.array(means)[order], xerr=np.array(stds)[order], color=BLUE,
            height=0.5, error_kw={"ecolor": MUTED, "capsize": 3})
    ax.set_yticks(y, np.array(names)[order])
    for yi, m, s in zip(y, np.array(means)[order], np.array(stds)[order]):
        ax.text(m + s + 0.02, yi, f"{m:.2f}", va="center", color=TEXT)
    ax.set_xlim(0, 1.1)
    _style(ax, "Model comparison (5-fold CV, PR-AUC, higher is better)", "Average precision")
    ax.grid(axis="y", visible=False)
    _save(fig, "model_comparison.png")


def plot_precision_recall(precision, recall, chosen_p, chosen_r):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(recall, precision, color=BLUE, linewidth=2)
    ax.scatter([chosen_r], [chosen_p], s=60, color=ORANGE, edgecolor=SURFACE,
               linewidth=1.5, zorder=3)
    ax.annotate("chosen threshold", (chosen_r, chosen_p), xytext=(-110, -30),
                textcoords="offset points", color=TEXT,
                arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    _style(ax, "Precision-recall curve (test set)", "Recall", "Precision")
    _save(fig, "precision_recall.png")


def plot_feature_importance(names, importances, stds):
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    y = np.arange(len(names))
    ax.barh(y, np.array(importances)[order], xerr=np.array(stds)[order], color=BLUE,
            height=0.55, error_kw={"ecolor": MUTED, "capsize": 3})
    ax.set_yticks(y, np.array(names)[order])
    _style(ax, "What drives the predictions? (permutation importance)",
           "Drop in PR-AUC when the feature is shuffled")
    ax.grid(axis="y", visible=False)
    _save(fig, "feature_importance.png")


def plot_confusion(cm):
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    ax.imshow(cm, cmap="Blues")
    labels = ["No failure", "Failure"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > vmax / 2 else TEXT)
    ax.set_facecolor(SURFACE)
    fig.set_facecolor(SURFACE)
    ax.set_title("Confusion matrix (test set)", loc="left", color=TEXT, pad=12)
    ax.set_xlabel("Predicted", color=MUTED)
    ax.set_ylabel("Actual", color=MUTED)
    ax.tick_params(colors=MUTED, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, "confusion_matrix.png")
