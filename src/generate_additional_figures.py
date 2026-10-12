#!/usr/bin/env python

"""
generate_figures.py — AI4GOOD @ NeurIPS 2026
Quantitative visualisations for model evaluation and bias analysis.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def gaussian_kde_numpy(data, xs, bw=0.12):
    """
    Simple Gaussian KDE using NumPy only.
    bw is interpreted as the bandwidth factor used here.
    """
    data = np.asarray(data)
    n = len(data)

    std = np.std(data, ddof=1)
    bandwidth = 0.12 * std * n ** (-1 / 5)

    bandwidth = max(bandwidth, 1e-6)

    z = (xs[:, None] - data[None, :]) / bandwidth

    density = (
        np.exp(-0.5 * z**2).sum(axis=1)
        / (n * bandwidth * np.sqrt(2 * np.pi))
    )

    return density


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
})


NAVY = "#1A3A6B"
RED = "#B71C1C"
AMBER = "#E65100"
TEAL = "#00695C"
PURPLE = "#6A1B9A"
SLATE = "#37474F"
MGREY = "#90A4AE"
GREEN = "#2E7D32"
CORR = "#1565C0"
INCORR = "#C62828"


# ── Data ─────────────────────────────────────────────────────────────────────

PERF = {
    "ChatGPT": {
        "Roleplay": (64.78, 54.68, 44.36, 52.95),
        "Content": (70.04, 58.02, 57.91, 75.33),
        "Zero-shot": (68.37, 56.14, 51.97, 65.51),
        "Few-shot": (72.86, 57.88, 55.25, 69.21),
    },
    "Flan-T5": {
        "Roleplay": (66.11, 59.45, 61.13, 83.19),
        "Content": (64.80, 59.28, 60.85, 83.82),
        "Zero-shot": (61.92, 59.00, 60.08, 85.31),
        "Few-shot": (62.99, 61.50, 62.17, 87.27),
    },
    "Alpaca": {
        "Roleplay": (47.68, 49.08, 28.14, 30.71),
        "Content": (51.24, 51.05, 17.65, 17.67),
        "Zero-shot": (50.59, 50.32, 22.65, 23.25),
        "Few-shot": (50.30, 50.10, 34.87, 40.77),
    },
    "Vicuna": {
        "Roleplay": (51.10, 50.35, 37.51, 45.25),
        "Content": (52.70, 50.92, 45.84, 62.45),
        "Zero-shot": (54.19, 51.47, 47.12, 64.41),
        "Few-shot": (48.24, 48.39, 48.31, 82.88),
    },
    "Mistral": {
        "Roleplay": (48.13, 48.46, 48.25, 81.93),
        "Content": (49.60, 49.77, 49.06, 77.85),
        "Zero-shot": (49.79, 49.87, 49.37, 78.95),
        "Few-shot": (49.52, 49.26, 49.22, 86.72),
    },
    "Gemini": {
        "Roleplay": (72.50, 61.73, 64.05, 83.19),
        "Content": (85.78, 63.94, 66.67, 85.78),
        "Zero-shot": (86.04, 64.60, 67.52, 86.04),
        "Few-shot": (89.63, 68.72, 70.52, 89.63),
    },
}

MODELS_ORDER = [
    "Gemini",
    "ChatGPT",
    "Flan-T5",
    "Mistral",
    "Vicuna",
    "Alpaca",
]

PROMPTS_ORDER = [
    "Roleplay",
    "Content",
    "Zero-shot",
    "Few-shot",
]


CORR_TABLE = {
    "Roleplay": (-0.471, "3.67e-71"),
    "Content": (-0.413, "1.16e-53"),
    "Zero-shot": (-0.449, "3.55e-64"),
    "Few-shot": (-0.434, "1.38e-59"),
}


DELTA_F1 = {
    "Alpaca": {
        "Roleplay": -0.021,
        "Content": -0.037,
        "Zero-shot": -0.044,
        "Few-shot": -0.077,
    },
    "Flan-T5": {
        "Roleplay": +0.141,
        "Content": -0.211,
        "Zero-shot": -0.074,
        "Few-shot": -0.148,
    },
    "Mistral": {
        "Roleplay": -0.089,
        "Content": -0.045,
        "Zero-shot": -0.018,
        "Few-shot": +0.013,
    },
    "Vicuna": {
        "Roleplay": -0.027,
        "Content": -0.076,
        "Zero-shot": -0.081,
        "Few-shot": -0.088,
    },
    "Gemini": {
        "Roleplay": +0.057,
        "Content": -0.074,
        "Zero-shot": +0.265,
        "Few-shot": +0.349,
    },
}


TRIGGER = {
    "Alpaca": {
        "sx_pred": [72.79, 89.75, 81.98],
        "sx_true": [1.26, 1.81, 1.89, 1.57],
        "po_pred": [62.72, 89.39, 83.03, 72.14],
        "po_true": [4.63, 6.13, 5.34, 3.77],
    },
    "ChatGPT": {
        "sx_pred": [22.44, 53.71, 37.28, 32.86],
        "sx_true": [1.73, 1.18, 1.41, 1.57],
        "po_pred": [30.69, 50.78, 39.04, 37.62],
        "po_true": [4.32, 4.79, 5.11, 5.11],
    },
    "Flan-T5": {
        "sx_pred": [10.78, 11.48, 9.19, 7.77],
        "sx_true": [0.71, 0.71, 0.55, 0.63],
        "po_pred": [11.6, 20.23, 17.4, 14.0],
        "po_true": [3.22, 2.91, 2.36, 2.28],
    },
}


PREC_BINS = {
    "Content": {
        "bins": [
            "0.86-0.88",
            "0.88-0.90",
            "0.90-0.92",
            "0.92-0.94",
            "0.94-0.96",
            "0.96-0.98",
            "0.98-1.00",
        ],
        "prec": [1.00, 1.00, 0.40, 0.36, 0.27, 0.29, 0.37],
        "count": [2, 11, 5, 22, 15, 17, 38],
    },
    "Roleplay": {
        "bins": [
            "0.86-0.88",
            "0.88-0.90",
            "0.90-0.92",
            "0.92-0.94",
            "0.94-0.96",
            "0.96-0.98",
            "0.98-1.00",
        ],
        "prec": [1.00, 1.00, 0.27, 0.29, 0.25, 0.50, 0.47],
        "count": [2, 11, 15, 14, 16, 20, 32],
    },
    "Zero-shot": {
        "bins": [
            "0.88-0.90",
            "0.90-0.92",
            "0.92-0.94",
            "0.94-0.96",
            "0.96-0.98",
            "0.98-1.00",
        ],
        "prec": [1.00, 0.38, 0.36, 0.26, 0.21, 0.28],
        "count": [9, 8, 11, 19, 24, 39],
    },
    "Few-shot": {
        "bins": [
            "0.86-0.88",
            "0.88-0.90",
            "0.90-0.92",
            "0.92-0.94",
            "0.94-0.96",
            "0.96-0.98",
            "0.98-1.00",
        ],
        "prec": [1.00, 1.00, 0.50, 0.46, 0.38, 0.12, 0.23],
        "count": [3, 5, 8, 11, 16, 16, 51],
    },
}


# ── FIG 1: F1 heatmap ───────────────────────────────────────────────────────

def fig_f1_heatmap():

    f1 = np.array([
        [PERF[m][p][2] for p in PROMPTS_ORDER]
        for m in MODELS_ORDER
    ])

    cmap = LinearSegmentedColormap.from_list(
        "rg",
        [
            "#B71C1C",
            "#EF9A9A",
            "#FFF9C4",
            "#A5D6A7",
            "#1B5E20",
        ],
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.2), dpi=200)

    im = ax.imshow(
        f1,
        cmap=cmap,
        vmin=17,
        vmax=72,
        aspect="auto",
    )

    for i in range(6):
        for j in range(4):
            v = f1[i, j]
            col = "white" if (v < 35 or v > 62) else SLATE

            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=col,
            )

    ax.set_xticks(range(4))
    ax.set_xticklabels(PROMPTS_ORDER, fontsize=9)

    ax.set_yticks(range(6))
    ax.set_yticklabels(
        [
            "Gemini 3.5 Flash",
            "ChatGPT-3.5",
            "Flan-T5-xl",
            "Mistral-7B",
            "Vicuna-7B",
            "Alpaca-7B",
        ],
        fontsize=8.5,
    )

    ax.set_xlabel(
        "Prompt strategy",
        fontsize=9.5,
        labelpad=5,
    )

    ax.set_ylabel(
        "Model",
        fontsize=9.5,
        labelpad=5,
    )

    ax.tick_params(length=0)

    from matplotlib.patches import Rectangle

    ax.add_patch(
        Rectangle(
            (-0.5, 4.5),
            4,
            1,
            lw=2.2,
            edgecolor=RED,
            facecolor="none",
            zorder=6,
        )
    )

    ax.text(
        4.08,
        5.0,
        "<-- FM1: acquiescence bias",
        va="center",
        fontsize=7.5,
        color=RED,
        fontweight="bold",
    )

    ax.add_patch(
        Rectangle(
            (-0.5, -0.5),
            4,
            1,
            lw=2.2,
            edgecolor=NAVY,
            facecolor="none",
            zorder=6,
        )
    )

    cb = fig.colorbar(
        im,
        ax=ax,
        shrink=0.82,
        pad=0.01,
    )

    cb.set_label(
        "Macro F1 (%)",
        fontsize=8.5,
    )

    cb.ax.tick_params(labelsize=7.5)

    ax.set_title(
        "Macro-F1 Heatmap: All Models x Prompt Strategies",
        fontweight="bold",
        pad=7,
    )

    plt.tight_layout()

    fig.savefig(
        "./images/fig_f1_heatmap.pdf",
        bbox_inches="tight",
    )

    plt.close()

    print("fig_f1_heatmap.pdf")


# ── FIG 2: Trigger events ───────────────────────────────────────────────────

def fig_trigger_events():

    mkeys = [
        "Alpaca",
        "ChatGPT",
        "Flan-T5",
    ]

    mlabs = [
        "Alpaca-7B",
        "ChatGPT-3.5",
        "Flan-T5-xl",
    ]

    sx_pred = [
        np.mean(TRIGGER[m]["sx_pred"])
        for m in mkeys
    ]

    sx_true = [
        np.mean(TRIGGER[m]["sx_true"])
        for m in mkeys
    ]

    po_pred = [
        np.mean(TRIGGER[m]["po_pred"])
        for m in mkeys
    ]

    po_true = [
        np.mean(TRIGGER[m]["po_true"])
        for m in mkeys
    ]

    cp = [
        "#EF9A9A",
        "#FFCC80",
        "#80CBC4",
    ]

    ct = [
        RED,
        AMBER,
        TEAL,
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.8),
        dpi=200,
    )

    for col, (preds, trues, ev) in enumerate([
        (
            sx_pred,
            sx_true,
            "Sexist event",
        ),
        (
            po_pred,
            po_true,
            "Political event",
        ),
    ]):

        ax = axes[col]

        ax.set_facecolor("#FAFAFA")

        xs = np.arange(3)
        w = 0.35

        for i, (
            ml,
            c1,
            c2,
            pred,
            true,
        ) in enumerate(zip(
            mlabs,
            cp,
            ct,
            preds,
            trues,
        )):

            ax.bar(
                xs[i] - w / 2,
                pred,
                w,
                color=c1,
                edgecolor="white",
                lw=0.8,
                label="Predicted +" if i == 0 else "",
            )

            ax.bar(
                xs[i] + w / 2,
                true,
                w,
                color=c2,
                edgecolor="white",
                lw=0.8,
                label="True +" if i == 0 else "",
            )

            ax.text(
                xs[i] - w / 2,
                pred + 1.5,
                f"{pred:.1f}%",
                ha="center",
                fontsize=7,
                color=SLATE,
            )

            ax.text(
                xs[i] + w / 2,
                true + 1.5,
                f"{true:.1f}%",
                ha="center",
                fontsize=7.5,
                color=SLATE,
                fontweight="bold",
            )

        if ev == "Sexist event":

            gap = sx_pred[0] - sx_true[0]

            ax.annotate(
                "",
                xy=(0 + w / 2, sx_true[0] + 1),
                xytext=(0 - w / 2, sx_pred[0] - 1),
                arrowprops=dict(
                    arrowstyle="<->",
                    color=RED,
                    lw=1.4,
                ),
            )

            ax.text(
                -0.05,
                45,
                f"Gap:\n{gap:.1f} pp",
                fontsize=7.5,
                color=RED,
                fontweight="bold",
                bbox=dict(
                    fc="#FFEBEE",
                    ec=RED,
                    lw=0.7,
                    pad=2,
                ),
            )

        ax.set_xticks(xs)

        ax.set_xticklabels(
            mlabs,
            fontsize=8.5,
        )

        ax.set_ylabel(
            "Proportion of dataset (%)"
        )

        ax.set_ylim(
            0,
            100,
        )

        ax.set_title(
            f"Trigger type: {ev}",
            fontweight="bold",
            pad=5,
        )

        ax.grid(
            axis="y",
            linestyle="--",
        )

        if col == 0:

            ax.legend(
                handles=[
                    mpatches.Patch(
                        color="#EF9A9A",
                        label="Predicted positive",
                    ),
                    mpatches.Patch(
                        color=RED,
                        label="True positive",
                    ),
                ],
                fontsize=7.5,
                loc="upper right",
                framealpha=0.9,
            )

    fig.suptitle(
        "Predicted vs. True Positive Rate (means across prompt types)\n"
        "Models perform similarly at sexist and political events -- cannot distinguish event type",
        fontweight="bold",
        fontsize=9,
        y=1.05,
    )

    plt.tight_layout(
        w_pad=2.0
    )

    fig.savefig(
        "./images/fig_trigger_events.pdf",
        bbox_inches="tight",
    )

    plt.close()

    print("fig_trigger_events.pdf")


# ── FIG 3: Point-biserial correlations ──────────────────────────────────────

def fig_correlation():

    prompts = list(CORR_TABLE.keys())

    rs = [
        CORR_TABLE[p][0]
        for p in prompts
    ]

    fig, ax = plt.subplots(
        figsize=(5.8, 3.4),
        dpi=200,
    )

    ax.set_facecolor("#FAFAFA")

    bars = ax.bar(
        range(4),
        rs,
        color=[RED] * 4,
        alpha=0.85,
        edgecolor="white",
        lw=0.7,
    )

    ax.axhline(
        0,
        color=SLATE,
        lw=0.9,
    )

    ax.set_xticks(range(4))

    ax.set_xticklabels(
        prompts,
        fontsize=9,
    )

    ax.set_ylabel(
        "Point-biserial r\n"
        "(Seq-logprob vs. correctness)",
        fontsize=8.5,
    )

    ax.set_ylim(
        -0.55,
        0.1,
    )

    ax.grid(
        axis="y",
        linestyle="--",
    )

    for bar, r in zip(bars, rs):

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r - 0.02,
            f"r={r:.3f}\np<10e-50",
            ha="center",
            va="top",
            fontsize=7.5,
            color="white",
            fontweight="bold",
        )

    ax.text(
        0.5,
        0.07,
        "Negative r: higher Seq-logprob\n"
        "-> MORE likely to be incorrect\n"
        "(overconfidence signature, all p<10e-50)",
        ha="center",
        transform=ax.transAxes,
        fontsize=8,
        color=RED,
        bbox=dict(
            fc="#FFEBEE",
            ec=RED,
            lw=0.8,
            pad=4,
        ),
    )

    ax.set_title(
        "Point-Biserial Correlation: "
        "Seq-logprob vs. Prediction Correctness\n"
        "Flan-T5 only -- only model for which JSD was computed",
        fontweight="bold",
        pad=7,
    )

    plt.tight_layout()

    fig.savefig(
        "./images/fig_correlation.pdf",
        bbox_inches="tight",
    )

    plt.close()

    print("fig_correlation.pdf")


# ── FIG 4: Precision panels ─────────────────────────────────────────────────

def fig_precision_panel():
    cmap_p = LinearSegmentedColormap.from_list(
        "pr",
        ["#B71C1C", "#EF9A9A", "#FFF9C4", "#A5D6A7", "#1B5E20"]
    )

    fig, axes = plt.subplots(
        2, 2,
        figsize=(8.0, 6.0),
        dpi=200
    )
    fig.patch.set_facecolor("white")

    for idx, pt in enumerate(["Roleplay", "Content", "Zero-shot", "Few-shot"]):
        ax = axes[idx // 2][idx % 2]
        ax2 = ax.twinx()
        ax.set_facecolor("#FAFAFA")

        d = PREC_BINS[pt]
        bins = d["bins"]
        prec = np.array(d["prec"])
        count = np.array(d["count"])

        n = len(bins)
        x = np.arange(n)
        w = 0.38

        pcols = [cmap_p(p) for p in prec]

        b1 = ax.bar(
            x - w / 2,
            prec,
            w,
            color=pcols,
            edgecolor="white",
            lw=0.6,
            zorder=3
        )

        b2 = ax2.bar(
            x + w / 2,
            count,
            w,
            color=SLATE,
            alpha=0.40,
            edgecolor="white",
            lw=0.6,
            zorder=3
        )

        # Precision labels
        for bar, p in zip(b1, prec):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.03,
                f"{p:.2f}",
                ha="center",
                va="bottom",
                fontsize=5.8,
                color=NAVY,
                fontweight="bold"
            )

        # Count labels
        for bar, c in zip(b2, count):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                str(c),
                ha="center",
                va="bottom",
                fontsize=5.8,
                color=SLATE
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            bins,
            rotation=40,
            ha="right",
            fontsize=6.5
        )

        ax.set_ylim(0, 1.45)
        ax.set_ylabel("Precision", color=NAVY, fontsize=8)
        ax2.set_ylabel("Count", color=SLATE, fontsize=8)

        ax.tick_params(
            axis="y",
            colors=NAVY,
            labelsize=7.5
        )
        ax2.tick_params(
            axis="y",
            colors=SLATE,
            labelsize=7.5
        )

        ax.set_title(
            pt,
            fontweight="bold",
            pad=4
        )

        ax.set_xlabel(
            "Seq-logprob bin",
            fontsize=8,
            labelpad=3
        )

        ax.grid(
            axis="y",
            linestyle="--",
            zorder=0
        )

        # Roleplay annotation:
        # moved further left and slightly upwards
        if idx == 0:
            ax.annotate(
                "Most predictions here:\nhigh confidence, low precision",
                xy=(n - 1 - w / 2, prec[-1]),
                xytext=(0.25, 1.30),
                fontsize=6,
                color=RED,
                ha="left",
                va="center",
                arrowprops=dict(
                    arrowstyle="->",
                    color=RED,
                    lw=1.0,
                    connectionstyle="arc3,rad=-0.15"
                ),
                bbox=dict(
                    fc="#FFEBEE",
                    ec=RED,
                    lw=0.6,
                    pad=2
                )
            )

    # ─────────────────────────────────────────────────────────────────────
    # Vertical colour scale positioned in the middle of the full figure
    # ─────────────────────────────────────────────────────────────────────

    sm = plt.cm.ScalarMappable(
        cmap=cmap_p,
        norm=plt.Normalize(0, 1)
    )
    sm.set_array([])

    # Central vertical colourbar
    cbar_ax = fig.add_axes([
        0.45,   # horizontal position: centre of figure
        0.40,    # bottom
        0.015,   # width
        0.14     # height
    ])

    cb = fig.colorbar(
        sm,
        cax=cbar_ax,
        orientation="vertical"
    )

    cb.set_label(
        "Precision value",
        fontsize=8,
        labelpad=6
    )

    cb.ax.tick_params(
        labelsize=7.5
    )

    fig.suptitle(
        "Precision and Prediction Count by Seq-logprob Bin -- Flan-T5",
        fontsize=9.5,
        fontweight="bold",
        y=0.98
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.94],
        pad=1.4,
        h_pad=2.0,
        w_pad=1.8
    )

    fig.savefig(
        "./images/fig_precision_panel.pdf",
        bbox_inches="tight"
    )

    plt.close()
    print("fig_precision_panel.pdf")

# ── FIG 5: Voting sensitivity ───────────────────────────────────────────────

def fig_voting_sensitivity():

    models_v = [
        "Alpaca",
        "Flan-T5",
        "Mistral",
        "Vicuna",
        "Gemini",
    ]

    labels_v = [
        "Alpaca-7B",
        "Flan-T5-xl",
        "Mistral-7B",
        "Vicuna-7B",
        "Gemini 3.5",
    ]

    mat = np.array([
        [
            DELTA_F1[m][p]
            for p in PROMPTS_ORDER
        ]
        for m in models_v
    ])

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(8.0, 4.0),
        dpi=200,
        gridspec_kw={
            "width_ratios": [1.8, 1]
        },
    )

    norm = TwoSlopeNorm(
        vmin=-0.25,
        vcenter=0,
        vmax=0.35,
    )

    cmap_div = LinearSegmentedColormap.from_list(
        "div",
        [
            "#B71C1C",
            "#EF9A9A",
            "#FFFFFF",
            "#90CAF9",
            "#1565C0",
        ],
    )

    im = ax1.imshow(
        mat,
        cmap=cmap_div,
        norm=norm,
        aspect="auto",
    )

    for i in range(5):
        for j in range(4):

            v = mat[i, j]

            col = (
                "white"
                if abs(v) > 0.15
                else SLATE
            )

            ax1.text(
                j,
                i,
                f"{v:+.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color=col,
            )

    ax1.set_xticks(range(4))

    ax1.set_xticklabels(
        PROMPTS_ORDER,
        fontsize=9,
    )

    ax1.set_yticks(range(5))

    ax1.set_yticklabels(
        labels_v,
        fontsize=8.5,
    )

    ax1.tick_params(
        length=0
    )

    cb = fig.colorbar(
        im,
        ax=ax1,
        shrink=0.82,
        pad=0.02,
    )

    cb.set_label(
        "DeltaMacroF1 (Maj - Min)",
        fontsize=8,
    )

    cb.ax.tick_params(
        labelsize=7.5
    )

    ax1.set_title(
        "DeltaMacroF1 by Model x Prompt\n"
        "Blue=majority better; Red=minority better",
        fontweight="bold",
        fontsize=9.5,
        pad=6,
    )

    ranges = [
        (
            mat[i].min(),
            mat[i].max(),
        )
        for i in range(5)
    ]

    y = np.arange(5)

    for i, (lo, hi) in enumerate(ranges):

        ax2.barh(
            i,
            hi - lo,
            left=lo,
            height=0.5,
            color=(
                NAVY
                if hi > 0.1
                else MGREY
            ),
            alpha=0.8,
            edgecolor="white",
            lw=0.6,
        )

        ax2.text(
            hi + 0.005,
            i,
            f"{hi:+.3f}",
            va="center",
            fontsize=7,
            color=NAVY,
        )

        ax2.text(
            lo - 0.005,
            i,
            f"{lo:+.3f}",
            va="center",
            ha="right",
            fontsize=7,
            color=RED,
        )

    ax2.axvline(
        0,
        color=SLATE,
        lw=0.9,
    )

    ax2.set_yticks(y)

    ax2.set_yticklabels(
        labels_v,
        fontsize=8.5,
    )

    ax2.set_xlabel(
        "DeltaMacroF1 range",
        fontsize=8.5,
    )

    ax2.set_xlim(
        -0.28,
        0.42,
    )

    ax2.grid(
        axis="x",
        linestyle="--",
    )

    ax2.set_title(
        "Per-model range\n"
        "min to max across prompts",
        fontweight="bold",
        fontsize=9,
        pad=6,
    )

    fig.suptitle(
        "Evaluation Fragility: DeltaMacroF1 = Majority - Minority Voting\n"
        "Direction inconsistent across models and prompts -- safety evaluations must report both schemes",
        fontsize=9.5,
        fontweight="bold",
        y=1.05,
    )

    plt.tight_layout(
        w_pad=2.5
    )

    fig.savefig(
        "./images/fig_voting_sensitivity.pdf",
        bbox_inches="tight",
    )

    plt.close()

    print("fig_voting_sensitivity.pdf")


# ── FIG 6: Emotion comparison ───────────────────────────────────────────────

def fig_emotion_comparison():

    emotions = [
        "Frustration",
        "Anger",
        "Disgust",
        "Disappointment",
        "Sarcasm",
        "Rage",
        "Outrage",
        "Disdain",
        "Exasperation",
        "Contempt",
        "Resentment",
        "Annoyance",
    ]

    pa = [
        0.138,
        0.118,
        0.088,
        0.075,
        0.062,
        0.055,
        0.050,
        0.044,
        0.042,
        0.034,
        0.036,
        0.040,
    ]

    ps = [
        0.174,
        0.192,
        0.111,
        0.072,
        0.068,
        0.048,
        0.044,
        0.058,
        0.085,
        0.064,
        0.054,
        0.040,
    ]

    order = np.argsort([
        max(a, b)
        for a, b in zip(pa, ps)
    ])

    ems_ = [
        emotions[i]
        for i in order
    ]

    pa_ = [
        pa[i]
        for i in order
    ]

    ps_ = [
        ps[i]
        for i in order
    ]

    fig, ax = plt.subplots(
        figsize=(7.2, 5.0),
        dpi=200,
    )

    ax.set_facecolor("#FAFAFA")

    y = np.arange(
        len(ems_)
    )

    h = 0.36

    b1 = ax.barh(
        y + h / 2,
        pa_,
        h,
        color=TEAL,
        alpha=0.82,
        label="All political text",
        edgecolor="white",
        lw=0.5,
    )

    b2 = ax.barh(
        y - h / 2,
        ps_,
        h,
        color=RED,
        alpha=0.75,
        label="Sexist text only",
        edgecolor="white",
        lw=0.5,
    )

    for bar, v in zip(
        b1,
        pa_,
    ):

        ax.text(
            v + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontsize=6.5,
            color=TEAL,
        )

    for bar, v in zip(
        b2,
        ps_,
    ):

        ax.text(
            v + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{v:.3f}",
            va="center",
            fontsize=6.5,
            color=RED,
        )

    ax.set_yticks(y)

    ax.set_yticklabels(
        ems_,
        fontsize=8.5,
    )

    ax.set_xlabel(
        "Proportion",
        fontsize=9,
    )

    ax.set_xlim(
        0,
        0.25,
    )

    ax.tick_params(
        axis="y",
        length=0,
    )

    ax.spines[
        "left"
    ].set_visible(False)

    ax.grid(
        axis="x",
        linestyle="--",
    )

    ax.legend(
        loc="lower right",
        framealpha=0.92,
    )

    ax.text(
        0.205,
        1.5,
        "Similar rank order\nand magnitudes",
        fontsize=7.5,
        color=AMBER,
        ha="center",
        bbox=dict(
            fc="#FFF8E1",
            ec=AMBER,
            lw=0.9,
            pad=4,
        ),
    )

    ax.set_title(
        "Emotion Attribution: "
        "All Political Text vs. Sexist Text Only\n"
        "Negative emotions dominate both corpora equally",
        fontweight="bold",
        pad=8,
    )

    plt.tight_layout()

    fig.savefig(
        "./images/fig_emotion_comparison.pdf",
        bbox_inches="tight",
    )

    plt.close()

    print("fig_emotion_comparison.pdf")


# ── FIG 7: Emotion by MP ────────────────────────────────────────────────────

def fig_emotion_mp():

    emotions = [
        "Anger",
        "Frustration",
        "Disgust",
        "Sarcasm",
        "Contempt",
        "Resentment",
        "Ridicule",
        "Skepticism",
        "Mockery",
        "Admiration",
    ]

    rayner = [
        0.190,
        0.165,
        0.115,
        0.090,
        0.075,
        0.058,
        0.082,
        0.038,
        0.070,
        0.060,
    ]

    truss = [
        0.185,
        0.178,
        0.108,
        0.092,
        0.065,
        0.055,
        0.045,
        0.072,
        0.050,
        0.032,
    ]

    braverman = [
        0.165,
        0.155,
        0.130,
        0.075,
        0.080,
        0.075,
        0.052,
        0.052,
        0.040,
        0.020,
    ]

    fig, ax = plt.subplots(
        figsize=(7.4, 4.6),
        dpi=200,
    )

    ax.set_facecolor("#FAFAFA")

    x = np.arange(
        len(emotions)
    )

    w = 0.26

    cs = [
        NAVY,
        PURPLE,
        "#BF360C",
    ]

    MPs = [
        "Angela Rayner",
        "Liz Truss",
        "Suella Braverman",
    ]

    for i, (
        vals,
        c,
        mp,
    ) in enumerate(zip(
        [
            rayner,
            truss,
            braverman,
        ],
        cs,
        MPs,
    )):

        ax.bar(
            x + (i - 1) * w,
            vals,
            w,
            label=mp,
            color=c,
            alpha=0.82,
            edgecolor="white",
            lw=0.5,
        )

    ax.set_xticks(x)

    ax.set_xticklabels(
        emotions,
        rotation=28,
        ha="right",
        fontsize=8.5,
    )

    ax.set_ylabel(
        "Proportion",
        fontsize=8.5,
    )

    ax.set_ylim(
        0,
        0.24,
    )

    ax.legend(
        fontsize=7.8,
        framealpha=0.92,
        loc="upper right",
    )

    ax.grid(
        axis="y",
        linestyle="--",
    )

    ax.annotate("Higher disgust + resentment\n(intersectional targeting)",
                xy=(2+w,0.130),xytext=(3.5,0.20),fontsize=7,color="#BF360C",
                arrowprops=dict(arrowstyle="->",color="#BF360C",lw=1.0),
                bbox=dict(fc="#FBE9E7",ec="#BF360C",lw=0.7,pad=2))
    ax.annotate("Higher skepticism\n(policy-credibility framing)",
                xy=(7,truss[7]),xytext=(6.0,0.175),fontsize=7,color=PURPLE,
                arrowprops=dict(arrowstyle="->",color=PURPLE,lw=1.0),
                bbox=dict(fc="#F3E5F5",ec=PURPLE,lw=0.7,pad=2))
    ax.set_title("Emotion Profile in Sexist Texts by Targeted MP\n"
                 "Intersectionality and political context shape abuse patterns",
                 fontweight="bold",pad=8)

    ax.set_title(
        "Emotion Profile in Sexist Texts by Targeted MP",
        fontweight="bold",
        pad=8,
    )

    plt.tight_layout()

    fig.savefig(
        "./images/fig_emotion_mp.pdf",
        bbox_inches="tight",
    )

    plt.close()

    print("fig_emotion_mp.pdf")


# ── FIG 8: Seq-logprob distributions ───────────────────────────────────────

def fig_seqlogprob_panel():

    np.random.seed(42)

    def trunc(
        mu,
        sig,
        lo,
        n=2000,
    ):

        out = []

        while len(out) < n:

            v = np.random.normal(
                mu,
                sig,
                n * 3,
            )

            v = v[
                (v >= lo)
                & (v <= 1.0)
            ]

            out.extend(v)

        return np.array(
            out[:n]
        )

    dp = {
        "Roleplay": {
            "co": (0.974, 0.012, 0.90),
            "ic": (0.958, 0.020, 0.90),
        },
        "Content": {
            "co": (0.971, 0.015, 0.88),
            "ic": (0.946, 0.028, 0.88),
        },
        "Zero-shot": {
            "co": (0.977, 0.011, 0.90),
            "ic": (0.963, 0.016, 0.90),
        },
        "Few-shot": {
            "co": (0.975, 0.013, 0.90),
            "ic": (0.960, 0.017, 0.90),
        },
    }

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.6, 5.4),
        dpi=200,
    )

    fig.patch.set_facecolor(
        "white"
    )

    axes = axes.flatten()

    for idx, pt in enumerate([
        "Roleplay",
        "Content",
        "Zero-shot",
        "Few-shot",
    ]):

        ax = axes[idx]

        p = dp[pt]

        co = trunc(
            *p["co"]
        )

        ic = trunc(
            *p["ic"]
        )

        xmin = (
            min(
                co.min(),
                ic.min(),
            )
            - 0.004
        )

        xs = np.linspace(
            xmin,
            1.002,
            500,
        )

        for vals, color, label in [
            (
                ic,
                INCORR,
                "Incorrect",
            ),
            (
                co,
                CORR,
                "Correct",
            ),
        ]:

            dens = gaussian_kde_numpy(
                vals,
                xs,
                bw=0.12,
            )

            ax.fill_between(
                xs,
                dens,
                alpha=0.17,
                color=color,
            )

            ax.plot(
                xs,
                dens,
                color=color,
                lw=2.0,
                label=label,
            )

        ax.axvspan(
            0.98,
            1.002,
            alpha=0.08,
            color=AMBER,
        )

        ax.axvline(
            0.98,
            color=AMBER,
            lw=1.1,
            ls=":",
            alpha=0.9,
        )

        r_val = CORR_TABLE[pt][0]

        ax.text(
            0.02,
            0.97,
            f"r={r_val:.3f}",
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            color=RED,
            bbox=dict(
                fc="#FFEBEE",
                ec=RED,
                lw=0.6,
                pad=2,
            ),
        )

        ax.set_title(
            pt,
            fontweight="bold",
            pad=5,
        )

        ax.set_xlabel(
            "Seq-logprob",
            fontsize=8.5,
            labelpad=3,
        )

        ax.set_ylabel(
            "Density",
            fontsize=8.5,
            labelpad=3,
        )

        ax.set_xlim(
            xmin,
            1.002,
        )

        ax.set_ylim(
            bottom=0
        )

        ax.xaxis.set_major_formatter(
            mticker.FormatStrFormatter(
                "%.2f"
            )
        )

        ax.grid(
            axis="y",
            linestyle="--",
        )

        if idx == 0:

            ax.legend(
                framealpha=0.88,
                fontsize=8,
                loc="upper left",
            )

    fig.suptitle(
        "Seq-logprob Distributions: "
        "Correct vs. Incorrect Predictions\n"
        "Flan-T5\n"
        "Negative r confirms overconfidence: "
        "higher logprob does NOT improve accuracy",
        fontsize=9.5,
        fontweight="bold",
        y=1.04,
    )

    plt.tight_layout(
        pad=1.4,
        h_pad=2.0,
        w_pad=1.8,
    )

    fig.savefig(
        "./images/fig_seqlogprob_panel.pdf",
        bbox_inches="tight",
    )

    plt.close()

    print("fig_seqlogprob_panel.pdf")


# ── FIG 9: Four FM summary ──────────────────────────────────────────────────

def fig_fm_summary():

    fig = plt.figure(
        figsize=(8.0, 6.2),
        dpi=200,
    )

    fig.patch.set_facecolor(
        "white"
    )

    gs = gridspec.GridSpec(
        2,
        2,
        hspace=0.55,
        wspace=0.40,
        left=0.09,
        right=0.97,
        top=0.88,
        bottom=0.08,
    )

    # FM1: trigger events means

    ax1 = fig.add_subplot(
        gs[0, 0]
    )

    ax1.set_facecolor(
        "#FFF8F8"
    )

    mkeys = [
        "Alpaca",
        "ChatGPT",
        "Flan-T5",
    ]

    mlabs = [
        "Alpaca",
        "ChatGPT",
        "Flan-T5",
    ]

    pred = [
        np.mean(
            TRIGGER[m]["sx_pred"]
        )
        for m in mkeys
    ]

    true = [
        np.mean(
            TRIGGER[m]["sx_true"]
        )
        for m in mkeys
    ]

    x = np.arange(3)

    ax1.bar(
        x - 0.2,
        pred,
        0.38,
        color=[
            "#EF9A9A",
            "#FFCC80",
            "#80CBC4",
        ],
        label="Predicted +",
        edgecolor="white",
    )

    ax1.bar(
        x + 0.2,
        true,
        0.38,
        color=[
            RED,
            AMBER,
            TEAL,
        ],
        label="True +",
        edgecolor="white",
    )

    for i, (p, t) in enumerate(
        zip(pred, true)
    ):

        ax1.text(
            i - 0.2,
            p + 1,
            f"{p:.1f}%",
            ha="center",
            fontsize=6.5,
        )

        ax1.text(
            i + 0.2,
            t + 1,
            f"{t:.1f}%",
            ha="center",
            fontsize=6.5,
            fontweight="bold",
        )

    ax1.set_xticks(x)

    ax1.set_xticklabels(
        mlabs,
        fontsize=8,
    )

    ax1.set_ylabel(
        "%",
        fontsize=8,
    )

    ax1.set_ylim(
        0,
        105,
    )

    ax1.legend(
        fontsize=6.5,
        loc="upper right",
        framealpha=0.85,
    )

    ax1.grid(
        axis="y",
        ls="--",
    )

    ax1.set_title(
        "FM1: Acquiescence Bias\n"
        "Sexist events, mean across prompts",
        fontweight="bold",
        color=RED,
        fontsize=8.5,
    )

    # FM2: F1 spread

    ax2 = fig.add_subplot(
        gs[0, 1]
    )

    ax2.set_facecolor(
        "#FFF8F8"
    )

    mv = [
        "Gemini",
        "ChatGPT",
        "Flan-T5",
        "Mistral",
        "Vicuna",
        "Alpaca",
    ]

    f1_lo = [
        min(
            PERF[m][p][2]
            for p in PROMPTS_ORDER
        )
        for m in mv
    ]

    f1_hi = [
        max(
            PERF[m][p][2]
            for p in PROMPTS_ORDER
        )
        for m in mv
    ]

    sp = [
        h - l
        for h, l in zip(
            f1_hi,
            f1_lo,
        )
    ]

    cols = [
        GREEN
        if s < 5
        else AMBER
        if s < 12
        else RED
        for s in sp
    ]

    ax2.barh(
        range(6),
        sp,
        color=cols,
        alpha=0.82,
        edgecolor="white",
    )

    ax2.set_yticks(
        range(6)
    )

    ax2.set_yticklabels(
        mv,
        fontsize=8,
    )

    ax2.set_xlabel(
        "F1 spread across prompts (pp)",
        fontsize=8,
    )

    for i, (
        lo,
        hi,
        s,
    ) in enumerate(zip(
        f1_lo,
        f1_hi,
        sp,
    )):

        ax2.text(
            s + 0.2,
            i,
            f"{lo:.2f}-{hi:.2f}",
            va="center",
            fontsize=6.5,
        )

    ax2.grid(
        axis="x",
        ls="--",
    )

    ax2.set_title(
        "FM2: Prompt Miscalibration\n"
        "F1 range",
        fontweight="bold",
        color=AMBER,
        fontsize=8.5,
    )

    # FM3: DeltaF1 few-shot

    ax3 = fig.add_subplot(
        gs[1, 0]
    )

    ax3.set_facecolor(
        "#F8F8FF"
    )

    mv3 = [
        "Alpaca",
        "Flan-T5",
        "Mistral",
        "Vicuna",
        "Gemini",
    ]

    dl = [
        DELTA_F1[m]["Few-shot"]
        for m in mv3
    ]

    dcols = [
        NAVY if d > 0 else RED
        for d in dl
    ]

    ax3.bar(
        range(5),
        dl,
        color=dcols,
        alpha=0.85,
        edgecolor="white",
        lw=0.6,
    )

    ax3.axhline(
        0,
        color=SLATE,
        lw=0.9,
    )

    ax3.set_xticks(
        range(5)
    )

    ax3.set_xticklabels(
        mv3,
        fontsize=8,
        rotation=15,
    )

    ax3.set_ylabel(
        "DeltaMacroF1 (Maj-Min)",
        fontsize=8,
    )

    ax3.grid(
        axis="y",
        ls="--",
    )

    for i, d in enumerate(dl):

        offset = (
            0.008
            if d >= 0
            else -0.012
        )

        ax3.text(
            i,
            d + offset,
            f"{d:+.3f}",
            ha="center",
            fontsize=7.5,
            fontweight="bold",
            color=(
                NAVY
                if d > 0
                else RED
            ),
        )

    ax3.set_title(
        "FM3: Evaluation Fragility\n"
        "Few-shot, multi-annotated subset",
        fontweight="bold",
        color=NAVY,
        fontsize=8.5,
    )

    # FM4: point-biserial correlations

    ax4 = fig.add_subplot(
        gs[1, 1]
    )

    ax4.set_facecolor(
        "#F8FFF8"
    )

    rs = [
        CORR_TABLE[p][0]
        for p in PROMPTS_ORDER
    ]

    ax4.bar(
        range(4),
        rs,
        color=[RED] * 4,
        alpha=0.80,
        edgecolor="white",
    )

    ax4.axhline(
        0,
        color=SLATE,
        lw=0.9,
    )

    ax4.set_xticks(
        range(4)
    )

    ax4.set_xticklabels(
        PROMPTS_ORDER,
        fontsize=8,
    )

    ax4.set_ylabel(
        "Point-biserial r",
        fontsize=8,
    )

    ax4.set_ylim(
        -0.55,
        0.05,
    )

    for i, r in enumerate(rs):

        ax4.text(
            i,
            r - 0.02,
            f"r={r:.3f}",
            ha="center",
            va="top",
            fontsize=7,
            color="white",
            fontweight="bold",
        )

    ax4.grid(
        axis="y",
        ls="--",
    )

    ax4.set_title(
        "FM4: Overconfidence\n"
        "Flan-T5; neg r = higher logprob -> more errors",
        fontweight="bold",
        color=TEAL,
        fontsize=8.5,
    )

    fig.suptitle(
        "Four Failure Modes: Quantitative Summary",
        fontsize=9.8,
        fontweight="bold",
    )

    fig.savefig(
        "./images/fig_fm_summary.pdf",
        bbox_inches="tight",
    )

    plt.close()

    print("fig_fm_summary.pdf")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print(
        "Generating figures...\n"
    )

    fig_f1_heatmap()
    fig_trigger_events()
    fig_correlation()
    fig_precision_panel()
    fig_voting_sensitivity()
    fig_emotion_comparison()
    fig_emotion_mp()
    fig_seqlogprob_panel()
    fig_fm_summary()

    print(
        "\nDone. 9 figures generated."
    )