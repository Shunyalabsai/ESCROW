"""Figure: tuned baselines against ESCROW, the knob at its oracle value and after transfer.

paper/figures/fig_baselines.pdf

Three panels, one per stream (two-group, eight-group noisy, Wikipedia infoboxes).
For each of the five baselines the oracle ARI (knob chosen on the test labels) is a
filled ink dot and the transferred ARI (knob tuned on another stream, worse of the
two transfers) is an open ring; a thin grey line joins them, so the drop on transfer
is the length of the line. k-means has no knob: its filled dot is handed the true
number of groups and its open ring picks that number by the silhouette score, as in
the paper's E4 table. ESCROW is a gold rule at its mean ARI over the 20 record orders of E5,
with a light gold band from the minimum to the maximum over those orders
labelled at the top right of each panel.

The two marker kinds are explained once, in two lines of key text across the top of
the figure, so no text sits on the data. Abbreviations are expanded in the figure
itself: the y label spells out "adjusted Rand index (ARI)" and the panel titles and
ESCROW labels say "groups" rather than K.

Every number is read from results/e4_first_tier.json at run time, and the values the
paper prints in Table E4 are asserted below so the figure cannot drift from the text.

Run:  python3 code/experiments/fig_baselines.py
      (needs matplotlib; set FIG_PREVIEW_DIR to also write a PNG preview there)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "paper", "figures")
PREVIEW = os.environ.get("FIG_PREVIEW_DIR")

# ---- style (shared with make_figures.py) ----------------------------------- #
GOLD = "#B06A1E"
INK = "#17140F"
MUTE = "#8C867B"
PAPER = "#FFFFFF"
TEXT_W = 5.5

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": MUTE,
    "axes.linewidth": 0.6,
    "xtick.color": MUTE,
    "ytick.color": MUTE,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
})

STREAMS = [
    ("twogroup", "two-group"),
    ("planted8", "eight-group noisy"),
    ("wikipedia", "Wikipedia infoboxes"),
]
# (json key, x label printed horizontally under the axis, x shift in data units,
# row). At 7 pt the five names do not fit on one row of a 1.5 in panel, so they
# sit on three rows, each name centred (within 2 pt) under its own column.
BASELINES = [
    ("kmeans", "k-means", -0.12, 0),
    ("hdbscan", "HDBSCAN", 0.0, 1),
    ("agglo_threshold", "agglomerative", 0.10, 0),
    ("cosine_components", "cosine components", 0.0, 2),
    ("dp_means", "DP-means", 0.0, 1),
]
ROW_PT = 8.6            # vertical distance between label rows, points

KEY_LINE_1 = ("filled dot: knob chosen on the test labels.   "
              "open ring: knob tuned on another stream, worse of two transfers.")
KEY_LINE_2 = ("k-means, no knob: filled = handed the true number of groups, "
              "open = number picked by the silhouette score.")


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def pair(stream, key):
    """(oracle ARI, transferred ARI) for one baseline on one stream."""
    if key == "kmeans":
        # no knob: handed the true K, then K chosen without labels by silhouette
        return (stream["kmeans_oracleK"]["ARI"],
                stream["kmeans_silhouette"]["ARI"])
    b = stream[key]
    worst = min(b["transferred_ARI"].values())      # worse of the two transfers
    return b["oracle_ARI"], worst


def main():
    e4 = load("e4_first_tier.json")                 # results/e4_first_tier.json
    with open(os.path.join(RESULTS, "e5_order_dependence.json")) as f5:
        e5 = json.load(f5)["streams"]

    # the numbers Table E4 in the paper prints, asserted so figure and text agree
    expect = {
        "twogroup": {"kmeans": (1.0, 1.0), "hdbscan": (1.0, 0.5121),
                     "agglo_threshold": (1.0, 0.0), "cosine_components": (0.0, 0.0),
                     "dp_means": (0.994, 0.0), "ours": 1.0, "K": 2},
        "planted8": {"kmeans": (0.9977, 0.9977), "hdbscan": (0.9565, 0.0184),
                     "agglo_threshold": (1.0, 0.0), "cosine_components": (0.0, 0.0),
                     "dp_means": (0.9845, 0.0), "ours": 1.0, "K": 8},
        "wikipedia": {"kmeans": (0.9898, 0.502), "hdbscan": (0.4989, 0.0),
                      "agglo_threshold": (0.9645, 0.0002),
                      "cosine_components": (0.4938, 0.0002),
                      "dp_means": (0.7286, 0.0017), "ours": 0.8846, "K": 5},
    }
    for sk, _ in STREAMS:
        for bk, _, _, _ in BASELINES:
            assert pair(e4[sk], bk) == expect[sk][bk], (sk, bk, pair(e4[sk], bk))
        pass

    fig = plt.figure(figsize=(TEXT_W, 2.1))
    gs = fig.add_gridspec(1, 3, wspace=0.10, left=0.115, right=0.99,
                          bottom=0.215, top=0.71)
    # the caption must quote what is drawn: print it, so a changed results file
    # is caught by reading one line rather than by a stale sentence in the paper
    for _k in ("twogroup", "planted8", "wikipedia"):
        _a, _K = e5[_k]["ARI"], e5[_k]["K"]
        print("CAPTION %s: ARI mean %.4f (%.4f to %.4f), K %d to %d over %d orders"
              % (_k, _a["mean"], _a["min"], _a["max"], _K["min"], _K["max"], 20))
    xs = list(range(len(BASELINES)))
    XR = len(BASELINES) - 1 + 0.85                 # right edge of the x range

    for i, (sk, title) in enumerate(STREAMS):
        s = e4[sk]
        ax = fig.add_subplot(gs[0, i])
        sp = e5[sk]
        ours = sp["ARI"]["mean"]
        lo, hi = sp["ARI"]["min"], sp["ARI"]["max"]

        # ESCROW: the gold rule is the mean over 20 record orders, the light
        # band the min to max over those orders (no knob anywhere)
        if hi - lo > 1e-9:
            ax.axhspan(lo, hi, color=GOLD, alpha=0.16, lw=0, zorder=1)
        ax.axhline(ours, color=GOLD, lw=1.5, zorder=2)

        for x, (bk, _, _, _) in zip(xs, BASELINES):
            o, t = pair(s, bk)
            ax.plot([x, x], [t, o], color=MUTE, lw=0.8, zorder=3)
            ax.plot([x], [o], ls="none", marker="o", ms=4.6, mfc=INK,
                    mec=INK, mew=0.8, zorder=5)
            ax.plot([x], [t], ls="none", marker="o", ms=6.2, mfc="none",
                    mec=INK, mew=0.9, zorder=4)

        # ESCROW value above the data at the top right of the panel; where the
        # rule sits lower a thin grey leader joins the label to it
        def fmt(x):
            v = ("%.2f" % x).rstrip("0")
            return v + "0" if v.endswith(".") else v
        # two short lines: three lines of this width ran into the panel on the left
        # the band already shows the range, so the label carries the mean and the
        # node count only: with the range spelled out the third panel's line ran
        # into the panel on its left
        if hi - lo > 1e-9:
            n_txt = "ESCROW, no knob, 20 orders\nARI %s mean, K %d to %d" % (
                fmt(ours), sp["K"]["min"], sp["K"]["max"])
        else:
            n_txt = "ESCROW, no knob, 20 orders\nARI %s every order, K = %d" % (
                fmt(ours), sp["K"]["min"])
        y_txt = 1.14
        ax.text(XR - 0.05, y_txt, n_txt, color=GOLD, fontsize=7,
                ha="right", va="bottom", zorder=6, linespacing=1.15)
        if y_txt - ours > 0.06:
            ax.plot([XR - 0.25, XR - 0.25], [ours, y_txt - 0.01], color=MUTE,
                    lw=0.6, zorder=1)

        ax.set_xlim(-0.5, XR)
        ax.set_ylim(-0.06, 1.62)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xticks(xs)
        ax.set_xticklabels([])
        for x, (_, lab, dx, row) in zip(xs, BASELINES):
            ax.annotate(lab, xy=(x + dx, 0), xycoords=("data", "axes fraction"),
                        xytext=(0, -(5.0 + row * ROW_PT)), textcoords="offset points",
                        ha="center", va="top", fontsize=7, color=INK,
                        annotation_clip=False)
        ax.set_title("%s\n%s records, %d true groups" % (
            title, format(s["n"], ","), s["true_K"]),
            loc="left", fontsize=7.5, pad=3, linespacing=1.15)
        ax.grid(axis="y", color=MUTE, lw=0.4, alpha=0.35)
        if i == 0:
            ax.set_ylabel("adjusted Rand index (ARI)\nagainst the true labels",
                          fontsize=7.5, linespacing=1.15)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # the key: two lines of direct text across the top of the figure, off the data
    fig.text(0.5, 0.985, KEY_LINE_1, fontsize=7, color=INK, ha="center", va="top")
    fig.text(0.5, 0.985 - 7.0 * 1.35 / (2.1 * 72), KEY_LINE_2, fontsize=7,
             color=INK, ha="center", va="top")

    pdf = os.path.join(FIGDIR, "fig_baselines.pdf")
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(pdf, format="pdf")
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, "fig_baselines.png"), dpi=200)
    plt.close(fig)
    return pdf


if __name__ == "__main__":
    print("wrote", main())
