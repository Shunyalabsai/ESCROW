"""Figure: what happens when there is nothing to find.

Every number is read from results/e41_symmetric.json at run time. The stream is the null generator
of E41, four keys of ten independent uniform values, so every node any method creates is false. One
line per seed for the language model, one flat line for ESCROW.

Run:  python3 code/experiments/fig_abstain.py
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

GOLD = "#B06A1E"
INK = "#17140F"
MUTE = "#8C867B"
PAPER = "#FFFFFF"

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTE, "axes.linewidth": 0.6, "xtick.color": MUTE, "ytick.color": MUTE,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "legend.frameon": False, "savefig.facecolor": PAPER,
    "figure.facecolor": PAPER,
})


def main():
    R = json.load(open(os.path.join(RESULTS, "e41_symmetric.json")))
    rows = R["null"]
    assert all(r["escrow_nodes"] == 0 for r in rows), "the flat line must be measured, not drawn"

    fig = plt.figure(figsize=(3.63, 2.00))
    ax = fig.add_axes([0.150, 0.185, 0.560, 0.660])

    n = None
    for r in rows:
        xs = r["records_after_each_chunk"]
        ys = r["llm_nodes_after_each_chunk"]
        n = xs[-1]
        ax.plot(xs, ys, color=INK, lw=1.1, zorder=3)
        ax.plot(xs[-1:], ys[-1:], ls="none", marker="o", ms=3.4, mfc=INK, mec=PAPER,
                mew=0.8, zorder=4)

    # the ceiling a rule that never declines would reach: one node per record
    ax.plot([0, n], [0, n], color=MUTE, lw=0.8, ls=(0, (2, 2)), zorder=1)
    ax.text(n * 0.52, n * 0.60, "one node per record", color=MUTE, fontsize=7,
            rotation=38, ha="center", va="bottom", zorder=2)

    xs0 = rows[0]["records_after_each_chunk"]
    ax.plot(xs0, [0] * len(xs0), color=GOLD, lw=2.0, zorder=5)
    ax.plot(xs0[-1:], [0], ls="none", marker="o", ms=4.2, mfc=GOLD, mec=PAPER, mew=0.9, zorder=6)

    hi = max(r["llm_nodes"] for r in rows)
    lo = min(r["llm_nodes"] for r in rows)
    xr = n * 1.04
    ax.text(xr, hi, "a language model,\nsame question,\nsame records", color=INK, fontsize=7,
            ha="left", va="center", linespacing=1.25, zorder=9)
    ax.text(xr, max(lo - n * 0.06, n * 0.05), "ESCROW: 0 nodes,\nnothing set", color=GOLD,
            fontsize=7, ha="left", va="center", linespacing=1.25, zorder=9)

    ax.set_xlabel("records seen")
    ax.set_ylabel("nodes created")
    ax.set_xlim(0, n * 1.02)
    ax.set_ylim(-n * 0.03, n * 1.02)
    ax.grid(True, color=MUTE, lw=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    out = os.path.join(FIGDIR, "fig_abstain.pdf")
    fig.savefig(out, bbox_inches=None)
    print("wrote", out, "| final node counts", [r["llm_nodes"] for r in rows], "vs 0")
    if PREVIEW:
        fig.savefig(os.path.join(PREVIEW, "fig_abstain.png"), dpi=200)
    cap = os.path.join(FIGDIR, "fig_abstain.caption.txt")
    open(cap, "w").write(
        "Nodes created against records seen on a stream with no latent structure, so every node "
        "created is false. One line per seed for Qwen2.5-14B-Instruct, greedy, given the same "
        "create-or-attach question on the same records in the same order. ESCROW in gold, zero at "
        "every length. Final counts %s against 0 (E41).\n" % ", ".join(
            str(r["llm_nodes"]) for r in rows))


if __name__ == "__main__":
    main()
