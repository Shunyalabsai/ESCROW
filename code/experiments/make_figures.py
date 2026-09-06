"""The two explanatory figures of the ESCROW paper, drawn from the results files.

Figure 1  paper/figures/fig_escrow.pdf   how a node is born
Figure 2  paper/figures/fig_null.pdf     nothing is born on noise

Every number is read from results/*.json at run time. The only computed curve is
the node price in Figure 1, which comes from the shipped escrow.codes.price, the
same function the engine charges. It reproduces price_at_mint from the results
file at the birth point (36.505 bits at t = 4 members, n = 20 records), which is
asserted below.

Run:  python3 code/experiments/make_figures.py
      (needs matplotlib; set FIG_PREVIEW_DIR to also write PNG previews there)
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
from escrow.codes import price  # noqa: E402  (the shipped price, E12)

RESULTS = os.path.join(ROOT, "results")
FIGDIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGDIR, exist_ok=True)
PREVIEW = os.environ.get("FIG_PREVIEW_DIR")

# ---- style ---------------------------------------------------------------- #
GOLD = "#B06A1E"
INK = "#17140F"
MUTE = "#8C867B"       # ink at reduced weight, for grid, spines and faint lines
PAPER = "#FFFFFF"
TEXT_W = 5.5           # ICLR text width, inches

plt.rcParams.update({
    "pdf.fonttype": 42,           # TrueType, fonts embedded
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 8,
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


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save(fig, stem):
    pdf = os.path.join(FIGDIR, stem + ".pdf")
    fig.savefig(pdf, format="pdf")
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, stem + ".png"), dpi=300)
    plt.close(fig)
    return pdf


# ========================================================================== #
# Figure 1: how a node is born
# ========================================================================== #
def fig_escrow():
    e9 = load("e9_support_curve.json")          # results/e9_support_curve.json
    grid = e9["grid"]
    cell = next(r for r in grid if r["k"] == 4 and r["d"] == 8)
    k, d = cell["k"], cell["d"]
    t_star = cell["t_star"]                     # 4 members
    n_at = cell["n_at_mint"]                    # 20 records
    G_at = cell["G_at_mint"]                    # 38.085 bits held
    P_at = cell["price_at_mint"]                # 36.505 bits price
    rate = cell["rate_bits_per_member"]         # 9.521 bits per member

    # The fixture alternates two groups and the candidate's members arrive at a
    # steady share of the stream, so records per member is n_at / t_star = 5.
    # The price is charged with K = 0 nodes, e = 0 edits and A_n = 2k keys,
    # exactly the arguments the engine used for the first mint.
    A_n = 2 * k
    def price_at(t):
        n = int(round(n_at / t_star * t))
        return price(t, k, n, 0, 0, A_n)
    assert abs(price_at(t_star) - P_at) < 1e-3, (price_at(t_star), P_at)
    assert abs(rate * t_star - G_at) < 0.01, (rate * t_star, G_at)

    T_MAX = 8
    ts = list(range(1, T_MAX + 1))
    account = [rate * t for t in ts]            # realised rate times members
    prices = [price_at(t) for t in ts]

    fig = plt.figure(figsize=(TEXT_W, 2.20))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.42,
                          left=0.085, right=0.985, bottom=0.19, top=0.855)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])

    # ---- left: account against price, one cell -------------------------- #
    ax.plot([0] + ts, [0.0] + account, color=GOLD, lw=1.6, zorder=3)
    ax.plot([0] + ts, [0.0] + account, ls="none", marker="o", ms=3.6,
            mfc=GOLD, mec=PAPER, mew=0.8, zorder=4)
    ax.plot(ts, prices, color=INK, lw=1.2, zorder=2)

    # the crossing: the first member count at which the account covers the price
    ax.plot([t_star], [G_at], marker="o", ms=7.5, mfc=PAPER, mec=GOLD,
            mew=1.4, zorder=5)
    ax.plot([t_star, t_star], [0, P_at], color=MUTE, lw=0.6, ls=(0, (2, 2)),
            zorder=1)

    ax.annotate("node born at t* = %d members\n%.1f bits held, %.1f bits price"
                % (t_star, G_at, P_at),
                xy=(t_star, G_at), xytext=(4.55, 21.0), fontsize=7,
                ha="left", va="center", color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.6,
                                shrinkA=0, shrinkB=4))
    ax.text(T_MAX + 0.15, account[-1], "escrow account\n%.2f bits per member"
            % rate, color=GOLD, fontsize=7, ha="left", va="center")
    ax.text(T_MAX + 0.15, prices[-1], "price of the node\nrises with log n",
            color=INK, fontsize=7, ha="left", va="center")

    ax.set_xlim(0, T_MAX + 3.2)
    ax.set_ylim(0, 80)
    ax.set_xticks(range(0, T_MAX + 1, 2))
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("records in the candidate's account (members t)")
    ax.set_ylabel("bits")
    ax.set_title("one cell: k = %d keys per record, d = %d values per key"
                 % (k, d), loc="left", fontsize=7.5, pad=6)
    ax.grid(axis="y", color=MUTE, lw=0.4, alpha=0.35)

    # ---- right: t* against k over the whole grid ------------------------- #
    ks = sorted({r["k"] for r in grid})
    ds = sorted({r["d"] for r in grid})
    # dodge on the categorical axis so equal integers stay visible
    dodge = {4: -0.10, 8: 0.0, 16: 0.10}
    style = {4: dict(color=INK, marker="^", mfc=PAPER),
             8: dict(color=GOLD, marker="o", mfc=GOLD),
             16: dict(color=INK, marker="s", mfc=INK)}
    xpos = {kv: i for i, kv in enumerate(ks)}
    for dv in ds:
        rows = sorted((r for r in grid if r["d"] == dv), key=lambda r: r["k"])
        xs = [xpos[r["k"]] + dodge[dv] for r in rows]
        ys = [r["t_star"] for r in rows]
        st = style[dv]
        bx.plot(xs, ys, color=st["color"], lw=0.9 if dv == 8 else 0.7,
                alpha=1.0 if dv == 8 else 0.75, zorder=2)
        bx.plot(xs, ys, ls="none", marker=st["marker"], ms=3.8,
                mfc=st["mfc"], mec=st["color"], mew=0.8, zorder=3)
        # labels at the k = 2 end, spread so the near-equal 21 and 22 stay apart
        label_y = {4: ys[0], 8: ys[0] - 1.6, 16: ys[0] + 1.6}[dv]
        bx.text(xs[0] - 0.16, label_y, "d = %d" % dv, fontsize=7,
                color=st["color"], ha="right", va="center")

    bx.plot([xpos[4] + dodge[8]], [cell["t_star"]], marker="o", ms=7.5,
            mfc="none", mec=GOLD, mew=1.2, zorder=4)
    bx.text(xpos[4], cell["t_star"] + 2.6, "the cell\non the left",
            fontsize=7, color=GOLD, ha="center", va="bottom")

    bx.set_xlim(-0.95, len(ks) - 0.55)
    bx.set_ylim(0, 25)
    bx.set_xticks(range(len(ks)))
    bx.set_xticklabels([str(kv) for kv in ks])
    bx.set_yticks([0, 5, 10, 15, 20, 25])
    bx.set_xlabel("keys per record k")
    bx.set_ylabel("members at birth t*")
    bx.set_title("all 12 cells: t* falls as k rises", loc="left",
                 fontsize=7.5, pad=6)
    bx.grid(axis="y", color=MUTE, lw=0.4, alpha=0.35)
    assert e9["monotone_in_k"] is True

    return save(fig, "fig_escrow")


# ========================================================================== #
# Figure 2: nothing is born on noise
# ========================================================================== #
def fig_null():
    ours = load("e8_false_mint_null.json")["plateau"]    # results/e8_false_mint_null.json
    trees = load("e8_tree_arms.json")                    # results/e8_tree_arms.json
    lengths = sorted(int(T) for T in ours)               # 2000, 5000, 10000, 20000
    our_mean = [ours[str(T)]["mean_K"] for T in lengths]  # 0.0, 0.0, 0.0, 0.0
    n_streams = sum(len(ours[str(T)]["all"]) for T in lengths)   # 48
    assert all(v == 0.0 for v in our_mean)
    assert all(ours[str(T)]["max_K"] == 0 for T in lengths)

    def arm(name, delta):
        return [trees[name]["delta=%s T=%d" % (delta, T)]["mean_nodes"]
                for T in lengths]
    efdt_loose = arm("efdt", "0.01")     # 15, 73, 73, 91
    vfdt_loose = arm("vfdt", "0.01")     # 1, 1, 1, 19
    efdt_strict = arm("efdt", "1e-07")   # 1, 1, 1, 1
    vfdt_strict = arm("vfdt", "1e-07")   # 1, 1, 1, 1

    # Drawn narrow (3.63 in) so that it can be set at 0.66 of the 5.5 in text
    # width beside the Ville table and still print its 7 pt labels at 7 pt.
    fig = plt.figure(figsize=(3.63, 2.00))
    ax = fig.add_axes([0.150, 0.185, 0.585, 0.635])

    # the dial set strict: both trees sit at one node, the root alone
    for ys in (efdt_strict, vfdt_strict):
        ax.plot(lengths, ys, color=MUTE, lw=0.8, alpha=0.6, zorder=1)
    ax.plot(lengths, efdt_strict, ls="none", marker="o", ms=2.6, mfc=PAPER,
            mec=MUTE, mew=0.7, alpha=0.8, zorder=1)

    # the usual setting, delta = 0.01
    ax.plot(lengths, efdt_loose, color=INK, lw=1.2, zorder=3)
    ax.plot(lengths, efdt_loose, ls="none", marker="o", ms=3.6, mfc=INK,
            mec=PAPER, mew=0.8, zorder=4)
    ax.plot(lengths, vfdt_loose, color=INK, lw=1.0, ls=(0, (4, 2)), zorder=3)
    ax.plot(lengths, vfdt_loose, ls="none", marker="s", ms=3.4, mfc=PAPER,
            mec=INK, mew=0.9, zorder=4)

    # ours: flat zero, every stream
    ax.plot(lengths, our_mean, color=GOLD, lw=2.0, zorder=5)
    ax.plot(lengths, our_mean, ls="none", marker="o", ms=4.2, mfc=GOLD,
            mec=PAPER, mew=0.9, zorder=6)

    # direct labels at the right edge
    xr = lengths[-1] * 1.09          # clear of the last marker of each curve
    ax.text(xr, efdt_loose[-1], "EFDT, delta = 0.01", color=INK, fontsize=7,
            ha="left", va="center", zorder=9)
    ax.text(xr, vfdt_loose[-1], "VFDT, delta = 0.01", color=INK, fontsize=7,
            ha="left", va="center", zorder=9)
    # The strict line is labelled in the empty band between the two tree curves.
    # The grid would rule through it, so each line sits on its own paper patch,
    # one patch per line so the patch never reaches past the text it backs.
    for _j, _ln in enumerate(("the dial set strict",
                              "(both trees, delta = 1e-7):",
                              "one node, the root")):
        ax.text(4800, 58.0 - 12.7 * _j, _ln, color=MUTE, fontsize=7,
                ha="left", va="top", zorder=6,
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=0.6))
    ax.text(xr, our_mean[-1] - 6.0, "ESCROW: 0 nodes\nin all %d streams" % n_streams,
            color=GOLD, fontsize=7, ha="left", va="center", linespacing=1.25, zorder=9)

    # point labels on the growing curve, so the numbers can be read off
    for T, y in zip(lengths, efdt_loose):
        ax.text(T, y + 4.0, "%d" % y, color=INK, fontsize=7, zorder=8,
                ha="right" if T == lengths[-1] else "center", va="bottom",
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=0.4))
    ax.text(lengths[-1] * 0.97, vfdt_loose[-1] + 4.0, "%d" % vfdt_loose[-1],
            color=INK, fontsize=7, ha="right", va="bottom", zorder=8,
            bbox=dict(facecolor=PAPER, edgecolor="none", pad=0.4))

    ax.set_xscale("log")
    ax.set_xlim(1700, 24000)
    ax.xaxis.set_major_locator(FixedLocator(lengths))
    ax.xaxis.set_major_formatter(FixedFormatter(["2,000", "5,000", "10,000", "20,000"]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_ylim(-26, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("stream length (records, log scale)")
    ax.set_ylabel("nodes created on pure noise")
    ax.set_title("the same iid noise stream, 4 keys, 10 values each;\nevery node is false",
                 loc="left", fontsize=7.5, pad=4, linespacing=1.2)
    ax.grid(axis="y", color=MUTE, lw=0.4, alpha=0.35)

    return save(fig, "fig_null")


if __name__ == "__main__":
    for p in (fig_escrow(), fig_null()):
        print("wrote", p)
