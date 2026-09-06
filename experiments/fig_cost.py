"""Figure: what it costs to build the graph, tokens and time against records.

paper/figures/fig_cost.pdf

Left: cumulative tokens (prompt plus completion) against records processed for
the four Qwen2.5-14B-Instruct runs on the first 1,000 Lazada listings, measured
chunk by chunk, then a thin dotted linear extension of each run at its own
per-record rate to the full 21,365 record stream. The measured segment is small
on the full-stream axis, so a magnified copy sits in the panel. ESCROW is a flat
gold line at zero tokens.

Right: cumulative wall-clock seconds against records for the same four runs on
one A100, and ESCROW's total for the same 1,000 records on one CPU core. The
hours quoted for the full stream are a projection at the measured rate.

Every number is read from results/ at run time:
  results/llm_gf_runs/llm_lazada_sampled_s{0,1,2}.json, llm_lazada_greedy_s0.json
      chunks[].records, chunks[].tokens_in, chunks[].tokens_out, chunks[].seconds
      tokens_in, tokens_out, wall_s
  results/llm_gf_runs/escrow_lazada_r0.json   wall_s, tokens_in, tokens_out
  results/llm_graph_formation.json            datasets.lazada.total_stream, reading.cost

Run:  python3 code/experiments/fig_cost.py
      (needs matplotlib; set FIG_PREVIEW_DIR to also write a PNG preview there)
"""
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(ROOT, "results")
RUNS = os.path.join(RESULTS, "llm_gf_runs")
FIGDIR = os.path.join(ROOT, "paper", "figures")
PREVIEW = os.environ.get("FIG_PREVIEW_DIR")

# ---- style (shared with make_figures.py) ---------------------------------- #
GOLD = "#B06A1E"
INK = "#17140F"
MUTE = "#8C867B"
PAPER = "#FFFFFF"
TEXT_W = 5.5
FIG_H = 2.0

plt.rcParams.update({
    "pdf.fonttype": 42,
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


def load(path):
    with open(path) as f:
        return json.load(f)


# one visual identity per run: line style and marker, never colour alone.
# The dotted style is reserved for the extrapolated extension.
RUNS_SPEC = [
    ("llm_lazada_sampled_s0", "seed 0", "-", "o"),
    ("llm_lazada_sampled_s1", "sampled, seed 1", (0, (3.5, 1.6)), "s"),
    ("llm_lazada_sampled_s2", "seed 2", (0, (4, 1.4, 1, 1.4)), "^"),
    ("llm_lazada_greedy_s0", "greedy", (0, (7, 2)), "D"),
]
EXT_LS = (0, (1, 2.2))


def cumulative(run):
    recs, toks, secs = [0], [0], [0.0]
    for c in run["chunks"]:
        recs.append(recs[-1] + c["records"])
        toks.append(toks[-1] + c["tokens_in"] + c["tokens_out"])
        secs.append(secs[-1] + c["seconds"])
    return recs, toks, secs


def fmt_m(x):
    return "%.2fM" % (x / 1e6)


def load_runs():
    gf = load(os.path.join(RESULTS, "llm_graph_formation.json"))
    n_full = gf["datasets"]["lazada"]["total_stream"]            # 21365
    cost_text = gf.get("reading", {}).get("cost", "")   # narrative cross-check, optional
    esc = load(os.path.join(RUNS, "escrow_lazada_r0.json"))
    assert esc["tokens_in"] == 0 and esc["tokens_out"] == 0
    esc_s = esc["wall_s"]                                        # 2.4 s after the engine fix

    runs = []
    for stem, label, ls, mk in RUNS_SPEC:
        r = load(os.path.join(RUNS, stem + ".json"))
        assert r["model"] == "Qwen/Qwen2.5-14B-Instruct" and r["chunk_size"] == 20
        recs, toks, secs = cumulative(r)
        n = recs[-1]
        assert n == r["n_records"] == 1000
        total_tok = r["tokens_in"] + r["tokens_out"]
        assert toks[-1] == total_tok, (toks[-1], total_tok)
        assert abs(secs[-1] - r["wall_s"]) < 1.0, (secs[-1], r["wall_s"])
        full_tok = total_tok / n * n_full          # linear, the run's own rate
        full_h = r["wall_s"] / n * n_full / 3600.0
        runs.append(dict(label=label, ls=ls, mk=mk, recs=recs, toks=toks,
                         secs=secs, total_tok=total_tok, wall=r["wall_s"],
                         full_tok=full_tok, full_h=full_h))

    # check the extrapolation against the written reading in the results file

    lo_tok, hi_tok = min(x["full_tok"] for x in runs), max(x["full_tok"] for x in runs)
    lo_h, hi_h = min(x["full_h"] for x in runs), max(x["full_h"] for x in runs)
    if cost_text:                       # cross-check the drawn extrapolation
        m = re.search(r"([\d.]+)M to ([\d.]+)M tokens", cost_text)
        assert m and abs(lo_tok / 1e6 - float(m.group(1))) < 0.01, (lo_tok, cost_text[:80])
        assert abs(hi_tok / 1e6 - float(m.group(2))) < 0.01, (hi_tok, m.group(2))
        m = re.search(r"so ([\d.]+) to ([\d.]+) hours", cost_text)
        assert m and abs(lo_h - float(m.group(1))) < 0.05 and abs(hi_h - float(m.group(2))) < 0.05
    else:
        print("note: no reading.cost in the results file; extrapolation drawn but not cross-checked")
    print("CAPTION cost: %.2fM to %.2fM tokens, %.1f to %.1f hours for %d records; ESCROW %.1f s, 0 tokens"
          % (lo_tok / 1e6, hi_tok / 1e6, lo_h, hi_h, n_full, esc_s))
    return runs, n_full, esc_s, lo_h, hi_h


def draw():
    """Build the figure. Returns (fig, ax, bx, ins) so a checker can measure it."""
    runs, N_FULL, ESC_S, lo_h, hi_h = load_runs()

    fig = plt.figure(figsize=(TEXT_W, FIG_H))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.30,
                          left=0.075, right=0.99, bottom=0.17, top=0.91)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])

    # ---- left: tokens against records, full stream with measured start --- #
    X_MAX = 35000
    Y_TOP = 3.7e6
    for x in runs:
        ax.plot(x["recs"], x["toks"], color=INK, lw=1.3, ls=x["ls"], zorder=3)
        ax.plot([1000, N_FULL], [x["total_tok"], x["full_tok"]], color=INK,
                lw=0.6, ls=EXT_LS, alpha=0.9, zorder=2)
        ax.plot([N_FULL], [x["full_tok"]], ls="none", marker=x["mk"], ms=3.2,
                color=INK, mfc=PAPER, mec=INK, mew=0.7, zorder=4)
    ax.plot([0, N_FULL], [0, 0], color=GOLD, lw=2.0, zorder=5)
    # the label starts where every dotted extension has already climbed past
    # its top, and its top stays at least 4 pt under the inset's tick labels
    ax.text(3900, 0.08e6, "ESCROW: 0 tokens", color=GOLD, fontsize=7,
            ha="left", va="bottom", zorder=6)

    # labels at the right for the extrapolated totals
    by_tok = sorted(runs, key=lambda x: x["full_tok"])
    top = by_tok[-1]
    XL = N_FULL + 950              # about 2.5 pt clear of the markers
    ax.text(XL, top["full_tok"], "%s\n%s" % (top["label"], fmt_m(top["full_tok"])),
            fontsize=7, color=INK, ha="left", va="center", linespacing=1.15)
    others = by_tok[:-1]
    mid = sum(x["full_tok"] for x in others) / len(others)
    lines = ["%s: %s" % (x["label"], fmt_m(x["full_tok"])) for x in sorted(
        others, key=lambda x: -x["full_tok"])]
    ax.text(XL, mid, "\n".join(lines), fontsize=7, color=INK,
            ha="left", va="center", linespacing=1.15)

    # the measured segment marked on the full-stream axis
    box_w, box_h = 1000, 175e3
    ax.add_patch(Rectangle((0, 0), box_w, box_h, fill=False, ec=MUTE, lw=0.6,
                           zorder=6))

    ax.set_xlim(0, X_MAX)
    ax.set_ylim(-0.08e6, Y_TOP)
    ax.set_xticks([0, 10000, 21365])
    ax.set_xticklabels(["0", "10,000", "21,365"])
    ax.set_yticks([0, 1e6, 2e6, 3e6])
    ax.set_yticklabels(["0", "1M", "2M", "3M"])
    ax.set_xlabel("records processed")
    ax.set_ylabel("tokens, prompt plus answer, cumulative")
    ax.set_title("tokens: solid is measured, dotted is extended",
                 loc="left", fontsize=7.5, pad=6)
    ax.grid(axis="y", color=MUTE, lw=0.4, alpha=0.35)

    # magnified copy of the measured segment, in the free lower right. It is
    # raised so its tick labels clear both the gold line and the gold label,
    # and placed far enough right that its y tick labels sit under the lowest
    # dotted extension (which is at 1.74M where the labels start, 2.07M at the
    # inset's left edge). No leader: the box at the origin is the only box.
    INS_X0, INS_Y0, INS_H = 19300, 0.52e6, 1.2e6
    ins = ax.inset_axes([INS_X0, INS_Y0, X_MAX - 500 - INS_X0, INS_H],
                        transform=ax.transData)
    for x in runs:
        ins.plot(x["recs"], x["toks"], color=INK, lw=1.0, ls=x["ls"], zorder=3)
        ins.plot(x["recs"][10::10], x["toks"][10::10], ls="none", marker=x["mk"],
                 ms=2.8, color=INK, mfc=PAPER, mec=INK, mew=0.6, zorder=4,
                 clip_on=False)
    ins.plot([0, 1000], [0, 0], color=GOLD, lw=1.6, zorder=5)
    ins.set_xlim(0, 1045)          # room so the markers at 1,000 are whole
    ins.set_ylim(-5e3, 210e3)      # headroom so the note clears the lines
    ins.set_xticks([0, 500, 1000])
    ins.set_xticklabels(["0", "500", "1,000"], fontsize=7)
    ins.set_yticks([0, 50e3, 100e3, 150e3])
    ins.set_yticklabels(["0", "50k", "100k", "150k"], fontsize=7)
    ins.tick_params(length=2, width=0.5, pad=1.5, colors=MUTE, labelcolor=INK)
    for s in ins.spines.values():
        s.set_visible(True)
        s.set_color(MUTE)
        s.set_linewidth(0.6)
    ins.text(0.04, 0.96, "small box,\nmagnified", transform=ins.transAxes,
             fontsize=7, ha="left", va="top", color=INK, linespacing=1.15)

    # ---- right: seconds against records, the measured 1,000 -------------- #
    for x in runs:
        bx.plot(x["recs"], x["secs"], color=INK, lw=1.2, ls=x["ls"], zorder=3)
        bx.plot(x["recs"][10::10], x["secs"][10::10], ls="none", marker=x["mk"],
                ms=3.0, color=INK, mfc=PAPER, mec=INK, mew=0.7, zorder=4)
    by_wall = sorted(runs, key=lambda x: -x["wall"])
    slow = by_wall[0]
    XR = 1000 + 70                 # clear of the 3 pt markers at 1,000
    # the slowest run's label sits a little below its marker so it stays
    # under the panel title; the axis label carries the unit
    bx.text(XR, slow["wall"] - 45, "%s\n%s" % (slow["label"], "{:,.1f}".format(slow["wall"])),
            fontsize=7, color=INK, ha="left", va="center", linespacing=1.15)
    rest = by_wall[1:]
    mid_s = sum(x["wall"] for x in rest) / len(rest)
    rest_lines = ["%s: %s" % (x["label"], "{:,.1f}".format(x["wall"]))
                  for x in rest]
    bx.text(XR, mid_s - 175, "\n".join(rest_lines), fontsize=7, color=INK,
            ha="left", va="center", linespacing=1.15)

    bx.plot([0, 1000], [0, ESC_S], color=GOLD, lw=1.6, zorder=5)
    bx.plot([1000], [ESC_S], ls="none", marker="o", ms=6.5, mfc=GOLD, mec=PAPER,
            mew=0.9, zorder=6)
    bx.text(XR, ESC_S + 125, "ESCROW: %.2f s\none CPU core" % ESC_S,
            fontsize=7, color=GOLD, ha="left", va="center", linespacing=1.15)

    # the full-stream hours are a projection at the measured rate, worded so
    bx.text(1780, 900, "model on one GPU (A100)\nat this rate the full\nstream of {:,} records\ntakes {:.1f} to {:.1f} hours".format(
        N_FULL, lo_h, hi_h), fontsize=7, color=INK, ha="right", va="top",
        linespacing=1.2)

    bx.set_xlim(0, 1800)
    bx.set_ylim(-50, 1950)
    bx.set_xticks([0, 500, 1000])
    bx.set_xticklabels(["0", "500", "1,000"])
    bx.set_yticks([0, 500, 1000, 1500])
    bx.set_yticklabels(["0", "500", "1,000", "1,500"])
    bx.set_xlabel("records processed")
    bx.set_ylabel("wall-clock seconds, cumulative")
    bx.set_title("time for the same 1,000 records",
                 loc="left", fontsize=7.5, pad=6)
    bx.grid(axis="y", color=MUTE, lw=0.4, alpha=0.35)
    return fig, ax, bx, ins


def main():
    fig, _, _, _ = draw()
    os.makedirs(FIGDIR, exist_ok=True)
    pdf = os.path.join(FIGDIR, "fig_cost.pdf")
    fig.savefig(pdf, format="pdf")
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, "fig_cost.png"), dpi=200)
    plt.close(fig)
    print("wrote", pdf)
    return pdf


if __name__ == "__main__":
    main()
