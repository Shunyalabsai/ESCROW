"""Figure: a language model building the same graph on identical records.

paper/figures/fig_llm.pdf, two panels.
Left:  Lazada, first 1,000 raw listings. Node count after every chunk of 20
       records for four Qwen2.5-14B-Instruct runs (three seeds at temperature
       0.7, one greedy), with ESCROW's final count as a single end-point marker.
Right: Wikipedia, 320 infoboxes. Adjusted Rand index against the three gold
       types for the same four model runs and for ESCROW.

Every number is read from results/ at run time:
  results/llm_gf_runs/llm_lazada_sampled_s{0,1,2}.json  chunks[].records, nodes_after
  results/llm_gf_runs/llm_lazada_greedy_s0.json         chunks[].records, nodes_after
  results/llm_gf_runs/escrow_lazada_r{0,1,2}.json       K, tokens_in, tokens_out, wall_s
      (final count only: no per-record node count is stored for ESCROW, so
      the figure shows one end-point marker, never a trajectory)
  results/llm_graph_formation.json
      datasets.lazada.methods.escrow.background_or_unparsed
  results/llm_graph_formation.json
      datasets.wikipedia.methods.{llm_sampled_s0,llm_sampled_s1,llm_sampled_s2,
                                  llm_greedy_s0,escrow}.ari
      datasets.wikipedia.escrow_determinism.identical_assignments
      datasets.lazada.llm_determinism.mean_ari, node_counts
The per-run Wikipedia files hold no ARI, so the per-run values come from the
summary file above (they are per run there, not means).

Run:  python3 code/experiments/fig_llm.py
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

# ---- style, the same as make_figures.py ----------------------------------- #
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


def load(*parts):
    with open(os.path.join(RESULTS, *parts)) as f:
        return json.load(f)


# one visual identity per run, shared by both panels: shape and line style,
# never colour alone
RUNS = [
    ("llm_sampled_s0", "seed 0", dict(marker="o", mfc=INK, ls="-")),
    ("llm_sampled_s1", "seed 1", dict(marker="s", mfc=INK, ls="-")),
    ("llm_sampled_s2", "seed 2", dict(marker="^", mfc=INK, ls="-")),
    ("llm_greedy_s0", "greedy", dict(marker="D", mfc=PAPER, ls=(0, (4, 2)))),
]
# ESCROW's own shape, used in both panels so identity is by shape, not colour
ESCROW_MARKER = dict(marker="*", ms=7.5, mfc=GOLD, mec=PAPER, mew=0.6)
PRINT_W, PRINT_H = TEXT_W, 2.1                    # inches, checked after saving


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    summary = load("llm_graph_formation.json")
    wiki = summary["datasets"]["wikipedia"]
    laz = summary["datasets"]["lazada"]

    # ---- Lazada trajectories ------------------------------------------- #
    traj = {}
    for key, _, _ in RUNS:
        run = load("llm_gf_runs", key.replace("llm_", "llm_lazada_") + ".json")
        chunks = run["chunks"]
        assert len(chunks) == 50 and all(c["records"] == 20 for c in chunks)
        xs, ys, n = [], [], 0
        for c in chunks:
            n += c["records"]
            xs.append(n)
            ys.append(c["nodes_after"])
        assert n == 1000
        traj[key] = (xs, ys)
    finals = [traj[k][1][-1] for k, _, _ in RUNS]
    # the summary's node_counts (24, 78, 28) are the three seeds; greedy is 28
    assert finals[:3] == laz["llm_determinism"]["node_counts"], finals
    assert finals[3] == laz["methods"]["llm_greedy_s0"]["nodes"]

    escrow_runs = [load("llm_gf_runs", "escrow_lazada_r%d.json" % r)
                   for r in range(3)]
    escrow_K = [e["K"] for e in escrow_runs]
    assert escrow_K == laz["escrow_determinism"]["K_per_run"]
    assert len(set(escrow_K)) == 1
    assert all(e["tokens_in"] == 0 and e["tokens_out"] == 0 for e in escrow_runs)
    K = escrow_K[0]                                   # 6
    # Guard: ESCROW's result files carry no node count after each chunk, so
    # nothing here may draw an ESCROW series over n. If a per-record log is
    # ever produced, save it to results/ and load it here; do not draw a
    # flat line or any invented path.
    for e in escrow_runs:
        assert "chunks" not in e and "nodes_after" not in e, (
            "an ESCROW per-record series appeared; load it explicitly, do not "
            "interpolate")
    escrow_series = None
    assert escrow_series is None, "ESCROW trajectory needs a source file"
    background = laz["methods"]["escrow"]["background_or_unparsed"]   # 745 after the engine fix
    model_det = laz["llm_determinism"]["mean_ari"]    # 0.1647

    # ---- Wikipedia accuracy --------------------------------------------- #
    wm = wiki["methods"]
    aris = [wm[k]["ari"] for k, _, _ in RUNS]         # 0.9896 0.991 0.991 0.9896
    ari_escrow = wm["escrow"]["ari"]                  # 0.9407 after the engine fix
    assert wiki["escrow_determinism"]["identical_assignments"] is True
    assert wiki["n_records"] == 320 and wiki["truth_classes"] == 3

    # ---- figure ----------------------------------------------------------- #
    fig = plt.figure(figsize=(PRINT_W, PRINT_H))
    # widths in inches: left 2.95 (of which 1.64 carries n = 0..1,000 and the
    # rest the right-end labels), gap 0.70, right 1.41
    gs = fig.add_gridspec(1, 2, width_ratios=[2.95, 1.41], wspace=0.70 / 2.18,
                          left=0.07, right=0.99, bottom=0.235, top=0.80)
    ax = fig.add_subplot(gs[0, 0])
    bx = fig.add_subplot(gs[0, 1])

    # left: node count against records processed
    for (key, name, st), yfinal in zip(RUNS, finals):
        xs, ys = traj[key]
        ax.plot(xs, ys, color=INK, lw=0.9, ls=st["ls"], zorder=3)
        ax.plot(xs, ys, ls="none", marker=st["marker"], ms=3.0, mfc=st["mfc"],
                mec=INK, mew=0.7, markevery=(4, 5), zorder=4)
    # ESCROW: one end-point marker at (1000, K). Only the final count is data.
    ax.plot([1000], [K], ls="none", zorder=6, **ESCROW_MARKER)
    ax.text(1030, 1.0, "ESCROW final count: %d,\nidentical on all 3 runs" % K,
            color=GOLD, fontsize=7, ha="left", va="bottom", linespacing=1.05)

    # right-end labels inside the axes, dodged so the two 28s and the 24
    # stay apart; the x axis itself stops at 1,000
    label_y = {"llm_sampled_s1": 78, "llm_greedy_s0": 46,
               "llm_sampled_s2": 35, "llm_sampled_s0": 24}
    for (key, name, st), yfinal in zip(RUNS, finals):
        ly = label_y[key]
        if ly != yfinal:
            ax.plot([1000, 1020], [yfinal, ly], color=MUTE, lw=0.5, zorder=2)
        ax.text(1030, ly, "%s: %d" % (name, yfinal), color=INK, fontsize=7,
                ha="left", va="center")

    ax.set_xlim(0, 1800)
    ax.spines["bottom"].set_bounds(0, 1000)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xticklabels(["0", "200", "400", "600", "800", "1,000"])
    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xlabel("records processed, n")
    ax.set_ylabel("nodes in the graph")
    ax.set_title("Lazada, 1,000 raw listings:\nnodes after n records, four model runs",
                 loc="left", fontsize=7, pad=5)
    # gridlines bounded to the data range (0 to 1,000), so they stop where the
    # bottom spine stops and do not run under the right-end label column
    ax.hlines([20, 40, 60, 80], 0, 1000, color=MUTE, lw=0.4, alpha=0.35,
              zorder=0)
    # note in three short lines so it ends well left of the seed 1 marker at
    # (1,000, 78); the determinism number is the headline of this experiment,
    # so it is set in ink, not grey. A white bbox hides the 80 gridline.
    note_bbox = dict(fc=PAPER, ec="none", pad=0.8)
    ax.text(20, 99, "same records, order and prompt;", color=MUTE, fontsize=7,
            ha="left", va="top", bbox=note_bbox, zorder=5)
    ax.text(20, 89.5, "the 3 seeds agree at", color=INK, fontsize=7,
            ha="left", va="top", bbox=note_bbox, zorder=5)
    ax.text(20, 80, "ARI %.4f" % model_det, color=INK, fontsize=7,
            ha="left", va="top", bbox=note_bbox, zorder=5)

    # right: ARI against the three gold types
    xpos = list(range(len(RUNS) + 1))
    for i, ((key, name, st), a) in enumerate(zip(RUNS, aris)):
        bx.plot([i], [a], ls="none", marker=st["marker"], ms=4.6,
                mfc=st["mfc"], mec=INK, mew=0.8, zorder=4)
        above = i % 2 == 0
        # 0.008 ARI is about 4.6 pt here: clears the 4.6 pt marker, and the
        # white bbox keeps gridlines from striking through the digits
        bx.text(i, a + (0.008 if above else -0.008), "%.4f" % a, color=INK,
                fontsize=7, ha="center", va="bottom" if above else "top",
                bbox=dict(fc=PAPER, ec="none", pad=0.8), zorder=3)
    ie = len(RUNS)
    bx.plot([ie], [ari_escrow], ls="none", zorder=5, **ESCROW_MARKER)
    bx.text(ie, ari_escrow + 0.008, "%.4f" % ari_escrow, color=GOLD,
            fontsize=7, ha="center", va="bottom",
            bbox=dict(fc=PAPER, ec="none", pad=0.8), zorder=3)

    bx.set_xlim(-0.75, len(RUNS) + 0.75)
    # the axis follows the data: ESCROW sits well below the model runs, and a
    # hardcoded floor clipped its marker off the panel
    lo_v = min(list(aris) + [ari_escrow])
    hi_v = max(list(aris) + [ari_escrow])
    pad = max(0.012, (hi_v - lo_v) * 0.22)
    bx.set_ylim(lo_v - pad, hi_v + pad)     # room for the raised upper labels
    bx.set_xticks(xpos)
    # staggered so five labels fit under a narrow panel
    names = [name for _, name, _ in RUNS] + ["ESCROW"]
    bx.set_xticklabels([("\n" + nm) if i % 2 else nm
                        for i, nm in enumerate(names)], linespacing=1.0)
    import math as _m
    _step = 0.02 if (hi_v - lo_v) < 0.09 else 0.05
    _t0 = _m.floor((lo_v - pad) / _step) * _step
    bx.set_yticks([round(_t0 + k * _step, 4)
                   for k in range(int((hi_v + pad - _t0) / _step) + 2)
                   if _t0 + k * _step <= hi_v + pad])
    bx.set_xlabel("single runs; ESCROW: 3 runs", labelpad=2, fontsize=7)
    bx.set_ylabel("adjusted Rand index\n(ARI), 3 gold types", fontsize=7,
                  linespacing=1.0)
    _t = ("Wikipedia, 320 infoboxes:\nmodel more accurate here"
          if max(aris) > ari_escrow else
          "Wikipedia, 320 infoboxes:\nESCROW more accurate here")
    bx.set_title(_t, loc="left", fontsize=7, pad=5)
    bx.grid(axis="y", color=MUTE, lw=0.4, alpha=0.35)

    pdf = os.path.join(FIGDIR, "fig_llm.pdf")
    fig.savefig(pdf, format="pdf")
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, "fig_llm.png"), dpi=200)
    plt.close(fig)
    w, h = fig.get_size_inches()
    print("intended print size %.1f x %.1f in; figure is %.2f x %.2f in"
          % (PRINT_W, PRINT_H, w, h))
    assert abs(w - PRINT_W) < 0.01 and abs(h - PRINT_H) < 0.01
    print("ESCROW Lazada: final count %d only (no trajectory drawn); "
          "%d of %d records in background" % (K, background, laz["n_records"]))
    return pdf


if __name__ == "__main__":
    print("wrote", main())
