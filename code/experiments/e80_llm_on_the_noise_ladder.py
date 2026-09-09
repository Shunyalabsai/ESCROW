"""E80: the language model on the same noise ladder, one record at a time.

WHY. The ladder in E75 sweeps a stream from pure noise to fully separable groups, and so far only
the rule and the classical baselines have walked it. The obvious objection to every abstention
result in this paper is that a language model would simply do better: it reads the values, it can
say "these two records are the same kind of thing", and it needs nothing told in advance. That
objection deserves a number rather than an opinion. This is the language model arm of the same
ladder, on the same fixture, with the same metrics, so its rows drop straight into the same tables
as the rule's rows.

WHAT IS GIVEN TO EACH METHOD, WHICH IS THE SAME THING. Every method sees the same records, in the
same order, as key equals value lines. Six keys, the same six for every record, so the key names
carry no group information at all and only the values separate the groups. The language model gets
the key names because the rule gets the key names. Neither gets the true number of groups, neither
gets the truth labels, and neither gets a second pass over the stream.

THE TWO MODES, AND WHY ONLY ONE OF THEM COUNTS. The primary mode is one record per call. The model
sees the current node list and one new record, and answers with an existing node name or a new node.
That is exactly the decision the rule makes, made by prompting, and it is the only mode that is a
fair comparison, because the rule never sees a record it has not yet been asked about. The second
mode sends twenty records per call. Twenty records at once is a small batch with lookahead inside
it, so the model can see how a record relates to nineteen of its neighbours before it commits. It is
reported because the published prompting work uses it, and it is labelled not comparable everywhere
it appears.

WHAT IS MEASURED, ON FIVE AXES AND NOT ONE. The adjusted Rand index, written ARI, against the eight
true groups. The node count K against the true 8, which at signal 0 is the whole answer because the
true node count there is zero. The number of records placed in a node. Tokens in and tokens out.
Wall seconds. And whether two runs of the same records give the same answer, which the rule settles
by construction and a served model does not.

A record can be left in no node, and there are two ways to count one. It can join a single shared
background cluster, which is what the rule has always been scored under, or it can be a cluster of
its own, which is what the served model runs in this paper report. The two disagree by as much as
0.07 ARI here, so every method is scored both ways and each table uses one convention for every
method in it. Scoring the rule one way and the model the other would be two rules of scoring in one
table.

ONE THING THE MODEL CANNOT DO, SAID BEFORE ITS NUMBERS ARE READ. The shipped prompt asks for one
answer per record, and the node list starts empty, so the first record has to create a node. The
model's lowest possible node count is 1 and the rule's is 0. The prompt is not changed here, because
it is the prompt the rest of the paper measures. So at signal 0 the model's row should be read as
how far above one it went, and one node there is the best it could have done.

THE COST, AND WHY IT DECIDES THE STREAM LENGTH. One record per call means one call per record. The
rule's published ladder is three thousand records at eleven signal settings and three seeds, which
is ninety nine thousand calls, so the language model arm runs on a prefix of the same stream and
both arms of the rule are re-run on that same prefix.

The prefix is three hundred records. That length is the one setting in this file chosen by looking
at the truth, so it is stated plainly. It was chosen on the null. Below three hundred the split arm
mints a node on pure noise, because a node is paid for out of the records that arrive and a short
stream is thin evidence either way: at one hundred and twenty records it builds one node at signal 0
on every seed tried. At three hundred and above the null holds. The script walks five lengths and
prints that table before the ladder, so the choice can be read as the boundary it is rather than
taken on trust. The length is shared, so it tunes nothing in the rule's favour against the model:
every method gets the same three hundred records in the same order. What it does do is keep the rule
inside the range where its own null holds, and that is worth knowing when the signal 0 row is read.

At three hundred the split arm reaches six nodes of the true eight at full signal, against eight on
the full stream. So the head to head table shows the shape of the two curves, and the rule's best
node count needs the full stream, which the script also walks, below the head to head table.

The script prints the number of calls and the estimated tokens for the whole ladder, and two
cheaper plans, before it spends anything.

WHAT WOULD REFUTE THE PAPER'S CLAIM. A language model that returns zero nodes at signal 0, where
every record carries the same six keys and the values are uniform, while still scoring at the top of
the ladder. That would show abstention is reachable by prompting, and the sentence in the paper
would have to be narrowed to say so. A model that builds nodes at signal 0 has invented every one of
them, and no accuracy number further up the ladder repairs that.

HOW TO RUN IT.

    python3 -u experiments/e80_llm_on_the_noise_ladder.py --dry-run
        every code path with a canned answer instead of a call. No key, no network, no tokens.

    python3 -u experiments/e80_llm_on_the_noise_ladder.py --estimate-only
        the calls and the tokens the real run would cost, then exit.

    GEMINI_API_KEY=... python3 -u experiments/e80_llm_on_the_noise_ladder.py --model <name>
        the real run. Without the key it exits with a message and a non-zero status and writes
        nothing. Use e53_gemini_api.py --list to see which models the key can reach.

    ... --no-full-reference
        skip the rule's own three thousand record ladder. It is on by default. The whole sweep of
        it was measured at about six and a half minutes, and the slowest single point at about
        thirty seconds, so it is not the slow part of anything.

The served reasoning setting is E53's: the private trace is turned off through reasoning_effort,
because at the shipped output cap the trace eats the answer and the parser then measures the
harness rather than the model. Change it with ESCROW_E53_REASONING if a model needs something else.

The client, the prompt, the chunking, the parser and the node list carried across calls are the ones
in llm_graph_formation.py and e53_gemini_api.py. Nothing about the language model side is new here
except the fixture it is pointed at and the scoring.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from escrow import batch as B                                              # noqa: E402
from escrow.protocol import record_labels, run_stream                      # noqa: E402
from escrow.provenance import stamped                                      # noqa: E402

import llm_graph_formation as L                                            # noqa: E402
import e53_gemini_api as E53                                               # noqa: E402
from e4_baseline_army import _ari                                          # noqa: E402
from e75_a_fixture_that_tests_the_values import stream                     # noqa: E402

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
NAME = "e80_llm_on_the_noise_ladder"
OUT = os.path.join(ROOT, "results", NAME + ".json")
OUT_DRY = os.path.join(ROOT, "results", NAME + ".dry_run.json")

# The ladder the user asked for, walked in the noise direction: signal 1.0 is fully separable and
# signal 0.0 is pure noise where the true node count is zero.
SIGNAL = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0)
SEEDS = (0, 1, 2)
TRUE_GROUPS = 8
FULL_N = int(os.environ.get("E80_FULL_N", "3000"))   # the rule's own ladder length
LLM_N = int(os.environ.get("E80_N", "300"))          # the prefix the model arm affords
LENGTH_PROBE = (120, 200, 300, 400, 500)   # why that prefix and not another, measured below
CHUNK_ONE = 1              # the primary mode, and the only comparable one
CHUNK_BATCH = 20           # the published prompting protocol, kept for reference only
REPEATS = 3                # runs of the same records, to measure determinism
REPEAT_SIGNAL = (0.0, 1.0)

# Used only by the cost estimate, and stated in the results file so the arithmetic can be checked.
CHARS_PER_TOKEN = 4        # the same rule llm_graph_formation uses when a server reports no usage
NODE_LINE_CHARS = 64       # one line of the node list: a name and a one line description
OUT_TOKENS_PER_RECORD = 40  # one small JSON object per record in the answer
SECONDS_PER_CALL = 1.2     # a served endpoint, called one at a time


# =========================================================== the scoring ===
def two_conventions(truth, labels, is_unplaced):
    """Score one labelling under both ways of counting a record that was left in no node.

    A record in no node can be counted two ways and the two disagree, by as much as 0.07 ARI on
    this fixture, so both are reported for every method. Reporting one for the rule and the other
    for the model would be two rules of scoring in one table.

    background   every such record joins one shared cluster. This is what record_labels does and
                 what the rule has always been scored under in this paper.
    singletons   every such record gets a cluster of its own. This is what the served model runs
                 report, on the ground that a reply the parser could not read is a broken reply
                 and not a decision to leave the record out.

    A node name is prefixed here so that a node actually called __none cannot collide with the
    marker for a record in no node.
    """
    back, single = [], []
    for i, x in enumerate(labels):
        if is_unplaced(x):
            back.append("__none")
            single.append("__none%d" % i)
        else:
            back.append("n:" + str(x))
            single.append("n:" + str(x))
    return {"ARI": round(_ari(truth, back), 4),
            "ARI_unplaced_as_singletons": round(_ari(truth, single), 4),
            "in_a_node": sum(1 for x in labels if not is_unplaced(x))}


# ============================================================ the rule =====
def escrow_row(recs, truth, split):
    """One run of the rule. `split` picks the arm. The flag is set here every time, never inherited.

    The rule has no calibrated parameter, so it gets one run and no sweep.
    """
    B.SPLIT_MOVES = bool(split)
    t0 = time.time()
    g, b = run_stream(recs)
    secs = time.time() - t0
    lab = record_labels(g, len(recs))
    row = two_conventions(truth, lab, lambda x: x == -1)
    row.update({"K": int(g.K), "bits": round(b.total(), 1), "seconds": round(secs, 2),
                "tokens_in": 0, "tokens_out": 0, "calls": 0, "labels": lab})
    return row


# ============================================ how long the prefix must be ==
def why_this_length(seeds, lengths=LENGTH_PROBE):
    """The prefix cannot be as short as one likes, and the reason is the rule and not the model.

    On a short stream the split arm mints a node on pure noise, because a node has to be paid for
    out of the records that arrive and a short stream is thin evidence either way. This walks a few
    lengths at the two ends of the ladder and reports the node count, so the prefix used for the
    head to head table is chosen on the rule's own behaviour and not on what a run costs.
    """
    rows = []
    for n in lengths:
        row = {"records": n}
        for f in (0.0, 1.0):
            for method, split in (("escrow_base", False), ("escrow_split", True)):
                ks, ar = [], []
                for s in seeds:
                    recs, truth = stream(f, s, n=n)
                    r = escrow_row(recs, truth, split=split)
                    ks.append(r["K"])
                    ar.append(r["ARI"])
                row["%s_at_signal_%s" % (method, f)] = {
                    "K_per_seed": ks, "ARI_mean": round(sum(ar) / len(ar), 4)}
        row["null_holds"] = (max(row["escrow_base_at_signal_0.0"]["K_per_seed"]) == 0
                             and max(row["escrow_split_at_signal_0.0"]["K_per_seed"]) == 0)
        rows.append(row)
    return rows


# ================================================== the language model =====
def score_llm(r, truth):
    """Score one language model run through the same two_conventions the rule goes through.

    Both conventions are reported, and the printed tables use the same one for every method.
    """
    a = r["assignments"]
    row = two_conventions(truth, a, lambda x: x is None)
    row.update({"K": len(r["nodes"]),
                "tokens_in": r["tokens_in"], "tokens_out": r["tokens_out"],
                "seconds": r["wall_s"], "calls": r["n_chunks"],
                "counters": dict(r["counters"]), "labels": list(a)})
    return row


class quiet:
    """Swallow the per call progress the shipped runner prints. The real run wants it, the dry run
    would be four thousand lines of it."""

    def __init__(self, on):
        self.on = on

    def __enter__(self):
        if self.on:
            self._old = sys.stdout
            sys.stdout = open(os.devnull, "w")

    def __exit__(self, *a):
        if self.on:
            sys.stdout.close()
            sys.stdout = self._old
        return False


def llm_row(model, key, recs, truth, chunk, tag, verbose):
    """One language model run on the given records, at the given number of records per call."""
    ds = {"name": tag, "records": recs, "truth": truth}
    with quiet(not verbose):
        r = E53.run_at_chunk(E53.BASE, model, key, ds, chunk)
    row = score_llm(r, truth)
    row["raw_first_two"] = r["raw_outputs"][:2]
    return row


# ================================================== the canned endpoint ====
_REC_RE = re.compile(r"^\[record (\d+)\]$")
_KV_RE = re.compile(r"^[A-Za-z0-9_]+=")


def _read_prompt(user):
    """Pull the listed node names and the records back out of the prompt the runner just built.

    The canned endpoint answers from the prompt alone, exactly as a served model would, so the
    prompt builder, the node list carried forward and the parser are all exercised for real.
    """
    listed, recs = [], []
    cur = None
    section = "nodes"
    for line in user.splitlines():
        if line.startswith("New records:"):
            section = "records"
            continue
        if line.startswith("For every record above"):
            break
        if section == "nodes":
            if line.startswith("- "):
                listed.append(line[2:].split(":", 1)[0].strip())
            continue
        m = _REC_RE.match(line.strip())
        if m:
            cur = (int(m.group(1)), [])
            recs.append(cur)
            continue
        if cur is not None and _KV_RE.match(line.strip()):
            cur[1].append(line.strip().split("=", 1)[1])
    return listed, recs


def _canned_answer(user, i, tally=None):
    """A deterministic fake reply, with a defect on a fixed schedule so every branch of the shipped
    parser and of apply_answer is reached at least once during a dry run.

    This is a test harness and nothing else. It groups records by an arbitrary sum of their value
    numbers, so its accuracy means nothing and is never reported as a result.
    """
    listed, recs = _read_prompt(user)
    known = set(listed)
    head, objs = [], []
    for rid, vals in recs:
        h = sum(int(v[1:]) for v in vals if v[1:].isdigit())
        name = f"kind_{h % 6}"
        o = {"record": rid, "node": name}
        if name not in known:
            if i % 13 != 7:                       # otherwise: a new name with no new flag
                o["new"] = True
                o["description"] = "a fake kind, invented by the dry run and not by a model"
            known.add(name)
        elif i % 41 == 23:
            o["new"] = True                       # the new flag on a name that already exists
        objs.append(o)
    if objs:
        rid0 = objs[0]["record"]
        # These two have to come before the real answer for that record, because the shipped
        # parser takes the first answer per record and calls every later one a duplicate.
        if i % 37 == 19:
            head.append({"record": rid0, "node": ""})            # no usable node name
        if i % 31 == 17 and listed:
            head.append({"record": rid0, "node": listed[0].upper() + " !"})   # a name variant
        if i % 29 == 13:
            objs.append(dict(objs[0]))            # the same record answered twice
        if i % 17 == 9:
            objs.append({"record": 99999, "node": "kind_0"})     # a record id not in this call
    text = json.dumps(head + objs)
    if i % 7 == 3:
        text = "```json\n" + text + "\n```"
        if tally is not None:
            tally["answers_in_a_code_fence"] += 1
    if i % 11 == 5:
        text = text.rstrip("]")                   # a cut off answer, for the salvage path
        if tally is not None:
            tally["answers_cut_off_mid_json"] += 1
    return text


class FakeEndpoint:
    """Stands in for urllib.request.urlopen. Never opens a socket and never sees a key."""

    def __init__(self, fail_on=(5, 8, 9)):
        self.calls = 0
        self.fail_on = set(fail_on)
        self.tally = {"canned_calls": 0, "canned_failures": 0,
                      "answers_in_a_code_fence": 0, "answers_cut_off_mid_json": 0,
                      "replies_with_no_usage_block": 0, "replies_stopped_on_length": 0}

    def urlopen(self, req, timeout=None):
        i = self.calls
        self.calls += 1
        self.tally["canned_calls"] += 1
        if i in self.fail_on:
            self.tally["canned_failures"] += 1
            raise urllib.error.URLError("dry run: a canned failure, to exercise the retry path")
        body = json.loads(req.data.decode())
        user = body["messages"][0]["content"]
        text = _canned_answer(user, i, self.tally)
        stop = "length" if i % 23 == 21 else "stop"
        if stop == "length":
            self.tally["replies_stopped_on_length"] += 1
        payload = {"choices": [{"message": {"content": text}, "finish_reason": stop}]}
        if i % 19 != 11:                          # otherwise: no usage block, so the runner
            payload["usage"] = {"prompt_tokens": max(1, len(user) // CHARS_PER_TOKEN),
                                "completion_tokens": max(1, len(text) // CHARS_PER_TOKEN)}
        else:                                     # falls back to counting the characters
            self.tally["replies_with_no_usage_block"] += 1
        return io.BytesIO(json.dumps(payload).encode())


# ======================================================= the cost sheet ====
def cost_of(signals, seeds, n, chunk, k_assume):
    """Calls and tokens for one plan, from the real prompts, with no call made.

    The prompts are built here exactly as the runner builds them, so the input side is measured and
    not guessed. The one guess is how long the node list gets, because that depends on how many
    nodes the model invents, and it is swept over `k_assume` rather than fixed.
    """
    calls = 0
    chars = 0
    for f in signals:
        for s in seeds:
            recs, _ = stream(f, s, n=n)
            n_calls = (len(recs) + chunk - 1) // chunk
            for c in range(n_calls):
                part = recs[c * chunk:(c + 1) * chunk]
                user = L.SYSTEM + "\n\n" + L.make_prompt({}, part, c * chunk)
                k_now = int(k_assume * (c + 1) / n_calls)
                chars += len(user) + (k_now * NODE_LINE_CHARS - len("(none yet)") if k_now else 0)
            calls += n_calls
    t_in = chars // CHARS_PER_TOKEN
    t_out = calls * min(L.MAX_NEW, OUT_TOKENS_PER_RECORD * chunk)
    return {"calls": calls, "tokens_in": t_in, "tokens_out": t_out,
            "minutes_at_%.1fs_per_call" % SECONDS_PER_CALL: round(calls * SECONDS_PER_CALL / 60, 1)}


def print_cost(signals, seeds, n, repeats, repeat_signal):
    print("\nWHAT THE REAL RUN COSTS. Input tokens are counted from the prompts this script builds,")
    print("at %d characters per token. The node list length is the one guess, so it is swept."
          % CHARS_PER_TOKEN)
    print("Output is charged at %d tokens per record. Wall time assumes calls made one after "
          "another." % OUT_TOKENS_PER_RECORD)
    sheet = {"assumptions": {"chars_per_token": CHARS_PER_TOKEN,
                             "node_list_line_chars": NODE_LINE_CHARS,
                             "output_tokens_per_record": OUT_TOKENS_PER_RECORD,
                             "seconds_per_call": SECONDS_PER_CALL,
                             "note": ("the node count the model reaches is not known before the "
                                      "run, so the input side is given at three of them")},
                "plans": []}
    print("\n  %-50s %8s %12s %12s %9s" % ("plan", "calls", "tokens in", "tokens out",
                                              "minutes"))
    rows = []
    for k_assume in (8, 30, 80):
        rows.append(["ladder, one record per call, K reaches %d" % k_assume,
                     cost_of(signals, seeds, n, CHUNK_ONE, k_assume), k_assume == 30])
    rows.append(["ladder, 20 records per call, K reaches 30 (not comparable)",
                 cost_of(signals, seeds, n, CHUNK_BATCH, 30), True])
    # The determinism block runs the model `repeats` times from scratch at each of these signals.
    # It does not reuse the ladder run, so all of them are billed here.
    rows.append(["determinism, %d runs at signal %s, one per call"
                 % (repeats, list(repeat_signal)),
                 cost_of(repeat_signal, seeds[:1] * repeats, n, CHUNK_ONE, 30), True])
    rows.append(["cheaper: the same ladder at seed %d only, one per call" % seeds[0],
                 cost_of(signals, seeds[:1], n, CHUNK_ONE, 30), False])
    rows.append(["cheaper but too short: 120 records, all seeds, one per call",
                 cost_of(signals, seeds, 120, CHUNK_ONE, 30), False])
    tot = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
    for label, c, counted in rows:
        mins = c["minutes_at_%.1fs_per_call" % SECONDS_PER_CALL]
        print("  %-50s %8d %12d %12d %9.1f" % (label[:50], c["calls"], c["tokens_in"],
                                               c["tokens_out"], mins))
        sheet["plans"].append(dict(plan=label, counted_in_the_total=counted, **c))
        if not counted:
            continue
        tot["calls"] += c["calls"]
        tot["tokens_in"] += c["tokens_in"]
        tot["tokens_out"] += c["tokens_out"]
    print("  %-50s %8d %12d %12d %9.1f" % ("WHOLE RUN, at K reaches 30", tot["calls"],
                                           tot["tokens_in"], tot["tokens_out"],
                                           tot["calls"] * SECONDS_PER_CALL / 60))
    print("  the rule walks the same ladder for 0 calls and 0 tokens.")
    print("  the two cheaper plans are not in the total. Pass --seeds %d for the first. The second"
          % seeds[0])
    print("  is below the length at which the rule's own null holds, so it costs less and says "
          "less.")
    sheet["whole_run_at_k_30"] = dict(tot, minutes=round(tot["calls"] * SECONDS_PER_CALL / 60, 1))
    sheet["escrow_cost"] = {"calls": 0, "tokens_in": 0, "tokens_out": 0}
    return sheet


# ============================================================== the run ====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="canned answers instead of calls: no key, no network, no tokens")
    ap.add_argument("--estimate-only", action="store_true",
                    help="print the calls and tokens the real run would cost, then stop")
    ap.add_argument("--model", default=os.environ.get("ESCROW_E80_MODEL")
                    or (E53.MODELS[0] if E53.MODELS else "gemini-2.5-flash"))
    ap.add_argument("--n", type=int, default=LLM_N, help="records in the language model prefix")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--no-full-reference", dest="full_reference", action="store_false",
                    help=("skip the rule's own %d record ladder. It is on by default: it is the "
                          "length the paper's ladder uses, and it was measured at about six and a "
                          "half minutes for the whole sweep" % FULL_N))
    ap.set_defaults(full_reference=True)
    ap.add_argument("--verbose", action="store_true", help="show every call as it happens")
    ap.add_argument("--yes", action="store_true", help="do not ask before spending tokens")
    args = ap.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "")
    n = args.n
    dry = args.dry_run

    print("E80: the language model on the same noise ladder, one record at a time.")
    print("  fixture   E75, 8 groups, 6 shared keys, the group lives in the values only")
    print("  ladder    signal " + ", ".join(str(f) for f in SIGNAL))
    print("  seeds     %s" % (list(seeds),))
    print("  records   %d for every method in the head to head table (the rule's own ladder uses "
          "%d)" % (n, FULL_N))
    print("  model     %s at %s" % (args.model if not dry else "CANNED, no model", E53.BASE))

    sheet = print_cost(SIGNAL, seeds, n, args.repeats, REPEAT_SIGNAL)
    if args.estimate_only:
        return 0

    key = "DRY-RUN-NO-KEY"
    if not dry:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            print("\nGEMINI_API_KEY is not set. This experiment makes real calls to a served "
                  "model,\nso it stops here rather than write a half empty results file. Set the "
                  "key in the\nenvironment, or run it with --dry-run to prove the code path "
                  "without spending anything.", file=sys.stderr)
            return 2
        if not args.yes and sys.stdin.isatty():
            if input("\nspend the tokens above? type yes to go on: ").strip().lower() != "yes":
                print("stopped, nothing spent.")
                return 0

    fake = None
    if dry:
        fake = FakeEndpoint()
        urllib.request.urlopen = fake.urlopen
        L.API_RETRIES = 2          # so the canned give up path is reached without a long wait
        print("\nDRY RUN. Every call is answered from a canned reply built out of the prompt "
              "itself.\nThe numbers below are not a measurement of any model. They exist to prove "
              "the prompt,\nthe parser, the scoring and the results file.")

    report = {
        "experiment": "E80 the language model on the same noise ladder, one record at a time",
        "dry_run": dry,
        "model": None if dry else args.model,
        "endpoint": E53.BASE,
        "reasoning_effort": E53.REASONING,
        "reused_from": ["experiments/llm_graph_formation.py (prompt, chunking, parser, counters)",
                        "experiments/e53_gemini_api.py (the served client and the key handling)",
                        "experiments/e75_a_fixture_that_tests_the_values.py (the stream)"],
        "what_each_method_was_given": {
            "records": ("the same records in the same order, as key equals value lines, six keys "
                        "per record and the same six for every record"),
            "escrow_base": "the records, nothing else. No node count, no labels, one pass",
            "escrow_split": "the same, with the split move and the residual mint turned on",
            "language_model": ("the records, the same key names, plus the node list it has built so "
                               "far. No node count, no labels, one pass, and no example answers"),
            "nobody_got": ["the true number of groups", "any truth label", "a second pass",
                           "any knob tuned on the test labels"],
        },
        "how_a_record_in_no_node_is_scored": {
            "why_both": ("a record left in no node can be counted two ways, and on this fixture "
                         "they differ by as much as 0.07 ARI, so both are reported for every "
                         "method. One convention for the rule and the other for the model would "
                         "be two rules of scoring in one table"),
            "background": ("every record in no node joins one shared cluster. This is what "
                           "record_labels does and what the rule has always been scored under, "
                           "and it is the ARI column in the main table"),
            "singletons": ("every record in no node gets a cluster of its own. This is what the "
                           "served model runs in this paper report. It is the second table, and "
                           "it is stored per method as ARI_unplaced_as_singletons"),
        },
        "what_the_model_cannot_do": (
            "the shipped prompt asks for one answer per record, and the node list starts empty, so "
            "the first record has to create a node. The model's lowest possible node count is 1 "
            "and the rule's is 0. That is a real asymmetry in the signal 0 row and it is not "
            "repaired here, because repairing it would mean a different prompt from the one the "
            "rest of the paper measures. Read the model's signal 0 node count as how far above one "
            "it went"),
        "the_two_modes": {
            "one_record_per_call": ("primary. The model sees one new record and the node list. This "
                                    "is the rule's own decision, made by prompting"),
            "twenty_per_call": ("NOT COMPARABLE. Twenty records arrive together, so the model sees "
                                "nineteen neighbours before it commits to any of them. Reported "
                                "because published prompting work uses it"),
        },
        "stream_length": {"language_model_and_head_to_head": n, "escrow_own_ladder": FULL_N,
                          "why": ("one record per call is one call per record, so the full three "
                                  "thousand record ladder is ninety nine thousand calls. Both arms "
                                  "of the rule are re-run on the same prefix so the head to head "
                                  "table is honest")},
        "cost_estimate": sheet,
        "caveats": [
            ("the head to head table is %d records long, not %d, because one record per call is "
             "one call per record. Read the head to head table for the shape of the two curves, "
             "not for the rule's best node count, which needs the full stream. The rule's own %d "
             "record ladder is printed below it for that"
             % (n, FULL_N, FULL_N)),
            ("the prefix length is the one setting in this file chosen by looking at the truth. "
             "It was chosen on the null: below 300 records the split arm mints a node at signal 0 "
             "where the true count is zero. The block how_long_the_prefix_must_be walks the "
             "lengths and prints what happens, so the choice can be checked and is not asserted. "
             "The length is shared: every method in the head to head table gets the same %d "
             "records in the same order, so it tunes nothing in the rule's favour against the "
             "model. It does keep the rule inside the range where its own null holds, and that is "
             "said here rather than left for a reader to find" % n),
            ("the model cannot answer zero. The shipped prompt asks for one answer per record and "
             "the node list starts empty, so its lowest possible node count is 1 and the rule's is "
             "0. The prompt is not changed here, because it is the prompt the rest of the paper "
             "measures"),
            ("the language model gets one run per condition and no knob, because it has none to "
             "sweep. Temperature is 0 and the private reasoning trace is turned off, which is what "
             "E53 does and for the same reason: at the shipped output cap the trace eats the "
             "answer"),
            ("a reply the parser cannot place leaves the record in no node. That is a broken "
             "reply, not an abstention, and the two are told apart by the counters kept per "
             "condition"),
            ("the twenty records per call rows are not a comparison. They are there because the "
             "published prompting work is written that way"),
        ],
        "seeds": list(seeds), "signals": list(SIGNAL), "true_groups": TRUE_GROUPS,
        "escrow_own_ladder_note": ("the rule on its own %d record ladder, which the language "
                                   "model arm cannot afford. On by default and measured at about "
                                   "six and a half minutes for the whole sweep, so it is not the "
                                   "slow part of anything. Empty only if --no-full-reference was "
                                   "passed" % FULL_N),
        "ladder": [], "escrow_own_ladder": [], "determinism": {},
    }

    # ------------------------------------------- how long the prefix must be
    print("\nWHY THE PREFIX IS %d RECORDS AND NOT FEWER. On a short stream the split arm mints a "
          "node\non pure noise, so the length is chosen on the rule's own behaviour, not on what "
          "a run costs." % n)
    print("\n  %8s | %-24s | %-24s | %s"
          % ("records", "K at signal 0, truth 0", "K at signal 1, truth 8", "null holds"))
    print("  %8s | %11s %12s | %11s %12s |" % ("", "base", "split", "base", "split"))
    probe = why_this_length(seeds, tuple(sorted(set(LENGTH_PROBE + (n,)))))
    for r in probe:
        print("  %8d | %11s %12s | %11s %12s | %s"
              % (r["records"], r["escrow_base_at_signal_0.0"]["K_per_seed"],
                 r["escrow_split_at_signal_0.0"]["K_per_seed"],
                 r["escrow_base_at_signal_1.0"]["K_per_seed"],
                 r["escrow_split_at_signal_1.0"]["K_per_seed"],
                 "yes" if r["null_holds"] else "NO"))
    holds = [r["records"] for r in probe if r["null_holds"]]
    fails = [r["records"] for r in probe if not r["null_holds"]]
    verdict = ("the null fails at %s and holds at %s, so %d is the boundary and not a lucky point"
               % (fails, holds, min(holds))) if holds and fails else (
                   "every length probed behaves the same way, so this table settles nothing")
    print("  " + verdict)
    report["how_long_the_prefix_must_be"] = probe
    report["what_the_prefix_probe_says"] = verdict

    # ------------------------------------------------------------- ladder --
    print("\nTHE LADDER, every method on the same %d records. K is the node count against the true "
          "%d." % (n, TRUE_GROUPS))
    print("At signal 0 the true node count is ZERO, so K is the answer there and ARI is not.")
    print("Every ARI here counts the records left in no node as one shared cluster, for every")
    print("method alike. The same table with those records as one cluster each follows below.")
    print("\n  %6s %6s | %20s | %20s | %34s" % ("signal", "noise", "escrow_base",
                                                "escrow_split", "llm, one record per call"))
    print("  %6s %6s | %4s %8s %6s | %4s %8s %6s | %4s %8s %6s %7s %6s"
          % ("", "", "K", "ARI", "in_nd", "K", "ARI", "in_nd", "K", "ARI", "in_nd", "tokens",
             "secs"))
    for f in SIGNAL:
        row = {"signal": f, "noise": round(1 - f, 2), "per_seed": []}
        acc = {"escrow_base": [], "escrow_split": [], "llm_one_per_call": [], "llm_twenty": []}
        for s in seeds:
            recs, truth = stream(f, s, n=n)
            per = {"seed": s,
                   "escrow_base": escrow_row(recs, truth, split=False),
                   "escrow_split": escrow_row(recs, truth, split=True),
                   "llm_one_per_call": llm_row(args.model, key, recs, truth, CHUNK_ONE,
                                               f"ladder_s{f}_seed{s}_one", args.verbose),
                   "llm_twenty": llm_row(args.model, key, recs, truth, CHUNK_BATCH,
                                         f"ladder_s{f}_seed{s}_twenty", args.verbose)}
            for m in acc:
                acc[m].append(per[m])
            if not report.get("a_sample_of_what_came_back"):
                report["a_sample_of_what_came_back"] = {
                    "signal": f, "seed": s, "mode": "one record per call",
                    "first_two_replies": per["llm_one_per_call"]["raw_first_two"],
                    "note": ("kept from the first condition only, so the reply can be read rather "
                             "than trusted")}
            for m in acc:                                 # labels are big and are not kept
                per[m].pop("labels", None)
                per[m].pop("raw_first_two", None)
            row["per_seed"].append(per)

        def mean(m, field):
            return round(sum(x[field] for x in acc[m]) / len(acc[m]), 4)

        def total(m, field):
            return sum(x[field] for x in acc[m])

        for m in acc:
            row[m] = {"K_mean": mean(m, "K"), "K_per_seed": [x["K"] for x in acc[m]],
                      "ARI_mean": mean(m, "ARI"),
                      "ARI_unplaced_as_singletons_mean": mean(m, "ARI_unplaced_as_singletons"),
                      "in_a_node": "%d/%d" % (total(m, "in_a_node"), n * len(seeds)),
                      "tokens_in": total(m, "tokens_in"), "tokens_out": total(m, "tokens_out"),
                      "calls": total(m, "calls"), "seconds": round(total(m, "seconds"), 1)}
        report["ladder"].append(row)
        b, sp, o = row["escrow_base"], row["escrow_split"], row["llm_one_per_call"]
        print("  %6.2f %6.2f | %4.1f %8.4f %6d | %4.1f %8.4f %6d | %4.1f %8.4f %6d %7d %6.1f"
              % (f, 1 - f, b["K_mean"], b["ARI_mean"], total("escrow_base", "in_a_node"),
                 sp["K_mean"], sp["ARI_mean"], total("escrow_split", "in_a_node"),
                 o["K_mean"], o["ARI_mean"], total("llm_one_per_call", "in_a_node"),
                 o["tokens_in"] + o["tokens_out"], o["seconds"]))

    print("\nTHE SAME LADDER, SCORED THE OTHER WAY: a record in no node is a cluster of its own.")
    print("Same runs, same records, one convention for every method. The rule places fewer records")
    print("than the model does, so it is the arm this convention moves.")
    print("\n  %6s | %17s | %17s | %17s" % ("signal", "escrow_base", "escrow_split",
                                            "llm, one per call"))
    print("  %6s | %8s %8s | %8s %8s | %8s %8s"
          % ("", "backgrd", "singles", "backgrd", "singles", "backgrd", "singles"))
    for row in report["ladder"]:
        print("  %6.2f | %8.4f %8.4f | %8.4f %8.4f | %8.4f %8.4f"
              % (row["signal"],
                 row["escrow_base"]["ARI_mean"],
                 row["escrow_base"]["ARI_unplaced_as_singletons_mean"],
                 row["escrow_split"]["ARI_mean"],
                 row["escrow_split"]["ARI_unplaced_as_singletons_mean"],
                 row["llm_one_per_call"]["ARI_mean"],
                 row["llm_one_per_call"]["ARI_unplaced_as_singletons_mean"]))

    print("\nTHE SAME LADDER AT TWENTY RECORDS PER CALL. NOT COMPARABLE: the model sees nineteen")
    print("neighbours of every record before it answers, and the rule sees none.")
    print("\n  %6s | %4s %8s %6s %7s %6s" % ("signal", "K", "ARI", "in_nd", "tokens", "secs"))
    for row in report["ladder"]:
        t = row["llm_twenty"]
        print("  %6.2f | %4.1f %8.4f %6s %7d %6.1f"
              % (row["signal"], t["K_mean"], t["ARI_mean"], t["in_a_node"].split("/")[0],
                 t["tokens_in"] + t["tokens_out"], t["seconds"]))

    # ------------------------------------------------------- determinism ---
    print("\nDETERMINISM. The same records, run %d times, at both ends of the ladder." % args.repeats)
    print("The rule is deterministic by construction, and is measured rather than assumed.")
    print("\n  %6s %16s | %-24s %s" % ("signal", "method", "K per repeat",
                                       "records placed the same way"))
    for f in REPEAT_SIGNAL:
        recs, truth = stream(f, seeds[0], n=n)
        block = {}
        for method in ("escrow_base", "escrow_split", "llm_one_per_call"):
            runs = []
            for rep in range(args.repeats):
                if method == "llm_one_per_call":
                    runs.append(llm_row(args.model, key, recs, truth, CHUNK_ONE,
                                        f"repeat_s{f}_r{rep}", args.verbose))
                else:
                    runs.append(escrow_row(recs, truth, split=(method == "escrow_split")))
            labs = [r["labels"] for r in runs]
            same = sum(1 for i in range(n) if len({str(l[i]) for l in labs}) == 1)
            pair = [round(_ari([str(x) for x in labs[i]], [str(x) for x in labs[j]]), 4)
                    for i in range(len(labs)) for j in range(i + 1, len(labs))]
            block[method] = {"K_per_repeat": [r["K"] for r in runs],
                             "ARI_per_repeat": [r["ARI"] for r in runs],
                             "records_placed_identically": "%d/%d" % (same, n),
                             "ARI_between_repeats": pair,
                             "identical": same == n and len(set(pair)) <= 1 and all(
                                 abs(p - 1.0) < 1e-9 for p in pair),
                             "tokens": sum(r["tokens_in"] + r["tokens_out"] for r in runs)}
            print("  %6.2f %16s | %-24s %s" % (f, method, str(block[method]["K_per_repeat"]),
                                               block[method]["records_placed_identically"]))
        report["determinism"][str(f)] = block

    # ------------------------------------- the rule on its own full ladder -
    if args.full_reference:
        print("\nFOR CONTEXT, THE RULE ON ITS OWN LADDER AT %d RECORDS, which the language model "
              "arm\ncannot afford. The gap between this and the table above is the price of the "
              "prefix,\nnot a property of any method." % FULL_N)
        print("\n  %6s | %4s %8s | %4s %8s" % ("signal", "K", "ARI", "K", "ARI"))
        print("  %6s | %13s | %13s" % ("", "escrow_base", "escrow_split"))
        for f in SIGNAL:
            got = {}
            for method, split in (("escrow_base", False), ("escrow_split", True)):
                rs = []
                for s in seeds:
                    recs, truth = stream(f, s, n=FULL_N)
                    r = escrow_row(recs, truth, split=split)
                    r.pop("labels", None)
                    rs.append(r)
                got[method] = {"K_mean": round(sum(x["K"] for x in rs) / len(rs), 2),
                               "K_per_seed": [x["K"] for x in rs],
                               "ARI_mean": round(sum(x["ARI"] for x in rs) / len(rs), 4),
                               "ARI_unplaced_as_singletons_mean": round(
                                   sum(x["ARI_unplaced_as_singletons"] for x in rs)
                                   / len(rs), 4),
                               "in_a_node": "%d/%d" % (sum(x["in_a_node"] for x in rs),
                                                       FULL_N * len(seeds))}
            report["escrow_own_ladder"].append(dict(signal=f, noise=round(1 - f, 2), **got))
            print("  %6.2f | %4.1f %8.4f | %4.1f %8.4f"
                  % (f, got["escrow_base"]["K_mean"], got["escrow_base"]["ARI_mean"],
                     got["escrow_split"]["K_mean"], got["escrow_split"]["ARI_mean"]))

    # -------------------------------------------------- what the dry run --
    if dry:
        counters = {}
        for row in report["ladder"]:
            for per in row["per_seed"]:
                for k, v in per["llm_one_per_call"]["counters"].items():
                    counters[k] = counters.get(k, 0) + v
        report["dry_run_coverage"] = {
            "the_canned_endpoint": dict(fake.tally),
            "parser_and_apply_counters_reached": counters,
            "note": ("every counter above is a branch of the shipped parser and of apply_answer "
                     "that the canned answers reached, so the paths are proven and not assumed"),
            "why_the_canned_arm_looks_unstable": (
                "the canned endpoint puts its defect on a schedule counted in calls, and a repeat "
                "starts at a different call number, so the canned answers differ between repeats. "
                "That is the harness and not a model. It is useful: it shows the determinism "
                "check can see a difference rather than passing by construction")}
        print("\nDRY RUN COVERAGE. What the canned endpoint sent, and what the shipped parser "
              "made of it.")
        for k, v in fake.tally.items():
            print("  %-34s %8d" % (k, v))
        want = ["new_nodes", "loose_name_match", "implicit_new_node", "bad_record_id",
                "duplicate_record_entry", "missing_node_name", "new_flag_on_existing_name",
                "unparsed_records"]
        for k in want:
            print("  %-28s %8d %s" % (k, counters.get(k, 0),
                                      "reached" if counters.get(k, 0) else "NOT REACHED"))
        report["dry_run_coverage"]["all_branches_reached"] = all(counters.get(k, 0) for k in want)

    # ------------------------------------------------------------ headline -
    zero = report["ladder"][0]
    top = report["ladder"][-1]
    det = report["determinism"].get("1.0", {})
    report["headline"] = {
        "dry_run": dry,
        "records_per_method": n,
        "K_at_signal_0_truth_is_0": {
            "escrow_base": zero["escrow_base"]["K_per_seed"],
            "escrow_split": zero["escrow_split"]["K_per_seed"],
            "llm_one_record_per_call": zero["llm_one_per_call"]["K_per_seed"],
            "llm_twenty_per_call_not_comparable": zero["llm_twenty"]["K_per_seed"]},
        "ARI_at_signal_1_records_in_no_node_as_one_cluster": {
            "escrow_base": top["escrow_base"]["ARI_mean"],
            "escrow_split": top["escrow_split"]["ARI_mean"],
            "llm_one_record_per_call": top["llm_one_per_call"]["ARI_mean"],
            "llm_twenty_per_call_not_comparable": top["llm_twenty"]["ARI_mean"]},
        "ARI_at_signal_1_records_in_no_node_as_singletons": {
            "escrow_base": top["escrow_base"]["ARI_unplaced_as_singletons_mean"],
            "escrow_split": top["escrow_split"]["ARI_unplaced_as_singletons_mean"],
            "llm_one_record_per_call": top["llm_one_per_call"][
                "ARI_unplaced_as_singletons_mean"],
            "llm_twenty_per_call_not_comparable": top["llm_twenty"][
                "ARI_unplaced_as_singletons_mean"]},
        "the_model_cannot_answer_zero": (
            "the shipped prompt asks for one answer per record and the node list starts empty, so "
            "the first record has to create a node and the model's lowest possible K is 1. The "
            "rule's lowest possible K is 0. At signal 0 read the model's row as how far above one "
            "it went, and read the rule's row as whether it stayed at zero"),
        "tokens_for_the_whole_ladder": {
            "escrow_base": 0, "escrow_split": 0,
            "llm_one_record_per_call": sum(r["llm_one_per_call"]["tokens_in"]
                                           + r["llm_one_per_call"]["tokens_out"]
                                           for r in report["ladder"]),
            "llm_twenty_per_call": sum(r["llm_twenty"]["tokens_in"] + r["llm_twenty"]["tokens_out"]
                                       for r in report["ladder"])},
        "calls_for_the_whole_ladder": {
            "escrow_base": 0, "escrow_split": 0,
            "llm_one_record_per_call": sum(r["llm_one_per_call"]["calls"]
                                           for r in report["ladder"]),
            "llm_twenty_per_call": sum(r["llm_twenty"]["calls"] for r in report["ladder"])},
        "same_answer_on_a_repeat_at_signal_1": {
            m: det.get(m, {}).get("records_placed_identically") for m in
            ("escrow_base", "escrow_split", "llm_one_per_call")},
        "the_rule_on_its_own_full_ladder": (
            {"records": FULL_N,
             "K_at_signal_0_truth_is_0": {
                 m: report["escrow_own_ladder"][0][m]["K_per_seed"]
                 for m in ("escrow_base", "escrow_split")},
             "ARI_at_signal_1": {
                 m: report["escrow_own_ladder"][-1][m]["ARI_mean"]
                 for m in ("escrow_base", "escrow_split")},
             "K_at_signal_1_true_is_8": {
                 m: report["escrow_own_ladder"][-1][m]["K_per_seed"]
                 for m in ("escrow_base", "escrow_split")},
             "note": ("this is the length the paper's ladder uses. The model arm cannot reach it "
                      "at one call per record, so the head to head table is the 300 record prefix "
                      "and this is what the rule does when it is not held to that prefix")}
            if report["escrow_own_ladder"] else "skipped with --no-full-reference"),
        "reading": ("the row to read first is signal 0, where the true node count is zero. A node "
                    "there was invented out of uniform values that every record shares. The rule "
                    "walks the whole ladder for zero calls and zero tokens, and gives the same "
                    "answer every time it is run"),
    }
    if dry:
        report["headline"]["warning"] = ("DRY RUN. Every language model number here came from a "
                                         "canned reply and measures nothing about any model")
    print("\n" + json.dumps(report["headline"], indent=2))

    path = OUT_DRY if dry else OUT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(stamped(report), open(path, "w"), indent=2)
    print("written", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
