"""E79: if you tell a method what the keys and the values MEAN, does it do better?

WHY. A language model is not handed `k3 = v41`. It is handed `tag = sparrow`, and it already knows
that a sparrow is a bird, that an eagle is a bird too, and that neither is a hammer. That prior
knowledge of what words mean is most of what people think they are buying when they put a language
model in front of a stream of records. Our method reads the same records and, in the shipped code,
treats every value as an unrelated symbol. So the fair question, and the one asked here, is whether
the meaning of the strings is worth anything, and whether it is worth as much to us as it is to a
method built out of an embedding.

THE ONE THING THAT MUST BE HELD FIXED. The data. This experiment runs E75's value fixture twice
over. In the first condition every key is `k0` to `k5` and every value is `v0` to `v63`. In the
second condition the very same records are rewritten word for word, one to one: key `k2` becomes
`tag`, value `v0` becomes `sparrow`, value `v8` becomes `oak`. The group of every record, the count
of every value, the arrival order, all identical. Only the strings change. A method that ignores
what a string says must therefore score exactly the same in both conditions, and this experiment
asserts that rather than assuming it.

WHAT THE DOMAIN KNOWLEDGE IS. Two things, and both are things a language model is given too.

  a gloss for each key   the six keys are six places a word can appear in a catalogue entry: the
                         title, the caption, a tag, a related term, an index entry, a cross
                         reference. Every entry has all six, so the gloss says plainly that the key
                         names cannot separate the groups. That is true of the fixture by
                         construction and it is stated so that no reader thinks the key names are
                         doing the work.
  a similarity between   a unit vector for each of the sixty four value words, from which the
  the value words        cosine of any two words is read. Two birds are near each other, a bird and
                         a hammer are not. This is the same object a text embedding model returns
                         for these words, and it is the only channel through which meaning enters.

There is no embedding model on this machine and no network, so the vectors are written down here
from the vocabulary's own construction, and that is a stand-in a real embedding would replace. The
stand-in is not clean: the tree words and the fruit words are deliberately placed near each other,
because a cherry really is a tree, and four words are pulled toward a second sense the way `iron`
sits between the metals and the tools. The measured spread of the stand-in is reported in the output
so the reader can see how good it is.

WHY THIS IS THE MOST GENEROUS TEST DOMAIN KNOWLEDGE COULD GET, AND WHY THAT IS THE POINT. In this
fixture a group IS a slice of the vocabulary, and the eight themes of the word list are exactly
those eight slices. So the similarity handed to every method is very close to the answer, softened
by noise. Nothing about a real domain is this kind. If domain knowledge does not help here, it will
not help on a real stream, and that makes a null result worth reporting.

WHAT EACH METHOD IS GIVEN, AND IT IS THE SAME LIST FOR ALL OF THEM.

  every method              the records, keys and values as strings
  every method, named       the same sixty four word vectors
  sequential k-means        the true number of groups, which is eight, and which our method is never
                            told and has to discover
  leader clustering         a radius, reported twice: chosen on the test labels, which is not
                            available in use, and carried over from a separate stream
  ESCROW                    nothing else, and one run, because it has no calibrated parameter

HOW MEANING ENTERS EACH METHOD. For the baselines it enters as a vector per record, the mean of the
vectors of that record's six words, which is what a pipeline built on an embedding does. For our
method it enters through code/escrow/semantic.py, as a code and never as a score: the predictive
over values inside a block borrows counts from similar values, and at every symbol that semantic
code is mixed with the shipped categorical one at even odds, so no symbol can cost more than one bit
above the better of the two. That is weaker than the module's own guarantee, which is one bit over
the whole stream and needs the two codelengths of the finished stream. The engine needs a number per
record while the stream is still running, so the mixture is taken per symbol, and the price of the
weaker bound is that the null has to be measured rather than argued. Measuring it is the first row
of every table below. Nothing is tuned in either arm.

WHAT IS MEASURED. Across the ladder from noise 1.0 down to noise 0.0: the number of nodes K against
the true eight, the adjusted Rand index ARI against the planted groups, and how many records were
placed in a node at all. At signal 0 the truth is zero nodes, so K is the number to read there and
ARI is not.

WHAT WOULD REFUTE IT.
  Any node at signal 0 in the named condition. That would mean the semantic code is a similarity
  score in disguise: it would be manufacturing groups out of the fact that the words look alike,
  on a stream where the words were drawn uniformly and there is nothing to find.
  The named condition and the opaque condition disagreeing for a method that does not read strings.
  That would mean the two conditions are not the same data and every comparison here is void.
  The semantic arm under a similarity matrix that says every word is unrelated to every other
  failing to reproduce the shipped arm bit for bit. That would mean the wiring changes the method
  even when the domain knowledge is empty, and no number below could be attributed to meaning.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

import numpy as np

from escrow import batch as B
from escrow import codes as C
from escrow import engine as E
from escrow.codes import kt, naming_charge
from escrow.protocol import record_labels, run_stream
from escrow.provenance import stamped
from escrow.semantic import SCALE_GRID, mixture_bits
from e4_baseline_army import _ari
from e75_a_fixture_that_tests_the_values import GROUPS, PER_GROUP, stream

ROOT = os.environ.get("ESCROW_ROOT") or os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "results", "e79_does_domain_knowledge_help.json")

N = int(os.environ.get("E79_N", "3000"))
SEEDS = (0, 1, 2)
SIGNAL = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 0.9, 1.0)
RADII = (0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.05)
CENTRE_CAP = 512                # a budget on how many centres are kept, not a smaller radius
CAL_SEEDS = (100, 101)          # a separate stream, for the carried-over radius
CAL_SIGNAL = 0.7                # a level that is not on the ladder


# --------------------------------------------------------------------------- #
# The naming layer. Nothing below touches the data, only the strings that name it.
# --------------------------------------------------------------------------- #
KEY_NAMES = ["title_word", "caption_word", "tag", "related_term", "index_entry", "cross_reference"]
KEY_GLOSS = {
    "title_word": "a word from the entry's title",
    "caption_word": "a word from the picture caption",
    "tag": "a word the cataloguer attached as a tag",
    "related_term": "a word the entry lists as related",
    "index_entry": "the word this entry is filed under in the index",
    "cross_reference": "a word this entry points to elsewhere in the catalogue",
}
THEMES = [
    ("birds", ["sparrow", "eagle", "owl", "falcon", "heron", "pigeon", "raven", "swan"]),
    ("trees", ["oak", "maple", "birch", "willow", "cedar", "pine", "elm", "alder"]),
    ("metals", ["copper", "iron", "silver", "gold", "tin", "zinc", "lead", "nickel"]),
    ("instruments", ["violin", "flute", "drum", "harp", "trumpet", "cello", "oboe", "piano"]),
    ("vehicles", ["bicycle", "tram", "ferry", "wagon", "glider", "canoe", "scooter", "sledge"]),
    ("fruits", ["apple", "mango", "cherry", "plum", "grape", "peach", "lemon", "fig"]),
    ("weather", ["rain", "frost", "thunder", "mist", "hail", "breeze", "drizzle", "sunshine"]),
    ("tools", ["hammer", "chisel", "wrench", "pliers", "saw", "drill", "rasp", "clamp"]),
]
OPAQUE_VOCAB = [f"v{i}" for i in range(GROUPS * PER_GROUP)]
NAMED_VOCAB = [w for _, words in THEMES for w in words]
THEME_OF = [t for t, words in THEMES for _ in words]
VALUE_MAP = dict(zip(OPAQUE_VOCAB, NAMED_VOCAB))
KEY_MAP = {f"k{j}": KEY_NAMES[j] for j in range(6)}
M = len(OPAQUE_VOCAB)

# Four words are pulled toward a second sense, because real words have them and a real embedding
# reports them. An iron is a metal and also a thing you press a shirt with; a drum holds oil as well
# as a rhythm; a glider flies; a cherry is a fruit and also a tree.
CROSSINGS = {"iron": ("tools", 0.45), "drum": ("tools", 0.35),
             "glider": ("birds", 0.40), "cherry": ("trees", 0.45)}


def rename(recs):
    """The same records, written in the words of the catalogue. One to one, nothing else changes."""
    return [{KEY_MAP[k]: VALUE_MAP[v] for k, v in r.items()} for r in recs]


# --------------------------------------------------------------------------- #
# The two similarity sources, one per condition. Both are vectors over the same
# sixty four values, and both are read the same way.
# --------------------------------------------------------------------------- #
def named_vectors(dim=48, seed=17):
    """The stand-in for a text embedding of the sixty four words.

    Built from the vocabulary's own themes, so it is close to the answer, which is stated in the
    docstring above and is the point: this is the best case for domain knowledge, not a typical one.
    It is made imperfect on purpose. The tree anchor and the fruit anchor are correlated, and four
    words are pulled toward a second sense.
    """
    rng = np.random.default_rng(seed)
    names = [t for t, _ in THEMES]
    anchor = {t: rng.normal(size=dim) for t in names}
    for t in anchor:
        anchor[t] /= np.linalg.norm(anchor[t])
    anchor["fruits"] = anchor["fruits"] + 0.55 * anchor["trees"]      # a cherry really is a tree
    anchor["fruits"] /= np.linalg.norm(anchor["fruits"])
    vecs = {}
    for w, t in zip(NAMED_VOCAB, THEME_OF):
        v = anchor[t] + 0.8 * rng.normal(size=dim) / math.sqrt(dim)
        if w in CROSSINGS:
            other, pull = CROSSINGS[w]
            v = v + pull * anchor[other]
        vecs[w] = v / np.linalg.norm(v)
    return vecs


def opaque_vectors():
    """What a text embedding returns for strings that mean nothing: their surface form.

    Character trigrams of `v0` to `v63`, which is a real embedder's answer on tokens with no
    meaning, and is the same procedure applied to the strings this condition actually has. It is
    not blank: `v10` and `v15` share characters. The leak that creates is measured and reported
    rather than assumed away, because a spurious similarity is a hazard for the semantic code and
    the point of running it here is to see whether the method survives one.
    """
    grams = {}
    rows = {}
    for s in OPAQUE_VOCAB:
        pad = f"#{s}#"
        rows[s] = [pad[i:i + 3] for i in range(len(pad) - 2)]
        for gm in rows[s]:
            grams.setdefault(gm, len(grams))
    vecs = {}
    for s, gs in rows.items():
        v = np.zeros(len(grams))
        for gm in gs:
            v[grams[gm]] += 1.0
        vecs[s] = v / np.linalg.norm(v)
    return vecs


def similarity_report(vecs, vocab):
    """How much the similarity actually knows.

    Two numbers, because the first one on its own can mislead. The mean cosine inside a theme
    against across themes says how far apart the two populations sit. The share of words whose
    nearest neighbour is in their own theme says whether the similarity is usable, and it is the
    one to read: strings that share characters can have a wide mean gap and still put nobody next
    to anybody useful.
    """
    Xm = np.stack([vecs[w] for w in vocab])
    cos = Xm @ Xm.T
    th = np.array(THEME_OF)
    same = np.equal(th[:, None], th[None, :])
    off = ~np.eye(len(vocab), dtype=bool)
    nn = np.where(off, cos, -9.0).argmax(axis=1)
    return {"mean_cosine_same_theme": round(float(cos[same & off].mean()), 4),
            "mean_cosine_other_theme": round(float(cos[(~same) & off].mean()), 4),
            "gap": round(float(cos[same & off].mean() - cos[(~same) & off].mean()), 4),
            "nearest_neighbour_in_the_same_theme": round(float((th[nn] == th).mean()), 4),
            "chance_level_for_that": round(7.0 / (len(vocab) - 1.0), 4)}


def kernel_stack(vecs, vocab):
    """K[s][v][w] in [0, 1] with K[v][v] = 1, one layer per scale in the module's own grid.

    This is escrow.semantic._kernel, held as one array so the whole grid is read in a single
    multiply. Negative cosine is clamped to zero, because a negative similarity is not evidence
    against a value, it is an absence of evidence for it. The grid is not a chosen bandwidth: the
    predictive is a uniform mixture over it, and a mixture is not a choice.
    """
    Xm = np.stack([vecs[w] for w in vocab])
    cos = np.clip(Xm @ Xm.T, 0.0, 1.0)
    np.fill_diagonal(cos, 1.0)
    ks = np.stack([cos ** s for s in SCALE_GRID])
    for layer in ks:
        np.fill_diagonal(layer, 1.0)
    return ks


IDENTITY_STACK = np.stack([np.eye(M) for _ in SCALE_GRID])


# --------------------------------------------------------------------------- #
# The semantic code, wired into the run.
#
# WHERE IT ACTS, AND WHY ONLY THERE. Inside a value block the shipped code spends three stages: is
# this value new to the block, if it is not then which of the ones already held, and if it is then
# name it out of the key's inventory. The semantic code changes the middle stage only, so a value
# borrows counts from the values it resembles. That is the scope escrow/semantic.py itself argues
# for at its foot: E56 and E57 measured the naming charge at 8.7 bits against a 293 bit barrier, so
# semantics acting on naming could not move a decision, and the decision lives in the predictive.
#
# The two codes are combined at even odds at every step, which is the module's mixture written per
# symbol. It is a valid predictive because both parts are, and a symbol under it costs at most one
# bit more than it costs under the better of the two. Over T symbols that bound is T bits and not
# one bit, which is weaker than mixing the two finished codelengths, and it is taken knowingly: the
# engine reads evidence as the stream runs. Nothing here is calibrated.
#
# It acts on the evidence the engine reads and not on the price the batch objective charges, so a
# node minted with the help of the domain knowledge still has to survive the shipped price. That is
# deliberate and it is the reason the null at signal 0 is a real test rather than a formality.
# --------------------------------------------------------------------------- #
_KST2 = None                       # (len(SCALE_GRID) * M, M), the kernel stack flattened
_S = len(SCALE_GRID)
_shipped_cost = C.ValueBlock.cost
_shipped_intern = E.KeyInfo.intern


def _semantic_cost(self, vid, naming):
    N = self.N
    if N == 0:
        return naming_charge(naming, self.u)
    c = self.counts.get(vid)
    if c is None:                                   # new to this block: stages one and three only
        return kt(self.u, N, 2) + naming_charge(naming, self.u)
    stage_one = kt(N - self.u, N, 2)
    cat = stage_one + kt(c, N, self.u)
    u = self.u
    if u < 2:                                       # nothing to borrow from; the codes agree exactly
        return cat
    ks = np.fromiter(self.counts.keys(), dtype=np.intp, count=u)
    vs = np.fromiter(self.counts.values(), dtype=np.float64, count=u)
    cv = np.zeros(M)
    cv[ks] = vs
    mask = np.zeros(M)
    mask[ks] = 1.0
    w = (_KST2 @ cv).reshape(_S, M)                 # smoothed counts, one row per scale
    p = float(np.mean((w[:, vid] + 0.5) / (w @ mask + u / 2.0)))
    return mixture_bits(cat, stage_one - math.log2(p))


def _fixed_intern(self, value):
    """Every key gives a word the same identifier, taken from the declared vocabulary.

    This is bookkeeping and not a modelling choice: the naming charge still counts only the values
    this key has actually seen, so no cost changes. It exists because one similarity matrix has to
    serve all six keys, and the shipped interning numbers a value by when that key first met it, so
    `sparrow` is 3 at one key and 17 at another. The self-check below measures what this alone does
    to the two arms that carry no semantics, so the reader can see it is not doing the work.
    """
    vid = self.inventory.get(value)
    if vid is None:
        vid = _FIXED_IDS[value]
        self.inventory[value] = vid
    return vid


_FIXED_IDS: dict = {}


class semantics_on:
    def __init__(self, kernel, vocab):
        self.kernel, self.vocab = kernel, vocab

    def __enter__(self):
        global _KST2, _FIXED_IDS
        _FIXED_IDS = {w: i for i, w in enumerate(self.vocab)}
        _KST2 = np.ascontiguousarray(self.kernel.reshape(_S * M, M))
        C.ValueBlock.cost = _semantic_cost
        E.KeyInfo.intern = _fixed_intern

    def __exit__(self, *a):
        C.ValueBlock.cost = _shipped_cost
        E.KeyInfo.intern = _shipped_intern
        return False


class fixed_ids_only:
    """The interning change with no semantics, so its own effect can be read off."""

    def __init__(self, vocab):
        self.vocab = vocab

    def __enter__(self):
        global _FIXED_IDS
        _FIXED_IDS = {w: i for i, w in enumerate(self.vocab)}
        E.KeyInfo.intern = _fixed_intern

    def __exit__(self, *a):
        E.KeyInfo.intern = _shipped_intern
        return False


# --------------------------------------------------------------------------- #
# What the domain knowledge is worth in BITS, on the true grouping, with no search in the way.
#
# WHY THIS IS HERE. The semantic code above enters the evidence the engine reads and not the price
# the batch objective charges. That is a real limit of this wiring and it has to be separated from
# the question. If the named condition changes nothing, there are two possible reasons: the meaning
# is worth nothing in bits on this stream, or it is worth something and this wiring does not carry
# it as far as the price. The measurement below settles which.
#
# It prices the value blocks of the true eight group partition, and of the state the method returns
# when it abstains, which is everything in the background. Both are priced twice, once under the
# Krichevsky-Trofimov predictive the engine ships and once under the same predictive with its counts
# smoothed by the similarity, which is escrow.semantic.block_value_bits over the whole vocabulary.
# The number that decides whether a grouping is worth minting is the margin, the bits the background
# spends above what the eight blocks spend. If the similarity raises that margin, the meaning is
# worth something to the criterion; if it does not, the meaning is worth nothing here and no wiring
# would rescue it.
# --------------------------------------------------------------------------- #
def _seq_bits(seq, kst2, alpha=0.5):
    """Prequential bits for one sequence of value ids over the whole vocabulary.

    With kst2 None this is the shipped estimator, P(v) = (c_v + 1/2) / (N + M/2). With a kernel it
    is the same estimator with the counts smoothed by similarity, averaged over the scale grid, so
    a value borrows strength from the values it resembles. Both are normalised over the same
    alphabet at every step, so both are valid codes and the difference between them is a difference
    between two descriptions of the same data.
    """
    counts = np.zeros(M)
    bits = 0.0
    if kst2 is None:
        total = 0.0
        for v in seq:
            bits -= math.log2((counts[v] + alpha) / (total + alpha * M))
            counts[v] += 1.0
            total += 1.0
        return bits
    for v in seq:
        w = (kst2 @ counts).reshape(_S, M)
        p = float(np.mean((w[:, v] + alpha) / (w.sum(axis=1) + alpha * M)))
        bits -= math.log2(p)
        counts[v] += 1.0
    return bits


def value_bits_of_states(recs, truth, vocab, kst2):
    """Bits for the value blocks of the true partition and of the all background state."""
    ids = {w: i for i, w in enumerate(vocab)}
    keys = sorted({k for r in recs for k in r})
    by_group = {}
    background = {k: [] for k in keys}
    for r, g in zip(recs, truth):
        for k, v in r.items():
            by_group.setdefault((g, k), []).append(ids[v])
            background[k].append(ids[v])
    truth_bits = sum(_seq_bits(seq, kst2) for seq in by_group.values())
    bg_bits = sum(_seq_bits(seq, kst2) for seq in background.values())
    return truth_bits, bg_bits


# --------------------------------------------------------------------------- #
# The methods.
# --------------------------------------------------------------------------- #
def escrow_row(recs, truth, split):
    B.SPLIT_MOVES = bool(split)
    g, _ = run_stream(recs)
    lab = record_labels(g, len(recs))
    return {"K": int(g.K), "ARI": round(_ari(truth, lab), 4),
            "covered": sum(1 for x in lab if x != -1)}


def multihot(recs):
    """Keys and values as indicator columns. Reads a string as a symbol and never as a word, so
    this representation is the same matrix in both conditions, entry for entry."""
    ix = {}
    for r in recs:                                  # first-appearance order, which the naming layer
        for k, v in r.items():                      # carries over exactly, so the two conditions
            ix.setdefault(f"KEY:{k}", len(ix))      # build the very same matrix and not a permuted
            ix.setdefault(f"{k}={v}", len(ix))      # one
    X = np.zeros((len(recs), len(ix)))
    for i, r in enumerate(recs):
        for k, v in r.items():
            X[i, ix[f"{k}={v}"]] = 1.0
            X[i, ix[f"KEY:{k}"]] = 1.0
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def embed_mean(recs, vecs):
    """One vector per record: the mean of the vectors of the words it carries. This is the whole of
    what an embedding pipeline does with a record, and it is the channel the domain knowledge takes
    into every baseline."""
    dim = len(next(iter(vecs.values())))
    X = np.zeros((len(recs), dim))
    for i, r in enumerate(recs):
        for v in r.values():
            X[i] += vecs[v]
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


def _order(n, seed):
    return np.random.default_rng(seed).permutation(n)


def leader(X, radius, seed):
    """Leader clustering, Hartigan 1975. Attach to the first centre within the radius, else become a
    centre. One pass, and the radius is the whole model."""
    o = _order(len(X), seed)
    centres = np.zeros((CENTRE_CAP, X.shape[1]))
    nc = 0
    lab = [0] * len(X)
    for idx in o:
        x = X[idx]
        j = -1
        if nc:
            d = np.linalg.norm(centres[:nc] - x, axis=1)
            hit = np.nonzero(d <= radius)[0]
            if len(hit):
                j = int(hit[0])
        if j < 0:
            if nc >= len(centres):
                j = int(np.argmin(np.linalg.norm(centres - x, axis=1)))
            else:
                centres[nc] = x
                j = nc
                nc += 1
        lab[idx] = j
    return lab


def seq_kmeans(X, k, seed):
    """MacQueen sequential k-means, one pass, told the true number of groups."""
    o = _order(len(X), seed)
    centres = np.zeros((k, X.shape[1]))
    counts = np.zeros(k)
    nc = 0
    lab = [0] * len(X)
    for idx in o:
        x = X[idx]
        if nc < k:
            centres[nc] = x
            counts[nc] = 1
            lab[idx] = nc
            nc += 1
            continue
        j = int(np.argmin(np.linalg.norm(centres - x, axis=1)))
        counts[j] += 1
        centres[j] += (x - centres[j]) / counts[j]
        lab[idx] = j
    return lab


def scored(truth, lab):
    return {"K": len(set(lab)), "ARI": round(_ari(truth, lab), 4), "covered": len(lab)}


def mean_row(rows):
    return {"K_mean": round(sum(r["K"] for r in rows) / len(rows), 2),
            "K_per_seed": [r["K"] for r in rows],
            "ARI_mean": round(sum(r["ARI"] for r in rows) / len(rows), 4),
            "covered": sum(r["covered"] for r in rows)}


# --------------------------------------------------------------------------- #
def transferred_radius(vecs, condition, kind):
    """A radius chosen on a separate stream, which is what a practitioner actually carries.

    The calibration stream is the same fixture at a signal level that is not on the ladder and at
    seeds the ladder never uses. Its labels are used, which is allowed, because they are not the
    labels anything is reported against.
    """
    mats = []
    for s in CAL_SEEDS:
        recs, truth = stream(CAL_SIGNAL, s, n=N)
        if condition == "named":
            recs = rename(recs)
        mats.append((s, multihot(recs) if kind == "multihot" else embed_mean(recs, vecs), truth))
    best, best_r = -2.0, RADII[0]
    for r in RADII:
        v = sum(_ari(truth, leader(X, r, s)) for s, X, truth in mats) / len(mats)
        if v > best:
            best, best_r = v, r
    return best_r, round(best, 4)


def main():
    t0 = time.time()
    vec_by_condition = {"opaque": opaque_vectors(), "named": named_vectors()}
    vocab_by_condition = {"opaque": OPAQUE_VOCAB, "named": NAMED_VOCAB}
    kern_by_condition = {c: kernel_stack(vec_by_condition[c], vocab_by_condition[c])
                         for c in vec_by_condition}

    report = {
        "experiment": "E79 does telling a method what the keys and values mean help it",
        "question": ("a language model is handed real words and already knows what they mean. Does "
                     "that knowledge help, does it help our method, and does it help our method as "
                     "much as it helps a method built on an embedding"),
        "records": N, "groups": GROUPS, "seeds": list(SEEDS), "ladder_signal": list(SIGNAL),
        "conditions": {
            "opaque": "keys k0 to k5, values v0 to v63, the fixture as it ships",
            "named": ("the same records with the keys and values rewritten one to one into the "
                      "words of a catalogue, so only the strings differ"),
        },
        "domain_knowledge": {
            "key_gloss": KEY_GLOSS,
            "key_gloss_note": ("every entry carries all six keys, so the key names cannot separate "
                               "the groups, and this is true of the fixture by construction"),
            "value_similarity": ("a unit vector per value word, from which any two words' cosine is "
                                 "read"),
            "stand_in": ("there is no embedding model and no network on this machine, so the named "
                         "vectors are written down from the vocabulary's own themes and a real "
                         "embedding would replace them. The opaque vectors are character trigrams "
                         "of the meaningless strings, which is what a real embedder returns for "
                         "them"),
            "themes": {t: ws for t, ws in THEMES},
            "declared_second_senses": {w: f"pulled toward {t} by {p}" for w, (t, p) in
                                       CROSSINGS.items()},
            "this_is_the_best_case": ("in this fixture a group is a slice of the vocabulary and the "
                                      "eight themes are exactly those eight slices, so the "
                                      "similarity given to every method is close to the answer. A "
                                      "real domain is never this kind, so a result that domain "
                                      "knowledge does not help here is the stronger result"),
        },
        "what_each_method_was_given": {
            "escrow_base": "the records; the split move off; no similarity; one run; no knob",
            "escrow_split": "the records; the split move on; no similarity; one run; no knob",
            "escrow_semantic_base": "the records and the similarity; the split move off; one run",
            "escrow_semantic_split": "the records and the similarity; the split move on; one run",
            "leader_multihot": "the records as indicator columns, and a radius",
            "leader_embedding": "the same records as a mean of word vectors, and a radius",
            "kmeans_multihot": "the records as indicator columns, and the true number of groups",
            "kmeans_embedding": "the same records as a mean of word vectors, and the true number",
        },
        "similarity_quality": {c: similarity_report(vec_by_condition[c], vocab_by_condition[c])
                               for c in vec_by_condition},
        "self_checks": {},
        "transferred_radius": {},
        "ladder": {},
    }

    print("E79 does domain knowledge help\n")
    for c in ("opaque", "named"):
        q = report["similarity_quality"][c]
        print(f"  similarity, {c:>6}: same theme {q['mean_cosine_same_theme']:+.4f}   "
              f"other theme {q['mean_cosine_other_theme']:+.4f}   gap {q['gap']:+.4f}")

    # ---- self-check 1: the two conditions are the same data ---------------- #
    recs0, truth0 = stream(0.8, 0, n=N)
    named0 = rename(recs0)
    same_data = all(sorted(a.items()) != sorted(b.items()) for a, b in zip(recs0, named0))
    X_op, X_nm = multihot(recs0), multihot(named0)
    d_op = np.linalg.norm(X_op[:50] @ X_op[:50].T - X_nm[:50] @ X_nm[:50].T)
    report["self_checks"]["multihot_is_the_same_geometry_in_both_conditions"] = {
        "max_gram_difference_on_50_records": round(float(d_op), 12), "strings_did_change": same_data}
    assert d_op < 1e-9, "the naming layer changed the data, not only the strings"

    # ---- self-check 2: an empty similarity reproduces the shipped arm ------- #
    checks = []
    for sig in (0.3, 0.8, 1.0):
        rc, tr = stream(sig, 0, n=N)
        for split in (False, True):
            base = escrow_row(rc, tr, split)
            with fixed_ids_only(OPAQUE_VOCAB):
                fixed = escrow_row(rc, tr, split)
            with semantics_on(IDENTITY_STACK, OPAQUE_VOCAB):
                ident = escrow_row(rc, tr, split)
            checks.append({"signal": sig, "split_move": split, "shipped": base,
                           "fixed_ids_only": fixed, "identity_similarity": ident,
                           "identity_matches_fixed_ids": ident == fixed,
                           "fixed_ids_matches_shipped": fixed == base})
            assert ident == fixed, "the semantic wiring changed the method with an empty similarity"
    report["self_checks"]["empty_similarity_reproduces_the_shipped_arm"] = checks
    report["self_checks"]["what_the_identifier_bookkeeping_alone_does"] = (
        "the semantic arms number a value the same way at every key, which the shipped engine does "
        "not. fixed_ids_only above is that change with no semantics, so the reader can see its "
        "size. Where it moves a number, the semantic gain is read against fixed_ids_only and not "
        "against the shipped arm")
    print("\n  self-check: an empty similarity reproduces the shipped arm exactly, both arms")
    print("  self-check: the named records are the same data as the opaque ones")

    # ---- the carried-over radius, chosen on a separate stream --------------- #
    for c in ("opaque", "named"):
        for kind in ("multihot", "embedding"):
            r, v = transferred_radius(vec_by_condition[c], c, kind)
            report["transferred_radius"][f"{c}/{kind}"] = {"radius": r, "ARI_on_that_stream": v,
                                                           "chosen_on": f"signal {CAL_SIGNAL}, "
                                                                        f"seeds {list(CAL_SEEDS)}"}
    print("  carried-over radii:", {k: v["radius"]
                                    for k, v in report["transferred_radius"].items()})

    # ---- the ladder -------------------------------------------------------- #
    arms = ["escrow_base", "escrow_split", "escrow_semantic_base", "escrow_semantic_split",
            "leader_multihot_oracle", "leader_multihot_transferred",
            "leader_embedding_oracle", "leader_embedding_transferred",
            "kmeans_multihot", "kmeans_embedding"]
    for c in ("opaque", "named"):
        report["ladder"][c] = []

    kst2_by_condition = {c: np.ascontiguousarray(kern_by_condition[c].reshape(_S * M, M))
                         for c in kern_by_condition}
    for f in SIGNAL:
        streams = []
        for s in SEEDS:
            recs, truth = stream(f, s, n=N)
            streams.append((s, recs, truth))
        for c in ("opaque", "named"):
            vecs, vocab = vec_by_condition[c], vocab_by_condition[c]
            kern = kern_by_condition[c]
            rows = {a: [] for a in arms}
            oracle_pick = {}
            margins = []
            for s, recs_op, truth in streams:
                recs = rename(recs_op) if c == "named" else recs_op
                rows["escrow_base"].append(escrow_row(recs, truth, False))
                rows["escrow_split"].append(escrow_row(recs, truth, True))
                for split, name in ((False, "escrow_semantic_base"),
                                    (True, "escrow_semantic_split")):
                    with semantics_on(kern, vocab):
                        rows[name].append(escrow_row(recs, truth, split))
                Xs = {"multihot": multihot(recs), "embedding": embed_mean(recs, vecs)}
                for kind, X in Xs.items():
                    per_r = {r: scored(truth, leader(X, r, s)) for r in RADII}
                    best_r = max(per_r, key=lambda r: per_r[r]["ARI"])
                    rows[f"leader_{kind}_oracle"].append(per_r[best_r])
                    oracle_pick.setdefault(kind, []).append(best_r)
                    tr = report["transferred_radius"][f"{c}/{kind}"]["radius"]
                    rows[f"leader_{kind}_transferred"].append(per_r[tr])
                    rows[f"kmeans_{kind}"].append(scored(truth, seq_kmeans(X, GROUPS, s)))
                tc, bc = value_bits_of_states(recs, truth, vocab, None)
                ts, bs = value_bits_of_states(recs, truth, vocab, kst2_by_condition[c])
                margins.append({"seed": s, "margin_categorical": round(bc - tc, 1),
                                "margin_semantic": round(bs - ts, 1),
                                "change": round((bs - ts) - (bc - tc), 1)})
            row = {"signal": f, "noise": round(1 - f, 2),
                   "oracle_radius_per_seed": oracle_pick,
                   "value_bits_margin_of_the_true_grouping": {
                       "per_seed": margins,
                       "mean_margin_categorical": round(
                           sum(m["margin_categorical"] for m in margins) / len(margins), 1),
                       "mean_margin_semantic": round(
                           sum(m["margin_semantic"] for m in margins) / len(margins), 1),
                       "mean_change": round(sum(m["change"] for m in margins) / len(margins), 1)},
                   "arms": {a: mean_row(rows[a]) for a in arms}}
            report["ladder"][c].append(row)

        o = report["ladder"]["opaque"][-1]["arms"]
        nm = report["ladder"]["named"][-1]["arms"]
        for a in ("escrow_base", "escrow_split", "kmeans_multihot",
                  "leader_multihot_oracle", "leader_multihot_transferred"):
            assert o[a] == nm[a], (f"{a} reads strings as words: it moved between the two "
                                   f"conditions at signal {f}")
        mo = report["ladder"]["opaque"][-1]["value_bits_margin_of_the_true_grouping"]
        mn = report["ladder"]["named"][-1]["value_bits_margin_of_the_true_grouping"]
        assert abs(mo["mean_margin_categorical"] - mn["mean_margin_categorical"]) < 1.0, (
            "the categorical value bits moved between the conditions, so the data is not the same")
        print(f"\n  signal {f:.2f}  noise {1-f:.2f}   "
              f"(methods that ignore meaning are identical in both conditions: checked)")
        print(f"    value bits the true grouping saves over abstaining: "
              f"as symbols {mo['mean_margin_categorical']:+.0f}, "
              f"with the meaningless similarity {mo['mean_margin_semantic']:+.0f}, "
              f"with the meaning {mn['mean_margin_semantic']:+.0f}")
        print(f"    {'arm':<32} {'opaque K':>9} {'opaque ARI':>11} {'named K':>9} {'named ARI':>11}")
        for a in arms:
            print(f"    {a:<32} {o[a]['K_mean']:>9.2f} {o[a]['ARI_mean']:>11.4f} "
                  f"{nm[a]['K_mean']:>9.2f} {nm[a]['ARI_mean']:>11.4f}")

    # ---- reading the ladder ------------------------------------------------ #
    def at(cond, sig, arm, field):
        for r in report["ladder"][cond]:
            if r["signal"] == sig:
                return r["arms"][arm][field]
        return None

    gains = {}
    for arm in arms:
        per = []
        for sig in SIGNAL:
            if sig == 0.0:
                continue
            per.append({"signal": sig,
                        "opaque": at("opaque", sig, arm, "ARI_mean"),
                        "named": at("named", sig, arm, "ARI_mean"),
                        "gain": round(at("named", sig, arm, "ARI_mean")
                                      - at("opaque", sig, arm, "ARI_mean"), 4)})
        best = max(per, key=lambda x: x["gain"])
        gains[arm] = {"best_gain": best["gain"], "at_signal": best["signal"],
                      "mean_gain_over_the_ladder": round(sum(p["gain"] for p in per) / len(per), 4),
                      "per_signal": per}
    report["what_naming_buys"] = gains

    # The other reading of the same question, and the one to quote: inside the named condition,
    # what does a method gain by being told what the words mean rather than treating them as
    # symbols? For us that is the semantic code against the shipped one. For a baseline it is the
    # embedding representation against the indicator one.
    pairs = {"ours_with_the_split_move": ("escrow_semantic_split", "escrow_split"),
             "ours_without_the_split_move": ("escrow_semantic_base", "escrow_base"),
             "leader_radius_on_the_test_labels": ("leader_embedding_oracle",
                                                  "leader_multihot_oracle"),
             "leader_radius_carried_over": ("leader_embedding_transferred",
                                            "leader_multihot_transferred"),
             "kmeans_told_the_true_k": ("kmeans_embedding", "kmeans_multihot")}
    adds = {}
    for label, (with_meaning, without) in pairs.items():
        per = []
        for sig in SIGNAL:
            if sig == 0.0:
                continue
            a = at("named", sig, with_meaning, "ARI_mean")
            b = at("named", sig, without, "ARI_mean")
            per.append({"signal": sig, "with_meaning": a, "as_symbols": b,
                        "gain": round(a - b, 4)})
        best = max(per, key=lambda x: x["gain"])
        adds[label] = {"best_gain": best["gain"], "at_signal": best["signal"],
                       "mean_gain": round(sum(p["gain"] for p in per) / len(per), 4),
                       "per_signal": per}
    report["what_meaning_adds_inside_the_named_condition"] = adds

    null = {c: {"K_at_signal_0": at(c, 0.0, "escrow_semantic_split", "K_mean"),
                "K_at_signal_0_base": at(c, 0.0, "escrow_semantic_base", "K_mean"),
                "K_at_signal_0_no_semantics": at(c, 0.0, "escrow_split", "K_mean")}
            for c in ("opaque", "named")}
    report["the_null"] = null

    bits = []
    for sig in SIGNAL:
        o = next(r for r in report["ladder"]["opaque"] if r["signal"] == sig)
        n_ = next(r for r in report["ladder"]["named"] if r["signal"] == sig)
        bits.append({"signal": sig,
                     "margin_as_symbols": o["value_bits_margin_of_the_true_grouping"][
                         "mean_margin_categorical"],
                     "margin_with_a_meaningless_similarity": o[
                         "value_bits_margin_of_the_true_grouping"]["mean_margin_semantic"],
                     "margin_with_the_meaning": n_["value_bits_margin_of_the_true_grouping"][
                         "mean_margin_semantic"],
                     "meaning_buys_bits": round(
                         n_["value_bits_margin_of_the_true_grouping"]["mean_margin_semantic"]
                         - o["value_bits_margin_of_the_true_grouping"][
                             "mean_margin_categorical"], 1)})
    report["what_meaning_is_worth_in_bits"] = {
        "measured_on": ("the value blocks of the true eight group partition against the value "
                        "blocks of the all background state, which is what the method returns when "
                        "it abstains. A positive margin is the truth costing fewer bits than "
                        "abstaining"),
        "why_it_is_here": ("the semantic code enters the evidence the engine reads and not the "
                           "price the objective charges, so a null in the ARI column could be the "
                           "meaning being worthless or the wiring being short. This column is the "
                           "criterion with no search in the way, so it says which"),
        "per_signal": bits}

    ours = max(gains["escrow_semantic_split"]["best_gain"], gains["escrow_semantic_base"]["best_gain"])
    theirs = max(gains["leader_embedding_oracle"]["best_gain"],
                 gains["kmeans_embedding"]["best_gain"])
    report["headline"] = {
        "null_holds_in_the_named_condition": (null["named"]["K_at_signal_0"] == 0
                                              and null["named"]["K_at_signal_0_base"] == 0),
        "best_ARI_gain_from_naming_for_our_method": ours,
        "best_ARI_gain_from_naming_for_a_name_using_baseline": theirs,
        "naming_helps_us_more": bool(ours >= theirs),
        "where_it_helps_us_most": gains["escrow_semantic_split"]["at_signal"],
        "where_it_helps_the_embedding_baseline_most": gains["leader_embedding_oracle"]["at_signal"],
        "mean_gain_our_method": gains["escrow_semantic_split"]["mean_gain_over_the_ladder"],
        "mean_gain_embedding_leader_oracle": gains["leader_embedding_oracle"][
            "mean_gain_over_the_ladder"],
        "mean_gain_embedding_kmeans_told_k": gains["kmeans_embedding"][
            "mean_gain_over_the_ladder"],
        "meaning_over_symbols_ours_split": adds["ours_with_the_split_move"]["mean_gain"],
        "meaning_over_symbols_ours_base": adds["ours_without_the_split_move"]["mean_gain"],
        "meaning_over_symbols_leader_oracle": adds["leader_radius_on_the_test_labels"]["mean_gain"],
        "meaning_over_symbols_leader_carried_over": adds["leader_radius_carried_over"]["mean_gain"],
        "meaning_over_symbols_kmeans": adds["kmeans_told_the_true_k"]["mean_gain"],
        "best_meaning_buys_bits": max(b["meaning_buys_bits"] for b in bits),
        "at_signal": max(bits, key=lambda b: b["meaning_buys_bits"])["signal"],
        "abstention_margin_at_signal_0_as_symbols": bits[0]["margin_as_symbols"],
        "abstention_margin_at_signal_0_with_the_meaning": bits[0]["margin_with_the_meaning"],
        "abstention_margin_stays_negative_at_signal_0": (bits[0]["margin_with_the_meaning"] < 0
                                                         and bits[0]["margin_as_symbols"] < 0),
        "reading": ("domain knowledge here is close to the answer, because the word themes are the "
                    "planted groups. Read the gains as a ceiling on what meaning can buy, not as "
                    "what it would buy on a real stream"),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    print("\n" + json.dumps(report["headline"], indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(stamped(report), open(OUT, "w"), indent=2)
    print("written", OUT)


if __name__ == "__main__":
    main()
