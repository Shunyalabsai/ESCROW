"""Independent integer arithmetic encoder and decoder for the reference description.

The encoder does not call the scorer. Sequential integer frequencies integrate to
the scorer's closed-form block probabilities. A checksum detects corrupt files.
Node identifiers in the decoded state are canonical; stable UI ids are metadata.
"""
from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from .model import HierarchyState, HierarchyNode

MAGIC = b"EHR03"
FULL = 1 << 64
HALF, QUARTER = FULL >> 1, FULL >> 2


class BitWriter:
    def __init__(self):
        self.bits = []

    def bit(self, b):
        self.bits.append(int(b))

    def uint(self, n):
        word = bin(n + 1)[2:]
        self.bits.extend([0] * (len(word) - 1))
        self.bits.extend(int(c) for c in word)

    def raw(self, data):
        for byte in data:
            for i in range(7, -1, -1):
                self.bit((byte >> i) & 1)

    def string(self, value):
        raw = value.encode("utf-8")
        self.uint(len(raw))
        self.raw(raw)

    def bytes(self):
        bits = self.bits + [0] * ((-len(self.bits)) % 8)
        return bytes(sum(bits[i + j] << (7 - j) for j in range(8))
                     for i in range(0, len(bits), 8))


class BitReader:
    def __init__(self, data):
        self.data, self.pos = data, 0

    def bit(self, pad=False):
        if self.pos >= len(self.data) * 8:
            if pad:
                self.pos += 1
                return 0
            raise ValueError("truncated categorical description")
        p = self.pos
        self.pos += 1
        return (self.data[p // 8] >> (7 - p % 8)) & 1

    def uint(self):
        zeros = 0
        while self.bit() == 0:
            zeros += 1
            if zeros > 63:
                raise ValueError("integer header is too large")
        n = 1
        for _ in range(zeros):
            n = 2 * n + self.bit()
        return n - 1

    def raw(self, n):
        return bytes(sum(self.bit() << (7 - i) for i in range(8)) for _ in range(n))

    def string(self):
        return self.raw(self.uint()).decode("utf-8")


class ArithmeticEncoder:
    def __init__(self, writer):
        self.writer, self.lo, self.hi, self.pending = writer, 0, FULL - 1, 0

    def emit(self, b):
        self.writer.bit(b)
        for _ in range(self.pending):
            self.writer.bit(1 - b)
        self.pending = 0

    def symbol(self, symbol, frequencies):
        if len(frequencies) == 1:
            return
        total = sum(frequencies)
        lower = sum(frequencies[:symbol])
        width = self.hi - self.lo + 1
        self.hi = self.lo + width * (lower + frequencies[symbol]) // total - 1
        self.lo += width * lower // total
        while True:
            if self.hi < HALF:
                self.emit(0)
            elif self.lo >= HALF:
                self.emit(1)
                self.lo -= HALF
                self.hi -= HALF
            elif self.lo >= QUARTER and self.hi < 3 * QUARTER:
                self.pending += 1
                self.lo -= QUARTER
                self.hi -= QUARTER
            else:
                break
            self.lo *= 2
            self.hi = self.hi * 2 + 1

    def finish(self):
        self.pending += 1
        self.emit(0 if self.lo < QUARTER else 1)


class ArithmeticDecoder:
    def __init__(self, reader):
        self.reader, self.lo, self.hi, self.value = reader, 0, FULL - 1, 0
        for _ in range(64):
            self.value = self.value * 2 + reader.bit(pad=True)

    def symbol(self, frequencies):
        if len(frequencies) == 1:
            return 0
        total = sum(frequencies)
        width = self.hi - self.lo + 1
        target = ((self.value - self.lo + 1) * total - 1) // width
        lower, symbol = 0, 0
        while target >= lower + frequencies[symbol]:
            lower += frequencies[symbol]
            symbol += 1
        self.hi = self.lo + width * (lower + frequencies[symbol]) // total - 1
        self.lo += width * lower // total
        while True:
            if self.hi < HALF:
                pass
            elif self.lo >= HALF:
                self.lo -= HALF
                self.hi -= HALF
                self.value -= HALF
            elif self.lo >= QUARTER and self.hi < 3 * QUARTER:
                self.lo -= QUARTER
                self.hi -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.lo *= 2
            self.hi = self.hi * 2 + 1
            self.value = self.value * 2 + self.reader.bit(pad=True)
        return symbol


def _write_kt(coder, symbol, counts):
    coder.symbol(symbol, [2 * c + 1 for c in counts])
    counts[symbol] += 1


def _read_kt(coder, counts):
    symbol = coder.symbol([2 * c + 1 for c in counts])
    counts[symbol] += 1
    return symbol


def encode(state):
    state.validate()
    n, keys, order = len(state.records), state.keys, state.topological()
    ids = {nid: i + 1 for i, nid in enumerate(order)}
    hist = {key: Counter(r[key] for r in state.records if key in r) for key in keys}
    vocabs = {key: sorted(hist[key]) for key in keys}
    writer = BitWriter()
    writer.raw(MAGIC)
    writer.uint(n)
    writer.uint(len(keys))
    for key in keys:
        writer.string(key)
        writer.uint(len(vocabs[key]))
        for value in vocabs[key]:
            writer.string(value)
            writer.uint(hist[key][value])
    writer.uint(len(order))
    coder = ArithmeticEncoder(writer)
    parent_counts = [0, 0]
    for i, nid in enumerate(order):
        node = state.nodes[nid]
        for p in order[:i]:
            _write_kt(coder, int(p in node.parents), parent_counts)
        counts = [0, 0]
        for key in keys:
            _write_kt(coder, int(key in node.support), counts)
        counts = [0, 0]
        for r in sorted(state.eligible_rows(nid)):
            _write_kt(coder, int(r in node.members), counts)
    for key in keys:
        supporting = [nid for nid in order if key in state.nodes[nid].support]
        routes, pres, vals = {}, defaultdict(lambda: [0, 0]), {}
        vocab = vocabs[key]
        idx = {v: i for i, v in enumerate(vocab)}
        bg_value = [2 * hist[key][v] + 1 for v in vocab]
        pubs = sum(hist[key].values())
        bg_pres = [2 * (n - pubs) + 1, 2 * pubs + 1]
        for r, record in enumerate(state.records):
            eligible = (0,) + tuple(nid for nid in supporting if r in state.nodes[nid].members)
            owner = state.owners[r].get(key, 0)
            counts = routes.setdefault(eligible, [0] * len(eligible))
            _write_kt(coder, eligible.index(owner), counts)
            published = int(key in record)
            if owner == 0:
                coder.symbol(published, bg_pres)
            else:
                _write_kt(coder, published, pres[owner])
            if published:
                value = idx[record[key]]
                if owner == 0:
                    coder.symbol(value, bg_value)
                else:
                    _write_kt(coder, value, vals.setdefault(owner, [0] * len(vocab)))
    coder.finish()
    body = writer.bytes()
    # Explicit framing makes complete descriptions self-delimiting, including the checksum.
    data = len(body).to_bytes(8, "big") + body
    return data + hashlib.sha256(data).digest()


def decode(data):
    if len(data) < 46 or hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("description checksum mismatch")
    if int.from_bytes(data[:8], "big") != len(data) - 40:
        raise ValueError("description length does not match its frame")
    reader = BitReader(data[8:-32])
    if reader.raw(len(MAGIC)) != MAGIC:
        raise ValueError("unknown categorical description format")
    n, size = reader.uint(), reader.uint()
    keys, vocab, hist = [], {}, {}
    for _ in range(size):
        key = reader.string()
        keys.append(key)
        vocab[key], hist[key] = [], {}
        for _ in range(reader.uint()):
            value, count = reader.string(), reader.uint()
            vocab[key].append(value)
            hist[key][value] = count
    count_nodes = reader.uint()
    state = HierarchyState([{} for _ in range(n)], {}, [{} for _ in range(n)], count_nodes + 1)
    coder = ArithmeticDecoder(reader)
    parent_counts = [0, 0]
    for nid in range(1, count_nodes + 1):
        parents = {p for p in range(1, nid) if _read_kt(coder, parent_counts)}
        counts = [0, 0]
        support = {key for key in keys if _read_kt(coder, counts)}
        node = HierarchyNode(nid, set(), support, parents)
        state.nodes[nid] = node
        counts = [0, 0]
        node.members = {r for r in sorted(state.eligible_rows(nid)) if _read_kt(coder, counts)}
    for key in keys:
        supporting = [nid for nid in range(1, count_nodes + 1) if key in state.nodes[nid].support]
        routes, pres, vals = {}, defaultdict(lambda: [0, 0]), {}
        pubs = sum(hist[key].values())
        bg_pres = [2 * (n - pubs) + 1, 2 * pubs + 1]
        bg_value = [2 * hist[key][v] + 1 for v in vocab[key]]
        for r in range(n):
            eligible = (0,) + tuple(nid for nid in supporting if r in state.nodes[nid].members)
            counts = routes.setdefault(eligible, [0] * len(eligible))
            owner = eligible[_read_kt(coder, counts)]
            if owner:
                state.owners[r][key] = owner
                published = _read_kt(coder, pres[owner])
            else:
                published = coder.symbol(bg_pres)
            if published:
                if owner == 0:
                    value = coder.symbol(bg_value)
                else:
                    value = _read_kt(coder, vals.setdefault(owner, [0] * len(vocab[key])))
                state.records[r][key] = vocab[key][value]
    state.validate()
    actual = {key: Counter(r[key] for r in state.records if key in r) for key in keys}
    if actual != hist:
        raise ValueError("decoded data does not match the transmitted histograms")
    return state
