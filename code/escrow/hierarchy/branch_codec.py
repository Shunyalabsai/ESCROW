"""Independent EHR05 bitstream implementation. Does not import the reference scorer."""
from __future__ import annotations

import hashlib
from collections import Counter, deque

from .codec import BitReader, BitWriter, ArithmeticEncoder, ArithmeticDecoder, _write_kt, _read_kt
from .model import HierarchyState, HierarchyNode

MAGIC = b"EHR05"


def _visits(state, order, balanced):
    if not balanced:
        return list(range(len(state.records)))
    bins = {}
    for row in range(len(state.records)):
        mask = 0
        for nid in order:
            mask = 2 * mask + int(row in state.nodes[nid].members)
        bins.setdefault(mask, deque()).append(row)
    queues, result = [bins[m] for m in sorted(bins)], []
    while queues:
        following = []
        for queue in queues:
            result.append(queue.popleft())
            if queue:
                following.append(queue)
        queues = following
    return result


def _families(state, order):
    waiting = {}
    for nid in order:
        waiting.setdefault(tuple(sorted(state.nodes[nid].parents)), []).append(nid)
    done = set()
    while waiting:
        choices = [key for key in waiting if all(p in done for p in key)]
        if not choices:
            raise ValueError("cyclic membership family")
        key = min(choices, key=lambda ps: min(order.index(v) for v in waiting[ps]))
        children = waiting.pop(key)
        rows = [r for r in range(len(state.records)) if all(r in state.nodes[p].members for p in key)]
        yield children, rows
        done.update(children)


def _write_uint(coder, n):
    # Gamma integers inside the arithmetic stream use fair binary actions.
    bits = bin(n + 1)[2:]
    for bit in "0" * (len(bits) - 1) + bits:
        coder.symbol(int(bit), [1, 1])


def _read_uint(coder):
    zeros = 0
    while coder.symbol([1, 1]) == 0:
        zeros += 1
        if zeros > 63:
            raise ValueError("membership dictionary is too large")
    value = 1
    for _ in range(zeros):
        value = 2 * value + coder.symbol([1, 1])
    return value - 1


def encode(state, joint_membership=True, member_predictors=True, balanced_records=True):
    state.validate()
    keys, order, n = state.keys, state.topological(), len(state.records)
    hist = {k: Counter(r[k] for r in state.records if k in r) for k in keys}
    vocab = {k: sorted(hist[k]) for k in keys}
    writer = BitWriter()
    writer.raw(MAGIC)
    writer.raw(bytes([int(joint_membership) + 2 * int(member_predictors) + 4 * int(balanced_records)]))
    writer.uint(n)
    writer.uint(len(keys))
    for key in keys:
        writer.string(key)
        writer.uint(len(vocab[key]))
        for value in vocab[key]:
            writer.string(value)
            writer.uint(hist[key][value])
    writer.uint(len(order))
    coder = ArithmeticEncoder(writer)
    parents_count = [0, 0]
    for i, nid in enumerate(order):
        for p in order[:i]:
            _write_kt(coder, int(p in state.nodes[nid].parents), parents_count)
        support_count = [0, 0]
        for key in keys:
            _write_kt(coder, int(key in state.nodes[nid].support), support_count)
    if joint_membership:
        for children, rows in _families(state, order):
            sequence = [tuple(int(r in state.nodes[c].members) for c in children) for r in rows]
            patterns = sorted(set(sequence))
            _write_uint(coder, len(patterns))
            for pattern in patterns:
                counts = [0, 0]
                for bit in pattern:
                    _write_kt(coder, bit, counts)
            index = {pattern: i for i, pattern in enumerate(patterns)}
            counts = [0] * len(patterns)
            for pattern in sequence:
                _write_kt(coder, index[pattern], counts)
    else:
        for nid in order:
            counts = [0, 0]
            for r in sorted(state.eligible_rows(nid)):
                _write_kt(coder, int(r in state.nodes[nid].members), counts)
    visits = _visits(state, order, balanced_records)
    for key in keys:
        supporting = [nid for nid in order if key in state.nodes[nid].support]
        pres = {nid: [0, 0] for nid in supporting}
        vals = {nid: [0] * len(vocab[key]) for nid in supporting}
        routes = {}
        idx = {v: i for i, v in enumerate(vocab[key])}
        pubs = sum(hist[key].values())
        bg_pres = [2 * (n - pubs) + 1, 2 * pubs + 1]
        bg_vals = [2 * hist[key][v] + 1 for v in vocab[key]]
        for r in visits:
            record = state.records[r]
            members = [nid for nid in supporting if r in state.nodes[nid].members]
            choices = (0,) + tuple(members)
            owner = state.owners[r].get(key, 0)
            _write_kt(coder, choices.index(owner), routes.setdefault(choices, [0] * len(choices)))
            pub = int(key in record)
            coder.symbol(pub, bg_pres if not owner else [2 * c + 1 for c in pres[owner]])
            if pub:
                value = idx[record[key]]
                coder.symbol(value, bg_vals if not owner else [2 * c + 1 for c in vals[owner]])
            trained = members if member_predictors else ([owner] if owner else [])
            for nid in trained:
                pres[nid][pub] += 1
                if pub:
                    vals[nid][value] += 1
    coder.finish()
    body = writer.bytes()
    framed = len(body).to_bytes(8, "big") + body
    return framed + hashlib.sha256(framed).digest()


def decode(data):
    if len(data) < 47 or hashlib.sha256(data[:-32]).digest() != data[-32:]:
        raise ValueError("description checksum mismatch")
    if int.from_bytes(data[:8], "big") != len(data) - 40:
        raise ValueError("description length mismatch")
    reader = BitReader(data[8:-32])
    if reader.raw(5) != MAGIC:
        raise ValueError("unknown branch-code format")
    flags = reader.raw(1)[0]
    if flags > 7:
        raise ValueError("unknown branch-code options")
    joint, members_train = bool(flags & 1), bool(flags & 2)
    balanced = bool(flags & 4)
    n, key_count = reader.uint(), reader.uint()
    keys, vocab, hist = [], {}, {}
    for _ in range(key_count):
        key = reader.string()
        keys.append(key)
        vocab[key], hist[key] = [], {}
        for _ in range(reader.uint()):
            value, count = reader.string(), reader.uint()
            vocab[key].append(value)
            hist[key][value] = count
    k = reader.uint()
    state = HierarchyState([{} for _ in range(n)], {}, [{} for _ in range(n)], k + 1)
    order = list(range(1, k + 1))
    coder = ArithmeticDecoder(reader)
    parent_counts = [0, 0]
    for nid in order:
        parents = {p for p in range(1, nid) if _read_kt(coder, parent_counts)}
        counts = [0, 0]
        support = {key for key in keys if _read_kt(coder, counts)}
        state.nodes[nid] = HierarchyNode(nid, set(), support, parents)
    if joint:
        for children, rows in _families(state, order):
            size = _read_uint(coder)
            if not 1 <= size <= len(rows):
                raise ValueError("invalid membership dictionary size")
            patterns = []
            for _ in range(size):
                counts = [0, 0]
                patterns.append(tuple(_read_kt(coder, counts) for _ in children))
            if patterns != sorted(set(patterns)):
                raise ValueError("membership patterns must be sorted and distinct")
            counts = [0] * size
            for r in rows:
                pattern = patterns[_read_kt(coder, counts)]
                for child, bit in zip(children, pattern):
                    if bit:
                        state.nodes[child].members.add(r)
            if any(c == 0 for c in counts):
                raise ValueError("unused pattern in observed membership dictionary")
    else:
        for nid in order:
            counts = [0, 0]
            state.nodes[nid].members = {r for r in sorted(state.eligible_rows(nid)) if _read_kt(coder, counts)}
    visits = _visits(state, order, balanced)
    for key in keys:
        supporting = [nid for nid in order if key in state.nodes[nid].support]
        pres = {nid: [0, 0] for nid in supporting}
        vals = {nid: [0] * len(vocab[key]) for nid in supporting}
        routes = {}
        pubs = sum(hist[key].values())
        bg_pres = [2 * (n - pubs) + 1, 2 * pubs + 1]
        bg_vals = [2 * hist[key][v] + 1 for v in vocab[key]]
        for r in visits:
            members = [nid for nid in supporting if r in state.nodes[nid].members]
            choices = (0,) + tuple(members)
            owner = choices[_read_kt(coder, routes.setdefault(choices, [0] * len(choices)))]
            if owner:
                state.owners[r][key] = owner
            pub = coder.symbol(bg_pres if not owner else [2 * c + 1 for c in pres[owner]])
            if pub:
                value = coder.symbol(bg_vals if not owner else [2 * c + 1 for c in vals[owner]])
                state.records[r][key] = vocab[key][value]
            trained = members if members_train else ([owner] if owner else [])
            for nid in trained:
                pres[nid][pub] += 1
                if pub:
                    vals[nid][value] += 1
    state.validate()
    actual = {key: Counter(r[key] for r in state.records if key in r) for key in keys}
    if actual != hist:
        raise ValueError("decoded data differs from the transmitted marginal counts")
    return state
