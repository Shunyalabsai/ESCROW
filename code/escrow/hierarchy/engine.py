"""Streaming reference search with explicit structural deferral and prefix repairs."""
from __future__ import annotations

import hashlib
import json
import math
import time

from .coding import Description, score
from .codec import encode
from .model import HierarchyState
from .moves import propose, revisions
from .predictive import record_loss
from .proposals import candidates


def signature(state):
    canonical = state.canonical()
    canonical.pop("records")
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def request_key(operation, arguments):
    return hashlib.sha256(json.dumps([operation, arguments], sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def equal_cost(a, b):
    # Floating point equality tolerance, not a statistical decision setting.
    return abs(a - b) <= 32 * max(math.ulp(a), math.ulp(b))


class HierarchicalEscrowGraph:
    description_type = Description
    encoder = staticmethod(encode)
    code_name = "EHR03"
    proposal_source = staticmethod(candidates)

    def describe(self, state):
        return self.description_type(state.records).score(state)

    def __init__(self, repair_every=100, candidate_budget=4096, allow_links=True,
                 fixed_ancestry=False):
        if repair_every < 1 or candidate_budget < 1:
            raise ValueError("repair cadence and search budget must be positive")
        self.state = HierarchyState()
        self.repair_every, self.candidate_budget = repair_every, candidate_budget
        self.allow_links, self.fixed_ancestry = allow_links, fixed_ancestry
        self.events, self.repair_log, self.escrow = [], [], {}
        self.predictive_loss_bits = 0.0
        self.arrival_losses = []

    def process(self, record):
        # Validate before touching the predictive history or accumulated loss.
        if not isinstance(record, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                               for k, v in record.items()):
            raise TypeError("categorical records must be dictionaries of strings to strings")
        loss = record_loss(self.state, record)
        self.arrival_losses.append(loss)
        self.predictive_loss_bits += loss
        self.state.append(record)
        start = len(self.events)
        # The reference implementation assigns new records at declared prefix repairs.
        # Between repairs they remain in the background; this latency is observable.
        if len(self.state.records) % self.repair_every == 0:
            self.repair()
        return dict(record=len(self.state.records) - 1, predictive_bits=loss,
                    events=self.events[start:], pending_until_repair=(
                        len(self.state.records) % self.repair_every != 0))

    def repair(self, final=False, proposals=None):
        """Greedy exact scoring. An injected proposal iterable supports search audits."""
        started, n = time.perf_counter(), len(self.state.records)
        if not n:
            return dict(prefix=0, evaluated=0, accepted=0, search_budget_exhausted=False)
        evaluated, accepted, tied, invalid = 0, 0, 0, 0
        exhausted = False
        serialisation_limited = False
        while evaluated < self.candidate_budget:
            objective = self.description_type(self.state.records)
            before = objective.score(self.state)
            incumbent_signature = signature(self.state)
            seen = {incumbent_signature}
            best_cost, winners = before["total"], []
            source = proposals if proposals is not None else self.proposal_source(
                self.state, self.allow_links, self.fixed_ancestry)
            pass_entries = []
            for op, args in source:
                if evaluated >= self.candidate_budget:
                    exhausted = True
                    break
                try:
                    after = propose(self.state, op, **args)
                except (ValueError, KeyError):
                    invalid += 1
                    continue
                sig = signature(after)
                if sig in seen:
                    continue
                seen.add(sig)
                evaluated += 1
                terms = objective.score(after)
                gain = before["total"] - terms["total"]
                key = request_key(op, args)
                previous = self.escrow.get(key, {})
                entry = dict(operation=op, arguments=args, first_seen_prefix=previous.get(
                    "first_seen_prefix", n), last_evaluated_prefix=n, current_gain_bits=gain,
                    delta_bits={k: terms[k] - before[k] for k in before},
                    status="no_coding_improvement", evaluated_against=incumbent_signature)
                # Evidence is replaced at each prefix. Overlapping gains are never summed.
                self.escrow[key] = entry
                pass_entries.append(key)
                if equal_cost(terms["total"], best_cost):
                    if terms["total"] < before["total"] and not equal_cost(terms["total"], before["total"]):
                        winners.append((op, args, after, terms, key))
                    else:
                        entry["status"] = "tied_with_unresolved"
                elif terms["total"] < best_cost:
                    best_cost = terms["total"]
                    winners = [(op, args, after, terms, key)]
                if gain > 0 and not equal_cost(terms["total"], before["total"]):
                    entry["status"] = "competing_arrangement"
            if not winners:
                break
            if len(winners) > 1:
                # No identifier-based resolution of an observational coding tie.
                for *_, key in winners:
                    self.escrow[key]["status"] = "tied_competing_arrangements"
                tied += len(winners)
                break
            op, args, after, terms, key = winners[0]
            independently_recomputed = self.describe(after)
            if not equal_cost(independently_recomputed["total"], terms["total"]):
                raise AssertionError("cached and independent descriptions disagree")
            # The real serialisation must improve too, not just its fractional ideal length.
            actual_before, actual_after = len(self.encoder(self.state)) * 8, len(self.encoder(after)) * 8
            if actual_after >= actual_before:
                self.escrow[key]["status"] = "no_serialised_improvement"
                serialisation_limited = True
                break
            changes = revisions(self.state, after)
            entry = self.escrow[key]
            entry["status"] = "accepted"
            event = dict(prefix=n, phase="final_repair" if final else "prefix_repair",
                         operation=op, arguments=args,
                         delta_bits={k: terms[k] - before[k] for k in before},
                         serialised_delta_bits=actual_after - actual_before,
                         proposal_first_seen_prefix=entry["first_seen_prefix"],
                         proposal_detection_delay=n - entry["first_seen_prefix"], **changes)
            revision_payload = json.dumps(dict(operation=op, arguments=args, **changes),
                                          sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            # A disclosed JSON transport measure, separate from the snapshot objective.
            event["revision_payload_bits"] = 64 + 8 * len(revision_payload)
            event["revised_records"] = len(set(r for change in changes["membership_revisions"]
                for r in change["added"] + change["removed"]) |
                {change["record"] for change in changes["ownership_revisions"]})
            self.events.append(event)
            self.state = after
            accepted += 1
            if proposals is not None:
                break
        if evaluated >= self.candidate_budget:
            exhausted = True
        report = dict(prefix=n, phase="final_repair" if final else "prefix_repair",
                      evaluated=evaluated, accepted=accepted, invalid_proposals=invalid,
                      tied_candidates=tied, search_budget_exhausted=exhausted,
                      elapsed_seconds=time.perf_counter() - started,
                      stopping_reason="budget" if exhausted else (
                          "coding_tie" if tied else "byte_rounding" if serialisation_limited
                          else "evaluated_neighbourhood_exhausted"))
        self.repair_log.append(report)
        return report

    def final_repair(self):
        return self.repair(final=True)

    def snapshot(self):
        state = self.state
        current = signature(state)
        unresolved = []
        for key, value in self.escrow.items():
            if value["status"] == "accepted":
                continue
            entry = dict(candidate_id=key, **value)
            entry["current_for_incumbent"] = (value["last_evaluated_prefix"] == len(state.records)
                                              and value["evaluated_against"] == current)
            if not entry["current_for_incumbent"]:
                entry["last_status"] = entry["status"]
                entry["status"] = "needs_re_evaluation"
            unresolved.append(entry)
        return dict(prefix=len(state.records), nodes=[dict(id=nid, support=sorted(v.support),
                    members=sorted(v.members), parents=sorted(v.parents))
                    for nid, v in sorted(state.nodes.items())],
                    parent_links=sorted((p, c) for c, v in state.nodes.items() for p in v.parents),
                    memberships=[[nid for nid, v in sorted(state.nodes.items()) if r in v.members]
                                 for r in range(len(state.records))],
                    ownership=[dict(row) for row in state.owners],
                    description_bits=self.describe(state), predictive_loss_bits=self.predictive_loss_bits,
                    unresolved=unresolved,
                    repair_log=list(self.repair_log), events=list(self.events),
                    protocol=dict(code=self.code_name, repair_every=self.repair_every, candidate_budget=self.candidate_budget,
                                  allow_links=self.allow_links, fixed_ancestry=self.fixed_ancestry,
                                  arrival_assignment="background_until_prefix_repair",
                                  final_repair_is_automatic=False))

    def flat_labels(self):
        """Supplementary deepest-membership projection. Background is explicitly -1."""
        order = self.state.topological()
        depth = {}
        for nid in order:
            depth[nid] = 1 + max((depth[p] for p in self.state.nodes[nid].parents), default=0)
        return [min((nid for nid in order if r in self.state.nodes[nid].members),
                    key=lambda nid: (-depth[nid], nid), default=-1)
                for r in range(len(self.state.records))]


class CorrectedFlatGraph(HierarchicalEscrowGraph):
    def __init__(self, **kwargs):
        super().__init__(allow_links=False, **kwargs)


class FixedAncestryGraph(HierarchicalEscrowGraph):
    def __init__(self, **kwargs):
        super().__init__(fixed_ancestry=True, **kwargs)
