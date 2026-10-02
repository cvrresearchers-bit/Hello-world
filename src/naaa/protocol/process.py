"""Honest parties.

HonestParty implements the proposed protocol (and its ablations / fixed baselines):

  instance 0  : Gather(inputs)      -> certified diameter D-hat (safe area), R = f(D-hat)
  instance k  : Gather(v_k)         -> v_{k+1} = centre(SafeArea(view_k))      (k < R)
  at k+1 == R : halting check       -> HALT (output) or grow R (restart variant)
  after HALT  : the party keeps *contributing its final value* to later Gather instances
                (halted-value substitution) so slower parties are never starved.

Gather (3 steps, over ideal reliable broadcast) gives every honest party a view of
>= n - ts values and all honest views share a common core of >= n - ts values
(needs n > 3*ts; proof in docs/THEORY.md).

SyncRoundParty is baseline B1: lock-step rounds with a fixed timer; it is only correct
under synchrony.
"""
from __future__ import annotations

from math import ceil, log2

import numpy as np

from .. import geometry as geo
from .params import Params

SZ_ID = 4


class GatherState:
    __slots__ = ("vals", "started", "sent_set", "sent_tset", "done", "set_recv",
                 "set_missing", "set_wait", "ok1", "tset_missing", "tset_wait", "ok2")

    def __init__(self):
        self.vals: dict[int, np.ndarray] = {}
        self.started = False
        self.sent_set = False
        self.sent_tset = False
        self.done = False
        self.set_recv: dict[int, frozenset] = {}
        self.set_missing: dict[int, set] = {}
        self.set_wait: dict[int, list] = {}
        self.ok1: set = set()
        self.tset_missing: dict[int, set] = {}
        self.tset_wait: dict[int, list] = {}
        self.ok2: set = set()


class Ctx:
    """Shared run context."""

    def __init__(self, engine, n, ts, ta, d, eps, M, params: Params, honest_ids, honest_inputs, max_iters=400):
        self.engine = engine
        self.n, self.ts, self.ta, self.d = n, ts, ta, d
        self.eps, self.M = eps, M
        self.params = params
        self.honest_ids = list(honest_ids)
        self.honest_inputs = honest_inputs
        self.max_iters = max_iters
        self.outputs_done = 0
        self.hist: dict[int, dict[int, np.ndarray]] = {}   # iteration -> {party: value} (for the oracle metric)

    def on_output(self):
        self.outputs_done += 1
        if self.outputs_done >= len(self.honest_ids):
            self.engine.stop = True


class HonestParty:
    def __init__(self, pid: int, ctx: Ctx, x0: np.ndarray):
        self.id, self.ctx = pid, ctx
        self.eng = ctx.engine
        self.p = ctx.params
        self.v = np.asarray(x0, dtype=float)
        self.vsize = 8 * ctx.d
        self.depth = 0
        self.gs: dict[int, GatherState] = {}
        self.R: int | None = None
        self.dhat0 = float("nan")
        self.raw0 = float("nan")
        self.iters = 0
        self.restarts = 0
        self.capped = False
        self.halted = False
        self.silent = False
        self.output: np.ndarray | None = None
        self.out_time = float("nan")
        self.out_depth = -1
        self.halts_seen: set = set()
        self.quorum_time = float("nan")

    # -- helpers -----------------------------------------------------------------------
    def _g(self, k: int) -> GatherState:
        g = self.gs.get(k)
        if g is None:
            g = self.gs[k] = GatherState()
        return g

    def start(self):
        self._start_instance(0)

    def _start_instance(self, k: int):
        g = self._g(k)
        g.started = True
        self.eng.rb(self.id, "VAL", k, self.v, self.depth, self.vsize)
        self._progress(k, g)

    # -- message handlers ----------------------------------------------------------------
    def on_rb(self, kind, tag, src, payload, depth):
        if self.silent:
            return
        if depth > self.depth:
            self.depth = depth
        if kind == "VAL":
            g = self._g(tag)
            if src in g.vals:
                return
            g.vals[src] = payload
            if self.halted and not g.started:        # ghost: join with the final value
                g.started = True
                self.eng.rb(self.id, "VAL", tag, self.output, self.depth, self.vsize)
            for j in g.set_wait.pop(src, ()):
                miss = g.set_missing[j]
                miss.discard(src)
                if not miss:
                    self._set_verified(g, j)
            self._progress(tag, g)
        elif kind == "HALT":
            self.halts_seen.add(src)
            self._check_quorum()
        elif kind in ("SET", "TSET"):
            self._on_set_msg(kind, tag, src, payload)

    def on_msg(self, *args):
        """Honest parties use no point-to-point messages: Gather sets travel over reliable
        broadcast (totality is what keeps a Byzantine party from stalling some honest parties)."""

    def _on_set_msg(self, kind, tag, src, payload):
        g = self._g(tag)
        if kind == "SET":
            if src in g.set_recv:
                return
            g.set_recv[src] = payload
            miss = {i for i in payload if i not in g.vals}
            if miss:
                g.set_missing[src] = miss
                for i in miss:
                    g.set_wait.setdefault(i, []).append(src)
            else:
                self._set_verified(g, src)
        elif kind == "TSET":
            if src in g.tset_missing or src in g.ok2:
                return
            miss = {i for i in payload if i not in g.ok1}
            if miss:
                g.tset_missing[src] = miss
                for i in miss:
                    g.tset_wait.setdefault(i, []).append(src)
            else:
                g.ok2.add(src)
        self._progress(tag, g)

    def _set_verified(self, g: GatherState, j: int):
        g.ok1.add(j)
        for q in g.tset_wait.pop(j, ()):
            miss = g.tset_missing[q]
            miss.discard(j)
            if not miss:
                g.ok2.add(q)

    # -- Gather ------------------------------------------------------------------------------
    def _progress(self, k: int, g: GatherState):
        n, ts = self.ctx.n, self.ctx.ts
        if g.done:
            return
        if self.p.gather == "plain":
            if g.started and len(g.vals) >= n - ts:
                ids = list(g.vals)[: n - ts]
                g.done = True
                self._gather_done(k, {i: g.vals[i] for i in ids})
            return
        if g.started and not g.sent_set and len(g.vals) >= n - ts:
            g.sent_set = True
            ids = frozenset(g.vals)
            self.eng.rb(self.id, "SET", k, ids, self.depth, SZ_ID * len(ids))
        if g.sent_set and not g.sent_tset and len(g.ok1) >= n - ts:
            g.sent_tset = True
            T = frozenset(g.ok1)
            self.eng.rb(self.id, "TSET", k, T, self.depth, SZ_ID * len(T))
        if g.sent_tset and len(g.ok2) >= n - ts:
            g.done = True
            U: set = set()
            for c in g.ok1:
                U |= g.set_recv[c]
            self._gather_done(k, {i: g.vals[i] for i in U})

    # -- contraction / adaptivity / halting ------------------------------------------------------
    def _gather_done(self, k: int, view: dict):
        if self.halted:
            return
        ctx, p = self.ctx, self.p
        ids = sorted(view)
        X = np.stack([view[i] for i in ids])
        res = geo.analyze(X, ctx.ts, p.exact)
        if k == 0:
            self.dhat0 = res.diam
            self.raw0 = geo.raw_diameter(X)
            if p.termination == "adaptive":
                est = res.diam if p.dhat == "safe_area" else self.raw0
                self.R = (ceil(log2(est / ctx.eps)) if est > ctx.eps else 0) + p.c
            else:
                self.R = ceil(log2(ctx.M / ctx.eps)) + p.c
            self.R = max(1, self.R)
        self.v = res.center
        self.iters = k + 1
        ctx.hist.setdefault(self.iters, {})[self.id] = self.v
        if self.iters >= self.R:
            if (not p.check) or res.diam <= p.check_threshold * ctx.eps:
                self._halt()
                return
            self.restarts += 1
            self.R += max(2, ceil(self.R * p.growth))
        if self.iters >= ctx.max_iters:
            self.capped = True
            self._halt()
            return
        self._start_instance(k + 1)

    def _halt(self):
        self.halted = True
        self.output = self.v
        self.out_time = self.eng.t
        self.out_depth = self.depth
        self.eng.rb(self.id, "HALT", self.iters, self.output, self.depth, self.vsize)
        self.halts_seen.add(self.id)
        if not self.p.substitute:
            self.silent = True
        self._check_quorum()
        self.ctx.on_output()

    def _check_quorum(self):
        if self.halted and self.quorum_time != self.quorum_time and len(self.halts_seen) >= self.ctx.n - self.ctx.ts:
            self.quorum_time = self.eng.t


class SyncRoundParty:
    """Baseline B1: lock-step rounds of length T = 3*Delta + margin, fixed R from M.

    Correct only in the synchronous network (all RB values arrive within T). Under
    asynchrony late values are dropped, so views shrink and agreement can fail.
    """
    ROUND_T = 3.05

    def __init__(self, pid: int, ctx: Ctx, x0: np.ndarray):
        self.id, self.ctx, self.eng, self.p = pid, ctx, ctx.engine, ctx.params
        self.v = np.asarray(x0, dtype=float)
        self.vsize = 8 * ctx.d
        self.depth = 0
        self.vals: dict[int, dict[int, np.ndarray]] = {}
        self.R = ceil(log2(ctx.M / ctx.eps)) + self.p.c
        self.iters = 0
        self.restarts = 0
        self.capped = False
        self.halted = False
        self.output = None
        self.out_time = float("nan")
        self.out_depth = -1
        self.dhat0 = float("nan")
        self.raw0 = float("nan")
        self.quorum_time = float("nan")

    def start(self):
        self._round(0)

    def _round(self, k: int):
        self.eng.rb(self.id, "VAL", k, self.v, self.depth, self.vsize)
        self.eng.at(self.ROUND_T, self._end_round, k)

    def on_rb(self, kind, tag, src, payload, depth):
        if kind == "VAL":
            self.vals.setdefault(tag, {}).setdefault(src, payload)
            self.depth = max(self.depth, depth)

    def on_msg(self, *a):
        pass

    def _end_round(self, k: int):
        ctx = self.ctx
        view = self.vals.get(k, {})
        m = len(view)
        if m > 2 * ctx.ts:
            X = np.stack([view[i] for i in sorted(view)])
            res = geo.analyze(X, ctx.ts, self.p.exact)
            if k == 0:
                self.dhat0 = res.diam
                self.raw0 = geo.raw_diameter(X)
            self.v = res.center
        self.iters = k + 1
        if self.iters >= self.R:
            self.halted = True
            self.output = self.v
            self.out_time = self.eng.t
            self.out_depth = self.depth
            ctx.on_output()
            return
        self._round(k + 1)
