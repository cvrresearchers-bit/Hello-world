"""Byzantine adversary strategies (static corruption, rushing, value-level attacks).

Because RB-broadcast values are consistent (reliable broadcast), a Byzantine party's power
in this simulator is: the *value* it injects per Gather instance (chosen after seeing honest
values of that instance -- rushing), whether it participates at all (crash / withhold),
the delay it adds, and forged HALT announcements.

A1 crash        : silent.
A2 outlier      : extreme values at +/- M.
A3 equivocation : tries to send two different values; RB resolves it to one value, which the
                  adversary picks just outside the honest range (alternating low/high) to
                  maximise the spread seen by the safe-area step. Attempts are counted.
A4 withhold     : extreme value released late (25*Delta); the scheduler additionally slows an
                  honest outlier party (set in the run builder).
A5 halt forgery : forges an early HALT announcement carrying an extreme value, otherwise silent.
"""
from __future__ import annotations

import numpy as np

SLOW_RELEASE = 25.0


class ByzParty:
    def __init__(self, pid: int, ctx, strategy: str, index: int):
        self.id, self.ctx, self.eng = pid, ctx, ctx.engine
        self.strategy, self.index = strategy, index
        self.seen: dict[int, dict[int, np.ndarray]] = {}
        self.sent: set = set()
        self.equivocation_attempts = 0
        self.vsize = 8 * ctx.d

    # honest range known to the (omniscient, rushing) adversary for instance k
    def _honest_range(self, k: int):
        pts = list(self.seen.get(k, {}).values())
        if not pts:
            pts = list(self.ctx.honest_inputs)
        P = np.stack(pts)
        return P.min(axis=0), P.max(axis=0)

    def _extreme(self) -> np.ndarray:
        sign = 1.0 if self.index % 2 == 0 else -1.0
        return np.full(self.ctx.d, sign * self.ctx.M)

    def _value(self, k: int):
        s = self.strategy
        if s == "A1":
            return None
        if s == "A2":
            return self._extreme(), 0.0
        if s == "A3":
            lo, hi = self._honest_range(k)
            width = np.maximum(hi - lo, self.ctx.eps)
            self.equivocation_attempts += 1
            return (lo - 0.5 * width, 0.0) if (self.index + k) % 2 == 0 else (hi + 0.5 * width, 0.0)
        if s == "A4":
            return self._extreme(), SLOW_RELEASE
        if s == "A5":
            return None
        raise ValueError(f"unknown adversary {s}")

    def start(self):
        if self.strategy == "A5":
            self.eng.rb(self.id, "HALT", 1, self._extreme(), 0, self.vsize)
        self._act(0)

    def _act(self, k: int):
        if k in self.sent:
            return
        self.sent.add(k)
        out = self._value(k)
        if out is not None:
            val, extra = out
            self.eng.rb(self.id, "VAL", k, val, 0, self.vsize, extra=extra)

    def on_rb(self, kind, tag, src, payload, depth):
        if kind == "VAL":
            self.seen.setdefault(tag, {})[src] = payload
            self._act(tag)

    def on_msg(self, *a):
        pass


ADVERSARIES = ("A1", "A2", "A3", "A4", "A5")
