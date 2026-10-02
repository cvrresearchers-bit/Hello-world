"""Deterministic discrete-event simulator with a sync / async network model.

Time unit: delta = 1.0 (the real maximum message delay in the synchronous network).

Network model
-------------
* ``sync``  : every point-to-point hop has delay in (0.1, 1.0]  (<= Delta).
* ``async`` : hop delays are drawn from a heavy-tailed distribution (exp / lognormal /
  pareto) or a group-split adversarial-style scheduler ('split') and an optional *slow set* of honest parties has all its hops multiplied by
  ``slow_factor``. Delays are finite, so delivery is eventual (the adversarial-scheduling
  model of asynchrony), but unbounded relative to Delta.
* Reliable broadcast (RB) is an *ideal functionality*: a broadcast reaches every party
  with the same payload (consistency); delay = three independent hops (INIT/ECHO/READY).
  Its message cost is charged analytically as n + 2n^2 messages per broadcast.
  Byzantine senders therefore cannot equivocate on RB-broadcast values; their power is the
  choice of value, withholding and delay (this is exactly what RB buys in the real protocol).
* Gather's SET/TSET messages are also sent over RB (totality prevents a Byzantine party from
  showing its set to only some honest parties). Point-to-point authenticated links are available
  (``send``/``bcast``) for baselines but unused by the honest protocol.

Every message carries a Lamport-style *causal depth*; the depth at which a party outputs
is its asynchronous round count.
"""
from __future__ import annotations

import heapq
import random
import time
from typing import Callable

HDR = 16  # bytes of header per message (sender, tag, signature stub)


class Engine:
    def __init__(self, seed: int, n: int, mode: str, delay: str = "lognormal",
                 slow: frozenset = frozenset(), slow_factor: float = 40.0):
        assert mode in ("sync", "async")
        self.rnd = random.Random(seed)
        self.n = n
        self.mode = mode
        self.delay = "uniform" if mode == "sync" else delay
        self.slow = slow
        self.slow_factor = slow_factor
        self.t = 0.0
        self._q: list = []
        self._seq = 0
        self.parties: dict = {}
        self.stop = False
        self.events = 0
        self.msgs = 0
        self.bytes = 0
        self.by_kind: dict[str, int] = {}

    # -- delay models ---------------------------------------------------------------
    def _hop(self, src: int, dst: int) -> float:
        r = self.rnd
        if self.delay == "uniform":
            d = 0.1 + 0.9 * r.random()
        elif self.delay == "exp":
            d = 0.05 + r.expovariate(1.0)
        elif self.delay == "lognormal":
            d = r.lognormvariate(-0.3, 1.0)
        elif self.delay == "pareto":
            d = 0.3 * r.paretovariate(1.5)
        elif self.delay == "split":
            # adversarial-style scheduler: two honest groups (by id parity); cross-group hops
            # are 15x slower, so the groups tend to assemble different views
            d = r.lognormvariate(-0.3, 0.5) * (15.0 if (src % 2) != (dst % 2) else 1.0)
        else:
            raise ValueError(f"unknown delay model {self.delay}")
        if src in self.slow or dst in self.slow:
            d *= self.slow_factor
        return d

    # -- scheduling ------------------------------------------------------------------
    def at(self, delay: float, fn: Callable, *args) -> None:
        self._seq += 1
        heapq.heappush(self._q, (self.t + delay, self._seq, fn, args))

    def _count(self, kind: str, nmsgs: int, nbytes: int) -> None:
        self.msgs += nmsgs
        self.bytes += nbytes
        self.by_kind[kind] = self.by_kind.get(kind, 0) + nmsgs

    # -- communication ---------------------------------------------------------------
    def send(self, src: int, dst: int, kind: str, tag, payload, depth: int, size: int) -> None:
        self._count(kind, 1, size + HDR)
        self.at(self._hop(src, dst), self._deliver_msg, dst, kind, tag, src, payload, depth)

    def bcast(self, src: int, kind: str, tag, payload, depth: int, size: int) -> None:
        for dst in range(self.n):
            self.send(src, dst, kind, tag, payload, depth, size)

    def rb(self, src: int, kind: str, tag, payload, depth: int, size: int, extra: float = 0.0) -> None:
        """Ideal reliable broadcast: INIT + ECHO + READY accounted as n + 2n^2 messages."""
        n = self.n
        self._count("RB_" + kind, n + 2 * n * n, (n + 2 * n * n) * (size + HDR))
        for dst in range(n):
            d = extra + self._hop(src, dst) + self._hop(src, dst) + self._hop(src, dst)
            self.at(d, self._deliver_rb, dst, kind, tag, src, payload, depth + 3)

    def _deliver_msg(self, dst, kind, tag, src, payload, depth):
        p = self.parties.get(dst)
        if p is not None:
            p.on_msg(kind, tag, src, payload, depth + 1)

    def _deliver_rb(self, dst, kind, tag, src, payload, depth):
        p = self.parties.get(dst)
        if p is not None:
            p.on_rb(kind, tag, src, payload, depth)

    # -- main loop -------------------------------------------------------------------
    def run(self, max_events: int = 4_000_000, max_wall: float = 120.0, max_time: float = 1e9) -> str:
        """Run until ``self.stop`` is set. Returns the stop reason."""
        t0 = time.monotonic()
        q = self._q
        while q:
            if self.stop:
                return "done"
            t, _, fn, args = heapq.heappop(q)
            if t > max_time:
                return "max_time"
            self.t = t
            self.events += 1
            fn(*args)
            if self.events >= max_events:
                return "max_events"
            if (self.events & 0x3FFF) == 0 and time.monotonic() - t0 > max_wall:
                return "max_wall"
        return "done" if self.stop else "quiescent"
