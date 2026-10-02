"""Protocol variants: the proposed method, baselines and ablations.

The constants ``c`` (extra iterations) and ``check_threshold`` are FROZEN here, i.e. never tuned
per experiment or per profile. They were set a priori from the nominal contraction factor
rho = 1/2 (c = 2 iterations of margin; halt only if the certified diameter is <= eps).
``scripts/pilot.py`` re-checks them on seeds disjoint from the reported ones.
"""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Params:
    name: str
    termination: str = "adaptive"   # adaptive | fixed | sync_rounds
    dhat: str = "safe_area"         # source of the iteration-count estimate: safe_area | raw
    substitute: bool = True         # halted parties keep contributing their final value
    gather: str = "full"            # full (3-step Gather) | plain (first n-ts values)
    check: bool = True              # halting check; on failure R grows (restart variant)
    c: int = 2                      # extra iterations on top of ceil(log2(Dhat/eps))
    check_threshold: float = 1.0    # halt only if certified diameter <= threshold * eps
    growth: float = 1.0             # on failed check: R += max(2, ceil(R * growth))
    exact: bool = True              # allow the exact LP safe area for d >= 2


PROPOSED = Params("proposed")

METHODS: dict[str, Params] = {
    "proposed": PROPOSED,
    # B1: synchronous-only lock-step rounds with fixed R(M); breaks in asynchrony
    "B1_sync_fixed": Params("B1_sync_fixed", termination="sync_rounds", check=False, gather="plain",
                            substitute=False),
    # B2: same event-driven engine with a fixed iteration count from the a-priori range M.
    # (Stand-in for the SPAA'23 fixed-iteration protocol; also plays the async-only role.)
    "B2_fixed": Params("B2_fixed", termination="fixed", check=False),
    # B4: classical adaptive estimate: raw range of the first-round view, no halting check
    "B4_classic_adaptive": Params("B4_classic_adaptive", dhat="raw", check=False),
    # ablations
    "AB1_raw_dhat": replace(PROPOSED, name="AB1_raw_dhat", dhat="raw"),
    "AB2_no_substitution": replace(PROPOSED, name="AB2_no_substitution", substitute=False),
    "AB3_plain_collect": replace(PROPOSED, name="AB3_plain_collect", gather="plain"),
}

# AB4 (randomised double-certificate design) is NOT implemented: the supplied design text
# does not specify it, and the document itself selects the deterministic track.
