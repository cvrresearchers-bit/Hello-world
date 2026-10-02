# Theory notes: what is proven, what is not

This file states the definitions and theorems behind the implementation and is explicit about
the status of each claim. **Proven** = a complete argument is given here (and is cross-checked by
the simulator / model checker). **Conjecture** = believed, supported only by experiments, *not* proven.
The design document asked for "all theorems, proofs, and variants"; the parts below marked
*Conjecture* or *Not attempted* are the honest remainder of that request.

## 0. Model

* `n` parties, a static Byzantine set of size `f`, authenticated channels, rushing adversary.
* Synchronous network: delays <= Delta, `f <= ts`. Asynchronous network: eventual delivery, `f <= ta <= ts`.
  The protocol is event-driven and uses **no timeouts**, so it never needs to know which network it is in.
* Reliable broadcast (RB): validity, consistency, totality. Treated as an ideal functionality.
* Inputs `x_i in R^d`. `D_H` = L-infinity diameter of the honest inputs.
* **eps-agreement**: honest outputs pairwise within `eps` (L-inf). **Validity**: outputs in the convex hull of honest inputs
  (box validity for the coordinate-trim fallback). **Termination**: every honest party outputs.

**Definition (instance-adaptive termination).** A protocol is *(alpha, beta)-adaptive* if in every execution
every honest party terminates after at most `alpha*log2(D_H/eps) + beta` iterations, where `D_H` is the honest
diameter *of that execution* and the protocol is not told `D_H` (nor any range bound `M`).

## 1. Gather (proven)

Protocol (instance `k`, resilience `n > 3*ts`, all messages over RB):
1. broadcast `VAL(v)`; on delivering `n-ts` values broadcast `SET(S1)` = ids delivered so far;
2. a `SET(S1_j)` is *verified* once every id in `S1_j` has been delivered; when `n-ts` sets are verified,
   broadcast `TSET(T)`, `T` = ids whose sets are verified;
3. a `TSET(T_j)` is verified once `T_j` is a subset of my verified-set ids; when `n-ts` TSETs are verified,
   output `U = union of S1_c over my verified c` and the view `{(id, value) : id in U}`.

**Theorem 1.** If at most `ts` parties are Byzantine and `n > 3*ts`, every honest party outputs a view with
`|U_i| >= n - ts`, and there is a set `C` with `|C| >= n - ts` contained in every honest view
(the *common core*).

*Proof (liveness).* At least `n-ts` honest parties broadcast, so every honest party delivers `n-ts` values and sends a SET.
By RB totality every delivered value / SET / TSET is eventually delivered everywhere, so every SET of an honest party
becomes verified at every honest party (its ids are delivered by totality). Hence each honest party verifies `>= n-ts`
SETs, sends a TSET with a verifiable `T` (each `c in T` had its SET verified at the sender, hence delivered by totality
to everybody and verifiable there), and finally verifies `>= n-ts` TSETs.

*Proof (common core).* Let `T_j` be the set in honest `j`'s TSET (`|T_j| >= n-ts`). Counting incidences over the
`>= n-ts` honest `j`: some `c` lies in at least `ts+1` honest `T_j` (average argument; `(n-ts)^2/n > ts` for `n > 3 ts`).
Take any honest `i`. It verified `>= n-ts` TSETs; `n-ts + (ts+1) > n`, so one of them is from a `j` with `c in T_j`.
Verification means `T_j` is a subset of `i`'s verified ids, so `c` is verified at `i` and `S1_c` is contained in `U_i`.
RB consistency makes `S1_c` (|S1_c| >= n-ts) the same object at every honest party. Put `C = S1_c`. qed.

## 2. Safe area, certified diameter, validity (proven)

`S(V) = intersection over all (m-ts)-subsets Q of V of conv(Q)` for a view `V` of `m` points.

**Lemma 2.** (a) If `m > (d+1)*ts`, `S(V)` is non-empty (Tverberg: partition `V` into `ts+1` parts with a common point of
the hulls; deleting any `ts` points leaves one part intact, so the point is in every `conv(Q)`).
(b) If at most `ts` points of `V` are Byzantine, `S(V) subseteq conv(honest points of V) subseteq conv(honest inputs-of-the-round)`.
(c) **Certified diameter:** `Dhat(V) := diam_inf S(V) <= D_H` at round 0, because `S(V)` lies in the honest hull.
Byzantine parties therefore cannot inflate `Dhat`. (The *raw* range of `V` can be inflated to `M`; this is ablation AB1.)
(d) If `V` contains a core `C`, then `S(V) supseteq S(C)` (every `(m-ts)`-subset `Q` of `V` contains at least `|C|-ts`
points of `C`, so `conv(Q) supseteq` the hull of a `(|C|-ts)`-subset of `C`). Hence all honest safe areas share the non-empty set `S(C)`
when `|C| > (d+1)*ts`.

With `|V| >= n-ts` and `n > (d+2)*ts` we get `m > (d+1)*ts`. This is why the implementation requires `n > (d+2)*ts`
(see the README for how this differs from the optimal network-agnostic bound).

## 3. Contraction (proven for d = 1, open for d >= 2)

**Theorem 3 (d = 1).** Let all honest views contain a common core `C`, `|C| >= n-ts`, `n > 3 ts`, at most `ts` Byzantine values.
Let honest current values lie in `[a, b]` and `v_i' = (lo_i + hi_i)/2` with `lo_i, hi_i` the `(ts+1)`-th smallest and
`(m_i - ts)`-th smallest value of `V_i`. Then all `v_i'` lie in `[a, b]` (validity) and `max_i v_i' - min_i v_i' <= (b-a)/2`.

*Proof.* Trimming `ts` values per side removes every Byzantine value outside `[a,b]`, so `[lo_i, hi_i] subseteq [a,b]`.
Let `y_(1) <= ... <= y_(|C|)` be the sorted core values and write `V_i = C + E_i` with `e = |E_i| <= ts` extra elements.
Adding elements can only decrease the `k`-th smallest value, so `lo_i = x_(ts+1) <= y_(ts+1)`; removing `e` elements lowers
the rank of any value by at most `e`, so `hi_i = x_(|C|+e-ts) >= y_(|C|-ts)`. Since `|C| >= 2 ts + 1`,
`y_(ts+1) <= y_(|C|-ts)`, hence every interval `[lo_i, hi_i]` contains the common point `z = y_(ts+1)`.
For two intervals containing `z` and lying in `[a,b]`: `mid <= (z+b)/2` and `mid' >= (a+z)/2`, so
`|mid - mid'| <= (b-a)/2`. qed

*Cross-check:* `modelcheck.py` verifies Theorem 3's statement exhaustively for `(n,ts) = (4,1), (5,1)` and by sampling for
larger parameters (worst observed ratio exactly 1/2), and `tla/AdaptiveAA.tla` checks it with TLC.

**d >= 2 (Conjecture).** The implementation uses as the new value the mean of the `2d` coordinate-extreme points of
`S(V)` (a point of `S(V)`, hence hull-valid). Lemma 2(d) gives a common point of all safe areas, but we have *not* proven
a contraction factor for this choice of centre. Empirically the contraction is much faster than 1/2.

## 4. Adaptive iteration count (Conjecture)

The protocol sets `R = ceil(log2(Dhat_0/eps)) + c` from the round-0 certified diameter and runs a halting check
(`Dhat_{R-1} <= eps`, otherwise `R` grows: the "restart" variant of the design document).

**Conjecture 4 (adaptive R is sufficient).** In every execution all honest parties output within `eps`, with
at most `R_max = O(log(D_H/eps))` iterations.

*What is known.* (i) `Dhat_0 <= D_H` (Lemma 2c), so no Byzantine inflation. (ii) For `d = 1` and *all honest parties running
the same number of contraction steps*, Theorem 3 gives `spread <= D_H / 2^R`. (iii) The delicate case is that `Dhat_0`
(an under-estimate of the honest spread, e.g. a hidden honest outlier) may be much smaller than the spread of the
values that matter, and that parties may stop at *different* iterations (a stopped party keeps its final value as a
constant input). We did not prove that the remaining parties still end within `eps` of the frozen values.
(iv) Empirically: no violation in any simulated run (see `report.md`, Table 3, with the rule-of-three bound),
including adversarial schedulers (`split`), hidden outliers (G3 + A4), and a forced too-small `R`
(`tests/test_protocol.py::test_halting_check_restart_path_is_correct`). The check was exercised but, in the schedules we tried,
never observed to be *necessary*. That is also why this is a conjecture and not a theorem.

*Why the hidden-outlier objection to adaptivity does not apply.* A honest outlier that is slow or excluded in round 0 is
trimmed away (it is one of at most `ts` extreme values) in every later safe area; its own next value is the centre of
a safe area dominated by the cluster, i.e. it *adopts* a value already inside the honest hull. Experiment G3+A4 measures this.

## 5. Lower bound (Not attempted)

The design document proposes adapting a Fekete-style synchronous lower bound
`Omega(log(D_H/eps) / log log(D_H/eps))` to the honest diameter. The literature check in the design document
indicates the real-valued bound has this `log log` form, so `O(log(D_H/eps))` is *not* claimed to be optimal. We did not
reproduce or adapt that proof. `report.md` plots the ratio `R / (log x / log log x)` only as a visual reference, which is
**not** evidence for a lower bound.

## 6. Gap to the literature (open)

* Optimal network-agnostic resilience `(d+1)*ts + ta < n` is not implemented.
* Whether a published adaptive network-agnostic multidimensional protocol already exists was *not* verified here.
  The design document marks this literature check (its "Phase 2") as mandatory before any novelty claim.
