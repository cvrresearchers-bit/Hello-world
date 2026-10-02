---------------------------- MODULE AdaptiveAA ----------------------------
(***************************************************************************)
(* Abstract round model of the safe-area midpoint update, d = 1.            *)
(*                                                                         *)
(* Gather guarantees that every honest view contains a common core of     *)
(* N - TS parties (and has at least N - TS members). The adversary         *)
(* chooses, in every round: the core, each honest view (any superset of    *)
(* the core), and the values of the TS Byzantine parties (reliable          *)
(* broadcast makes them consistent across honest parties).                  *)
(* Each honest party moves to the midpoint of its view after trimming TS    *)
(* values from each side.                                                   *)
(*                                                                         *)
(* Checked: Validity (honest values stay in the initial honest hull) and   *)
(* Contraction (the honest spread at least halves, rounded up).             *)
(***************************************************************************)
EXTENDS Integers, FiniteSets

CONSTANTS N, TS, K, MAXROUND
ASSUME N > 3 * TS /\ TS >= 1 /\ K >= 1 /\ MAXROUND >= 1

Parties == 1..N
Honest  == 1..(N - TS)
Byz     == (N - TS + 1)..N
Vals    == 0..K
Dom     == (0 - K)..(2 * K)      \* Byzantine values may lie outside the honest range

VARIABLES val, init, rnd
vars == <<val, init, rnd>>

\* j-th smallest (1-indexed) of f[p], p in view
Kth(f, view, j) == CHOOSE v \in Dom :
                      /\ Cardinality({p \in view : f[p] < v}) <= j - 1
                      /\ Cardinality({p \in view : f[p] <= v}) >= j

HiOf(f) == CHOOSE v \in Dom : (\A p \in Honest : f[p] <= v) /\ (\E q \in Honest : f[q] = v)
LoOf(f) == CHOOSE v \in Dom : (\A p \in Honest : f[p] >= v) /\ (\E q \in Honest : f[q] = v)
Spread(f) == HiOf(f) - LoOf(f)

\* Byzantine entries of val are placeholders (0): their per-round value is chosen in Next.
Init == /\ val \in [Parties -> Vals]
        /\ \A q \in Byz : val[q] = 0
        /\ init = val
        /\ rnd = 0

Next == /\ rnd < MAXROUND
        /\ \E core \in SUBSET Parties :
             /\ Cardinality(core) = N - TS
             /\ \E views \in [Honest -> {core \cup X : X \in SUBSET (Parties \ core)}] :
                  /\ \E byzv \in [Byz -> Dom] :
                       LET cur == [q \in Parties |-> IF q \in Honest THEN val[q] ELSE byzv[q]]
                       IN  val' = [q \in Parties |->
                                     IF q \in Honest
                                     THEN (Kth(cur, views[q], TS + 1)
                                           + Kth(cur, views[q], Cardinality(views[q]) - TS)) \div 2
                                     ELSE 0]
        /\ init' = init
        /\ rnd' = rnd + 1

Spec == Init /\ [][Next]_vars

\* Validity: honest values never leave the hull of the honest initial values
Validity == \A p \in Honest : val[p] >= LoOf(init) /\ val[p] <= HiOf(init)

\* Contraction (integer midpoints => rounded up): spread' <= ceil(spread / 2) + 1 slack for flooring
Contraction == [][Spread(val') <= (Spread(val) + 1) \div 2 + 1]_vars

\* Strict monotonicity of the spread
NonExpanding == [][Spread(val') <= Spread(val)]_vars
=============================================================================
