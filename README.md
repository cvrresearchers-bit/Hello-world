# Instance-adaptive, network-agnostic multidimensional approximate agreement

A runnable research prototype of the design in the project brief: a **deterministic** approximate-agreement
protocol whose iteration count adapts to the *actual* honest diameter `D_H` (not an a-priori range bound `M`),
works without knowing whether the network is synchronous or asynchronous, plus a message-level simulator,
adversaries, baselines, ablations, a bounded model checker, a TLA+ model, statistics and a self-contained report.

> **Read this first (honest status).** Everything runs end to end and produces results, but this is a *prototype*:
> several things the design document asked for are only partly delivered, and I list them rather than hide them.
> See [Status and limitations](#status-and-limitations) and [`docs/THEORY.md`](docs/THEORY.md) (what is proven vs conjectured).

## Quick start

```bash
# 0. check tool versions (Ubuntu)
python3 --version      # 3.10+ (developed on 3.11 / 3.12)
git --version
docker --version       # optional

# 1. get the code and run everything
git clone <repo-link>
cd <project-folder>
python3 -m pip install -r requirements.txt     # numpy scipy pandas matplotlib pytest hypothesis
python3 Main.py
```

`python3 Main.py` runs the whole workflow in one go (model check -> E1..E4, E6 simulations -> statistics,
tables, figures, report) and writes it to a **new folder per invocation**: `output/test1`, `output/test2`, ... `output/testN`.
Open `output/testN/report.md` first. The default `quick` profile takes about half a minute on 4 cores.

```bash
NAA_PROFILE=smoke python3 Main.py   # seconds, sanity check
NAA_PROFILE=full  python3 Main.py   # 30 seeds/cell, larger n; several minutes
python3 -m pytest -q tests          # 45+ tests (incl. property-based safety tests)
NAA_RUN_TESTS=true python3 Main.py  # run pytest as the first stage (log: output/testN/pytest.log)
docker compose up --build           # same thing in Docker; outputs land in ./output
```

Configuration lives in `.env` (copy `.env.example`): profile, seed, workers, output dir, log level,
optional `NAA_TLC_JAR` (also run TLC on `tla/AdaptiveAA.tla`), optional path to the real Intel Lab data.

## What one run produces (`output/testN/`)

| Path | Content |
|---|---|
| `report.md` | hypotheses H1-H4 evaluated on this run's data, all tables, figures, deviations |
| `figures/fig1..fig4*.png` | adaptivity scaling, network modes, certified-vs-raw diameter, sensor epochs |
| `tables/*.csv/.md` | T1 related work, E1 slopes, T2 by network, T3 safety, T4 communication, T5 model check, T6 ablations |
| `raw/E*.csv` | one row per simulated run (all metrics) |
| `modelcheck.json`, `tlc.log` | E5 outputs |
| `run.log`, `config.json`, `summary.json` | structured log, effective config + git hash, machine-readable summary |

## Project layout

```
Main.py                      single entry point (orchestrates everything)
src/naaa/
  config.py                  .env/env parsing, profiles, output/testN allocation
  geometry.py                safe area (exact LP / trimmed box), certified diameter, hull distance
  sim/engine.py              discrete-event simulator: sync / async (4 schedulers), ideal reliable broadcast, depth tracking
  protocol/process.py        honest party: Gather -> certified diameter -> R(D-hat) -> halting check -> halted-value substitution
  protocol/params.py         proposed method, baselines B1/B2/B4, ablations AB1-AB3 (frozen constants)
  protocol/run.py            builds one instance, runs it, extracts metrics (incl. oracle iterations)
  adversary/strategies.py    A1 crash, A2 outliers, A3 equivocation attempts, A4 withhold + slow honest outlier, A5 HALT forgery
  inputs/                    G1 uniform, G2 Gaussian, G3 cluster + honest outlier, G4 sensor data (real Intel Lab or surrogate)
  modelcheck.py              E5: exhaustive / sampled check of the abstract agreement step
  experiments.py             E1-E4, E6 definitions + parallel runner (graceful shutdown)
  analysis/stats.py          bootstrap CIs, paired Wilcoxon + Holm, Cliff's delta, OLS slope, rule of three
  analysis/report.py         tables, figures, report.md
tla/AdaptiveAA.tla (+ .cfg)  TLA+ model of the abstract round (checked with TLC)
tests/                       pytest + hypothesis
docs/THEORY.md               definitions, theorems, proofs, and which claims are NOT proven
scripts/pilot.py             re-checks the frozen constants on disjoint seeds
```

## The protocol in one paragraph

Per Gather instance every party reliably broadcasts its value, then a set of ids, then a set of verified sets; the output
view has `>= n-ts` values and **all honest views share a common core of `>= n-ts` values** (proved in `THEORY.md`).
Instance 0 yields the **certified diameter** `D-hat` = L-inf diameter of the safe area (the intersection of the hulls of all
`(m-ts)`-subsets); because the safe area lies in the honest hull, `D-hat <= D_H` and Byzantine parties cannot inflate it
(the raw view range can be inflated to `M`; that is ablation AB1). Each party runs `R = ceil(log2(D-hat/eps)) + 2`
contraction iterations (new value = centre of the safe area of the Gather view), then a **halting check** (`D-hat_k <= eps`,
else `R` grows: the "restart" fallback of the design document), then outputs, announces HALT, and keeps contributing its
final value to later instances (**halted-value substitution**) so slower parties are never starved. No timeouts anywhere,
so the same code runs in both network models.

## Results from the default run (seed 12345, `quick`, 4 cores; yours will match for the same seed/profile)

Numbers come from `output/testN/report.md`; run it to see them with confidence intervals.

* **Adaptivity (H1/H2):** the proposed protocol's iteration count grows with `log2(D_H/eps)` (OLS slope about 1.01, 95% CI [1.00, 1.02]),
  while the fixed-iteration baselines stay at 26 regardless of `D_H` (`M/eps = 2^24`). The async/sync slope ratio is about 0.98.
* **Safety (H3):** 0 liveness/agreement/validity failures in all 528 proposed-protocol runs (95% upper bound on the failure rate 0.57%,
  rule of three); the abstract step is checked exhaustively/sampled (Python) and by **TLC for N=4 and N=5** (no error; the N=5 run explores
  about 9.9M transitions / 18k distinct states). The synchronous-only baseline B1 violates eps-agreement in **all** runs under the
  adversarial split scheduler (spread up to ~1000 eps), which is the expected failure.
* **Certified diameter (H4):** on uniform inputs `D-hat/D_H` has median ~0.6 and min ~0.16. On "tight cluster + one honest outlier" inputs it is ~1e-4,
  *by design* (the outlier is one of the `ts` extreme values that the safe area discards). That outlier still ends up agreeing (G3 + A4 tests).
  Under outlier adversaries, the raw-range estimate (AB1/B4) costs ~26 iterations versus 12 for the proposed estimate.
* **Communication (E4):** the proposed protocol sends about 0.47x the messages of the fixed-iteration engine at the same `n` (it runs fewer iterations),
  i.e. the *per-iteration* cost is identical; there is no extra overhead.

### What these results do **not** show (please read)

* **The worst-case round count is not the actual need.** The simulator also records an omniscient *oracle* metric: the first iteration at which honest
  values are within `eps`. It is **1 in every run** of E1-E3 (the real contraction is far faster than the worst-case 1/2 that sets `R`).
  So "R adapts to `D_H`" is true for the protocol's *guaranteed* iteration count, and the report plots both. The schedulers I implemented do not
  find slow-contracting executions; a stronger adversarial scheduler search is future work.
* **The halting check was exercised but never observed to be necessary** (`scripts/pilot.py`: with `c` in 0..3 and the check on or off, 0 agreement failures and 0 restarts in 360 runs per setting).
  AB2 (no substitution) and AB3 (plain collect instead of Gather) show **no measurable effect** in these runs; they are included as ablations and
  reported as null results.
* **No theorem proves that adaptive `R` is always sufficient** (Conjecture 4 in `THEORY.md`). Zero violations in simulation is evidence, not proof.

## Status and limitations

Delivered: simulator, protocol with all components of the design (Gather, certified diameter, R(D-hat), halting + substitution, restart variant),
adversaries A1-A5, generators G1-G4, baselines B1/B2/B4, ablations AB1-AB3, experiments E1-E6, the statistics plan (bootstrap CIs, paired Wilcoxon
with Holm, Cliff's delta, OLS slope tests, rule of three), figures 1-4, tables 1-6, model check, TLA+/TLC, tests, Docker.

Deviations from the design document (also written into every `report.md`):

* **Resilience:** requires `n > (d+2)*ts` (and `n > 3*ts`), the conservative asynchronous bound, *not* the network-agnostic optimum `(d+1)*ts + ta < n`.
  `ta` only sets how many parties are corrupted when the network is asynchronous. Reaching the optimum needs the GTHR+-style machinery that is not reimplemented.
* **Baselines are stand-ins:** B2 = same engine with a fixed iteration count from `M` (also plays the async-only role, so B3 is not separate); B4 = raw-range first-round
  estimate. They are *not* line-by-line reimplementations of the cited papers, and I did not verify the cited papers' claims.
* **Reliable broadcast is ideal** (analytic cost `n + 2n^2`); Byzantine parties choose values / delays / withhold, but cannot equivocate on RB values. Gather and halting are simulated message by message.
* **Safe area:** exact LP for `d >= 2` while there are `<= 120` subsets, otherwise coordinate-wise trimmed box (box validity only); `d=1` exact.
* **Scale:** n up to 31 (61 in `full`) in the message-level simulator; the design's n = 301 and 1000 are not run.
* **G4 data:** the real Intel Berkeley Lab file is not bundled (150 MB). Without it a clearly labelled **synthetic surrogate** is used and `report.md` says so; put
  `data.txt` at `data/intel_lab/data.txt` (checksum your copy) to use real data.
* **AB4** (randomised double-certificate) is not implemented: the supplied text does not specify it.
* **Not done:** the synchronous lower-bound adaptation (no proof attempted), the literature check ("Phase 2" of the design: whether an adaptive network-agnostic multidimensional protocol already exists is *unverified*),
  partial synchrony with unknown GST, machine-checked proofs.
* Parquet is not used (CSV instead) to keep dependencies minimal.

## Reproducibility

Seeds are derived deterministically from `NAA_SEED`; every run record includes its seed; `config.json` stores effective settings and the git hash.
The protocol constants (`c = 2`, check threshold `= eps`) are frozen in `protocol/params.py` and were set a priori from the nominal contraction factor;
`scripts/pilot.py` re-checks them on disjoint seeds. Same seed + profile -> identical `raw/*.csv`.

## Git workflow used for this repo

```bash
git init && git add -A && git commit -m "..."      # committed in logical batches: core -> modules -> experiments/analysis -> tests/docs
git remote add origin <repo-link> && git push -u origin <branch>
```
