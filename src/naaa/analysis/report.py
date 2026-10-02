"""Turn raw run records into tables, figures and a self-contained report.md.

Every number in report.md is computed from the records of THIS run; nothing is hard-coded
except the static related-work matrix and the list of deviations from the design document.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import stats as S  # noqa: E402

log = logging.getLogger("naaa.report")

PALETTE = {"proposed": "#0072B2", "B1_sync_fixed": "#D55E00", "B2_fixed": "#009E73",
           "B4_classic_adaptive": "#CC79A7", "AB1_raw_dhat": "#E69F00", "AB2_no_substitution": "#56B4E9",
           "AB3_plain_collect": "#999999"}
NET_LABEL = {("sync", "uniform"): "sync", ("async", "exp"): "async-exp", ("async", "lognormal"): "async-lognorm",
             ("async", "pareto"): "async-pareto", ("async", "split"): "async-split"}


def _cell(v) -> str:
    if isinstance(v, (float, np.floating)):
        if v != v:
            return ""
        if float(v).is_integer() and abs(v) < 1e15:
            return str(int(v))
    return str(v)


def to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data)_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_cell(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, out: Path, name: str) -> str:
    (out / "tables").mkdir(exist_ok=True)
    df.to_csv(out / "tables" / f"{name}.csv", index=False)
    md = to_md(df)
    (out / "tables" / f"{name}.md").write_text(md)
    return md


def _viol(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean failure columns (liveness / agreement / validity)."""
    df = df.copy()
    tol = 1e-6 * np.maximum(1.0, df["DH"].fillna(1.0))
    df["fail_live"] = ~df["terminated"].astype(bool)
    df["fail_agree"] = df["terminated"].astype(bool) & (df["spread_over_eps"] > 1.0 + 1e-9)
    df["fail_valid"] = df["terminated"].astype(bool) & ((df["hull_violation"] > tol) | df["hull_violation"].isna())
    return df


# ----------------------------------------------------------------------------- tables ---
def table_related() -> pd.DataFrame:
    return pd.DataFrame([
        dict(work="Ghinea, Liu-Zhang, Wattenhofer (SPAA'23)", model="network-agnostic", dims="d-dim",
             resilience="(d+1)ts+ta<n", rounds="fixed iterations from range bound", adaptive="no (as described in the design doc)"),
        dict(work="Mendes-Herlihy style async D-AA", model="asynchronous", dims="d-dim",
             resilience="n>(d+2)t", rounds="fixed iterations", adaptive="no"),
        dict(work="Abraham-Amit-Dolev style async 1-D AA", model="asynchronous", dims="1-D",
             resilience="n>3t", rounds="estimated from first-round range", adaptive="yes (raw range)"),
        dict(work="This implementation (proposed)", model="timing-agnostic, event-driven", dims="d-dim (exact LP d<=5-ish, box beyond)",
             resilience="n>(d+2)ts (conservative; see README)", rounds="R(certified diameter) + halting check",
             adaptive="yes (safe-area certified diameter)"),
    ])


def table_slopes(frames: dict) -> pd.DataFrame:
    rows = []
    for exp in ("E1", "E1_d2", "E1_async"):
        df = frames.get(exp)
        if df is None or df.empty:
            continue
        for m, g in df[df["terminated"] == True].groupby("method"):  # noqa: E712
            o = S.ols_slope(g["log2_DH_eps"], g["R_iter"])
            rows.append(dict(exp=exp, method=m, metric="R_iter", slope=round(o["slope"], 3),
                             ci95=f"[{o['lo']:.3f}, {o['hi']:.3f}]", p_slope_eq_0=f"{o['p']:.2g}",
                             intercept=round(o["intercept"], 2), r2=round(o["r2"], 3), n=o["n"]))
            if m == "proposed" and g["R_oracle"].notna().any():
                o = S.ols_slope(g["log2_DH_eps"], g["R_oracle"])
                rows.append(dict(exp=exp, method="proposed (oracle iters)", metric="R_oracle", slope=round(o["slope"], 3),
                                 ci95=f"[{o['lo']:.3f}, {o['hi']:.3f}]", p_slope_eq_0=f"{o['p']:.2g}",
                                 intercept=round(o["intercept"], 2), r2=round(o["r2"], 3), n=o["n"]))
    return pd.DataFrame(rows)


def table_main(e2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    e2 = _viol(e2)
    for (mode, delay, m), g in e2.groupby(["mode", "delay", "method"], sort=False):
        t = g[g["terminated"] == True]  # noqa: E712
        med = np.median(t["R_iter"]) if len(t) else float("nan")
        lo, hi = S.bootstrap_ci(t["R_iter"], np.median)
        rows.append(dict(network=NET_LABEL.get((mode, delay), f"{mode}-{delay}"), method=m, runs=len(g),
                         termination_rate=f"{g['terminated'].mean():.2f}",
                         R_iter_median_ci=S.fmt_ci(med, lo, hi), causal_rounds_median=float(np.median(t["causal_rounds"])) if len(t) else float("nan"),
                         T_term_median=round(float(np.median(t["T_term"])), 1) if len(t) else float("nan"),
                         max_spread_over_eps=round(float(t["spread_over_eps"].max()), 3) if len(t) else float("nan"),
                         agreement_fail=int(g["fail_agree"].sum()), validity_fail=int(g["fail_valid"].sum())))
    return pd.DataFrame(rows)


def table_safety(frames: dict) -> pd.DataFrame:
    allr = pd.concat([f for f in frames.values() if f is not None and not f.empty], ignore_index=True)
    allr = _viol(allr)
    rows = []
    for (m, adv, mode), g in allr.groupby(["method", "adversary", "mode"]):
        n = len(g)
        k_a, k_v, k_l = int(g["fail_agree"].sum()), int(g["fail_valid"].sum()), int(g["fail_live"].sum())
        t = g[g["terminated"] == True]  # noqa: E712
        rows.append(dict(method=m, adversary=adv, mode=mode, runs=n, not_terminated=k_l, agreement_fail=k_a,
                         validity_fail=k_v, max_spread_over_eps=round(float(t["spread_over_eps"].max()), 3) if len(t) else float("nan"),
                         violation_rate_ub95=(f"<= {S.rule_of_three(n):.4f}" if (k_a + k_v) == 0 else f"observed {(k_a + k_v) / n:.4f}")))
    return pd.DataFrame(rows)


def table_comm(e4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (n, d), g in e4.groupby(["n", "d"]):
        p = g[g["method"] == "proposed"]
        b = g[g["method"] == "B2_fixed"]
        if p.empty or b.empty:
            continue
        mp, mb = p["msgs"].to_numpy(), b["msgs"].to_numpy()
        rows.append(dict(n=n, d=d, ts=int(g["ts"].iloc[0]), R_proposed=float(p["R_iter"].median()), R_B2=float(b["R_iter"].median()),
                         msgs_proposed=int(np.median(mp)), msgs_B2=int(np.median(mb)),
                         msg_ratio=round(float(np.median(mp) / np.median(mb)), 3),
                         bytes_proposed=int(p["bytes"].median()), bytes_B2=int(b["bytes"].median()),
                         bytes_ratio=round(float(p["bytes"].median() / b["bytes"].median()), 3),
                         events_per_run=int(p["events"].median())))
    return pd.DataFrame(rows).sort_values(["d", "n"])


def table_ablation(e3: pd.DataFrame) -> pd.DataFrame:
    e3 = _viol(e3)
    comps = []
    for (adv, mode), g in e3.groupby(["adversary", "mode"]):
        base = g[g["method"] == "proposed"].set_index("seed")
        for m in ["AB1_raw_dhat", "AB2_no_substitution", "AB3_plain_collect"]:
            a = g[g["method"] == m].set_index("seed")
            common = base.index.intersection(a.index)
            both = [s for s in common if base.loc[s, "terminated"] and a.loc[s, "terminated"]]
            pw = S.paired_wilcoxon(a.loc[both, "R_iter"], base.loc[both, "R_iter"]) if both else float("nan")
            comps.append(dict(adversary=adv, mode=mode, ablation=m, runs=len(a),
                              R_proposed_median=float(base["R_iter"].median()),
                              R_ablation_median=float(a["R_iter"].median()),
                              cliffs_delta=round(S.cliffs_delta(a["R_iter"].dropna(), base["R_iter"].dropna()), 3),
                              termination_rate=round(float(a["terminated"].mean()), 3),
                              agreement_fail=int(a["fail_agree"].sum()), validity_fail=int(a["fail_valid"].sum()),
                              p_wilcoxon=pw))
    df = pd.DataFrame(comps)
    if not df.empty:
        ps = df["p_wilcoxon"].fillna(1.0).tolist()
        df["p_holm"] = [round(v, 4) for v in S.holm(ps)]
        df["p_wilcoxon"] = df["p_wilcoxon"].round(4)
    return df


# ---------------------------------------------------------------------------- figures ---
def _style(ax):
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig1(e1: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    e1 = e1.copy()
    e1["lvl"] = e1["log2_DH_eps"].round()
    lv = sorted(e1["lvl"].unique())
    for m, g in e1[e1["terminated"] == True].groupby("method"):  # noqa: E712
        med, lo, hi = [], [], []
        for l in lv:
            v = g[g["lvl"] == l]["R_iter"]
            med.append(np.median(v) if len(v) else np.nan)
            a, b = S.bootstrap_ci(v, np.median) if len(v) else (np.nan, np.nan)
            lo.append(a); hi.append(b)
        ax.plot(lv, med, "-o", ms=4, color=PALETTE.get(m), label=m)
        ax.fill_between(lv, lo, hi, color=PALETTE.get(m), alpha=0.18)
    p = e1[(e1["method"] == "proposed") & e1["R_oracle"].notna()]
    if not p.empty:
        ax.plot(lv, [p[p["lvl"] == l]["R_oracle"].median() for l in lv], "--", color=PALETTE["proposed"], alpha=0.7,
                label="proposed: iterations actually needed (oracle)")
    xx = np.array(lv, dtype=float)
    ax.plot(xx, xx, ":", color="k", lw=1, label="log2(D_H/eps)")
    ax.set_xlabel("log2(D_H / eps)"); ax.set_ylabel("iterations R_term (max over honest)")
    ax.set_title("E1: rounds vs honest diameter (sync, adversary A2)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0)); _style(ax)
    ins = fig.add_axes([0.56, 0.14, 0.28, 0.17])
    g = e1[(e1["method"] == "proposed") & (e1["terminated"] == True)]  # noqa: E712
    if not g.empty:
        xv = np.array(lv, dtype=float)
        med = np.array([g[g["lvl"] == l]["R_iter"].median() for l in lv])
        ok = xv >= 4
        ins.plot(xv[ok], med[ok] / S.loglog_bound(2.0 ** xv[ok]), "-o", ms=3, color=PALETTE["proposed"])
    ins.set_title("R / (log x / loglog x)", fontsize=7); ins.tick_params(labelsize=6); ins.grid(alpha=0.25)
    fig.savefig(out / "figures" / "fig1_adaptivity.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig2(e2: pd.DataFrame, out: Path):
    e2 = _viol(e2)
    e2["net"] = [NET_LABEL.get((a, b), f"{a}-{b}") for a, b in zip(e2["mode"], e2["delay"])]
    nets = list(dict.fromkeys(e2["net"]))
    methods = list(dict.fromkeys(e2["method"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    w = 0.8 / len(methods)
    for ai, (col, lab) in enumerate([("R_iter", "iterations R_term"), ("T_term", "T_term / delta")]):
        ax = axes[ai]
        for mi, m in enumerate(methods):
            vals, errs, bad = [], [], []
            for net in nets:
                g = e2[(e2["net"] == net) & (e2["method"] == m)]
                t = g[g["terminated"] == True]  # noqa: E712
                vals.append(t[col].median() if len(t) else 0)
                lo, hi = S.bootstrap_ci(t[col], np.median) if len(t) else (0, 0)
                errs.append([max(0, vals[-1] - lo), max(0, hi - vals[-1])])
                bad.append(int(g["fail_agree"].sum() + g["fail_live"].sum()))
            xpos = np.arange(len(nets)) + mi * w
            ax.bar(xpos, vals, w, yerr=np.array(errs).T, color=PALETTE.get(m), label=m, capsize=2)
            for xp, v, b in zip(xpos, vals, bad):
                if b:
                    ax.text(xp, v, f"{b}x fail", ha="center", va="bottom", fontsize=6, color="red")
        ax.set_xticks(np.arange(len(nets)) + w * (len(methods) - 1) / 2); ax.set_xticklabels(nets, rotation=20, fontsize=8)
        ax.set_ylabel(lab); _style(ax)
        if col == "T_term":
            ax.set_yscale("log")
    axes[0].legend(fontsize=7); fig.suptitle("E2: same protocol, unknown network", fontsize=10)
    fig.savefig(out / "figures" / "fig2_network_modes.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig3(e3: pd.DataFrame, out: Path):
    g = e3[(e3["method"] == "proposed") & e3["dhat0_ratio_mean"].notna()]
    advs = sorted(g["adversary"].unique())
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    pos = np.arange(len(advs))
    d1 = [g[g["adversary"] == a]["dhat0_ratio_mean"].to_numpy() for a in advs]
    d2 = [g[g["adversary"] == a]["raw0_ratio_mean"].to_numpy() for a in advs]
    b1 = ax.boxplot(d1, positions=pos - 0.18, widths=0.3, patch_artist=True, manage_ticks=False)
    b2 = ax.boxplot(d2, positions=pos + 0.18, widths=0.3, patch_artist=True, manage_ticks=False)
    for b, c in ((b1, PALETTE["proposed"]), (b2, PALETTE["AB1_raw_dhat"])):
        for p in b["boxes"]:
            p.set_facecolor(c); p.set_alpha(0.6)
    ax.set_xticks(pos); ax.set_xticklabels(advs); ax.set_yscale("symlog", linthresh=1e-3); ax.set_ylim(bottom=0)
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_ylabel("estimate / D_H"); ax.set_title("E3: certified (safe-area) vs raw-range diameter estimate")
    ax.legend([b1["boxes"][0], b2["boxes"][0]], ["safe-area D-hat (proposed)", "raw view range (AB1 / B4)"], fontsize=7)
    _style(ax)
    fig.savefig(out / "figures" / "fig3_dhat_ratio.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def fig4(e6: pd.DataFrame, out: Path, source: str):
    p = e6[e6["method"] == "proposed"].set_index("epoch")
    b = e6[e6["method"] == "B2_fixed"].set_index("epoch")
    ep = p.index.intersection(b.index)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for e in ep:
        ax.plot([0, 1], [b.loc[e, "R_iter"], p.loc[e, "R_iter"]], "-", color="#bbbbbb", lw=0.6)
    ax.scatter(np.zeros(len(ep)), b.loc[ep, "R_iter"], color=PALETTE["B2_fixed"], label="B2_fixed", s=14)
    ax.scatter(np.ones(len(ep)), p.loc[ep, "R_iter"], color=PALETTE["proposed"], label="proposed", s=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["B2_fixed", "proposed"]); ax.set_ylabel("iterations R_term")
    ax.set_title(f"E6: per-epoch rounds on sensor data ({source})", fontsize=9); _style(ax)
    fig.savefig(out / "figures" / "fig4_sensor_epochs.png", dpi=150, bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------------------ report ---
def analyze(frames: dict, modelcheck: list, tlc: dict, out: Path, meta: dict) -> dict:
    (out / "figures").mkdir(exist_ok=True)
    md: dict[str, str] = {}
    summary: dict = {"meta": meta}
    frames = {k: v for k, v in frames.items() if v is not None and not v.empty}
    ok_frames = {k: v[v["method"].notna()] for k, v in frames.items()}

    md["T1 related work"] = save_table(table_related(), out, "table1_related_work")
    sl = table_slopes(ok_frames)
    md["T-slopes (adaptivity, E1)"] = save_table(sl, out, "table_e1_slopes")
    if "E2" in ok_frames:
        md["T2 main results by network (E2)"] = save_table(table_main(ok_frames["E2"]), out, "table2_main_by_network")
    safety = table_safety(ok_frames)
    md["T3 safety (all experiments)"] = save_table(safety, out, "table3_safety")
    if "E4" in ok_frames:
        md["T4 communication / scalability (E4)"] = save_table(table_comm(ok_frames["E4"]), out, "table4_communication")
    mc = pd.DataFrame([{k: v for k, v in r.items() if k != "worst_example"} for r in modelcheck])
    if tlc:
        mc = pd.concat([mc, pd.DataFrame([tlc])], ignore_index=True)
    md["T5 model checking (E5)"] = save_table(mc, out, "table5_model_check")
    if "E3" in ok_frames:
        ab = table_ablation(ok_frames["E3"])
        md["T6 ablations (E3)"] = save_table(ab, out, "table6_ablations")

    if "E1" in ok_frames:
        fig1(ok_frames["E1"][ok_frames["E1"]["terminated"] == True], out)  # noqa: E712
    if "E2" in ok_frames:
        fig2(ok_frames["E2"], out)
    if "E3" in ok_frames:
        fig3(ok_frames["E3"], out)
    if "E6" in ok_frames:
        fig4(ok_frames["E6"], out, meta.get("g4_source", "?"))

    hyp = hypotheses(ok_frames, sl, modelcheck)
    summary["hypotheses"] = hyp
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(render_report(md, hyp, meta, ok_frames))
    return summary


def hypotheses(frames: dict, sl: pd.DataFrame, modelcheck: list) -> dict:
    h: dict = {}

    def slope(exp, method):
        r = sl[(sl["exp"] == exp) & (sl["method"] == method)] if not sl.empty else sl
        return None if r.empty else r.iloc[0]

    p, b = slope("E1", "proposed"), slope("E1", "B2_fixed")
    if p is not None and b is not None:
        lo = float(str(p["ci95"]).strip("[]").split(",")[0])
        h["H1"] = dict(supported=bool(lo > 0 and abs(float(b["slope"])) < 0.05),
                       text=f"proposed slope {p['slope']} {p['ci95']} (p={p['p_slope_eq_0']}); B2_fixed slope {b['slope']} {b['ci95']}")
    pa = slope("E1_async", "proposed")
    if p is not None and pa is not None and float(p["slope"]) != 0:
        ratio = float(pa["slope"]) / float(p["slope"])
        h["H2"] = dict(supported=bool(0.5 <= ratio <= 2.0), text=f"async/sync slope ratio = {ratio:.2f} (proposed)")
    allp = pd.concat([f[f["method"] == "proposed"] for f in frames.values()], ignore_index=True)
    allp = _viol(allp)
    nruns = len(allp)
    viol = int(allp["fail_agree"].sum() + allp["fail_valid"].sum() + allp["fail_live"].sum())
    mc_bad = sum(r["validity_violations"] + r["contraction_violations"] for r in modelcheck)
    h["H3"] = dict(supported=bool(viol == 0 and mc_bad == 0),
                   text=f"{viol} failures (liveness/agreement/validity) in {nruns} runs of the proposed protocol "
                        f"(95% upper bound on failure rate: {S.rule_of_three(nruns):.4f} if 0); "
                        f"model-check violations: {mc_bad}")
    if "E3" in frames:
        e3 = frames["E3"]
        e3p = e3[(e3["method"] == "proposed") & e3["dhat0_ratio_mean"].notna()]
        g1 = e3p[e3p["gen"] == "G1"]["dhat0_ratio_mean"]
        g3 = e3p[e3p["gen"] == "G3"]["dhat0_ratio_mean"]
        ab1 = e3[(e3["method"] == "AB1_raw_dhat") & e3["terminated"]]
        pp = e3[(e3["method"] == "proposed") & e3["terminated"]]
        h["H4"] = dict(
            supported=bool(len(g1) and g1.min() > 0.05),
            text=(f"D-hat/D_H on uniform inputs (G1): min {g1.min():.3f}, median {g1.median():.3f}; "
                  f"on cluster+honest-outlier inputs (G3): min {g3.min():.4f}, median {g3.median():.4f} "
                  f"(small by design: the certified diameter discards <= ts outliers). "
                  f"Under A2 AB1 needs median R={ab1[ab1['adversary'] == 'A2']['R_iter'].median()} vs "
                  f"proposed {pp[pp['adversary'] == 'A2']['R_iter'].median()}"))
    for exp in ("E1", "E2", "E3"):
        if exp in frames:
            q = frames[exp][(frames[exp]["method"] == "proposed") & frames[exp]["R_oracle"].notna()]
            if len(q):
                h[f"slack_{exp}"] = dict(text=f"{exp}: proposed median R_iter={q['R_iter'].median():.0f}, oracle iterations actually needed (median)={q['R_oracle'].median():.0f}")
    return h


DEVIATIONS = """\
* **Resilience.** The implementation requires `n > (d+2)*ts` (and `n > 3*ts` for Gather). This is the conservative
  asynchronous bound, not the network-agnostic optimum `(d+1)*ts + ta < n` of the cited work. Reaching the optimum needs
  the GTHR+/adaptive-view machinery of that work, which is not reimplemented. `ta` is kept as a parameter (it sets how many
  parties are corrupted when the network is asynchronous).
* **Baselines.** B2 is the *same engine* with a fixed iteration count derived from the a-priori bound `M`; it stands in for the
  fixed-iteration SPAA'23 protocol and for the async-only baseline (B3 is therefore not run separately). B4 is the raw-range
  first-round estimate without the halting check. These are not line-by-line reimplementations of the cited papers.
* **Reliable broadcast** is an ideal functionality with analytic message accounting (`n + 2n^2`); Gather and halting are
  simulated message by message.
* **Safe area.** Exact LP for d >= 2 while the number of `(m-ts)`-subsets is <= 120, else coordinate-wise trimmed box
  (box validity only). `d=1` is exact in closed form.
* **Scale.** The message-level simulator handles n up to ~60 in reasonable time; the design's n in {301, 1000} is not run.
* **AB4** (randomised double-certificate) is not implemented: the supplied text does not specify it.
* **TLA+.** `tla/AdaptiveAA.tla` is provided for TLC but is only executed when `NAA_TLC_JAR` is set and Java is available. The
  always-run E5 check is the exhaustive/sampled Python check of the abstract step (`modelcheck.py`).
* **Proofs.** `docs/THEORY.md` states the theorems with proofs of the parts that are proven (Gather common core, validity,
  contraction in d=1) and marks the remaining statements (adaptive-R sufficiency, lower bound adaptation) as *conjectures /
  proof sketches*. The simulation does not prove them.
"""


def render_report(md: dict, hyp: dict, meta: dict, frames: dict) -> str:
    L = ["# Run report", "",
         f"- profile: `{meta.get('profile')}`, base seed: `{meta.get('seed')}`, workers: {meta.get('workers')}",
         f"- G4 data source: **{meta.get('g4_source')}**" + ("  (synthetic surrogate: NOT real-data evidence)" if "synthetic" in str(meta.get("g4_source")) else ""),
         f"- total simulated runs: {meta.get('total_runs')}, wall time: {meta.get('wall_s', 0):.1f}s", ""]
    L += ["## Hypotheses (evaluated on this run's data)", ""]
    for k, v in hyp.items():
        flag = "" if "supported" not in v else (" — **supported**" if v["supported"] else " — **NOT supported**")
        L.append(f"- **{k}**{flag}: {v['text']}")
    L += ["", "## Figures", ""]
    for f in ("fig1_adaptivity", "fig2_network_modes", "fig3_dhat_ratio", "fig4_sensor_epochs"):
        L.append(f"![{f}](figures/{f}.png)")
    L += [""]
    for name, text in md.items():
        L += [f"## {name}", "", text, ""]
    L += ["## Deviations from the design document and known limitations", "", DEVIATIONS]
    return "\n".join(L)
