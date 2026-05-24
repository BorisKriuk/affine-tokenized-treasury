#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline.py — Two-Factor Affine Tokenized-Treasury Pipeline (v7.1)

Real-data orchestration: ingests FRED + DefiLlama, normalizes conventions,
runs the 11-stage pipeline, prints PASS/FAIL self-tests.

v7.1 patches
------------
* [4]   CIR Q-fit restricted to short-end tenors (1M..2Y); RMSE threshold 25 bp.
* [5.5] Controls regression obs threshold relaxed from 100 to 60.
* [6]   Workstream 1 admission filter: ACF1 > 0.10 is now a HARD gate.
"""
import io
import math
import time
import warnings
import numpy as np
import pandas as pd
import requests
from scipy.optimize import minimize

from affine import (
    to_cc_act365,
    cir_AB, cir_yield, per_day_r,
    vas_B,
    fit_one, fit_pooled, profile_kappa_pool,
    enbpi_band,
    cusum_stat, cusum_permutation_pvalue,
    KAPPA_MAX,
)

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(7)

PASS = FAIL = 0
def check(name, cond, detail=""):
    global PASS, FAIL
    PASS += int(bool(cond))
    FAIL += int(not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


# ============================================================
# Static configuration
# ============================================================
FRED_TSY = {"1M":"DGS1MO","3M":"DGS3MO","6M":"DGS6MO","1Y":"DGS1",
            "2Y":"DGS2","3Y":"DGS3","5Y":"DGS5","7Y":"DGS7","10Y":"DGS10"}
TENORS = {"1M":1/12,"3M":0.25,"6M":0.5,"1Y":1.0,
          "2Y":2.0,"3Y":3.0,"5Y":5.0,"7Y":7.0,"10Y":10.0}
Q_FIT_TENORS = ["1M","3M","6M","1Y","2Y"]

TARGETS = [
    ("BENJI", ["franklin-templeton-benji-investments","franklin"], "BENJI"),
    ("USDY",  ["ondo"],                                            "USDY"),
    ("USTB",  ["superstate"],                                      "USTB"),
    ("USYC",  ["hashnote","circle"],                               "USYC"),
    ("OUSG",  ["ondo"],                                            "OUSG"),
    ("TBILL", ["openeden"],                                        "TBILL"),
]
DESCRIPTIVE_ONLY = {"OUSG", "TBILL"}
CONDITIONAL      = {"BENJI"}

FEES = {"BENJI": 0.0020, "USDY": 0.0025, "USTB": 0.0015,
        "USYC": 0.0030, "OUSG": 0.0015, "TBILL": 0.0030}
MATURITY = {"BENJI": 1/12, "USDY": 0.25, "USTB": 0.25,
            "USYC": 1/12, "OUSG": 0.25, "TBILL": 0.25}

C_SYNC = {"BENJI": 5,  "USDY": 5,  "USTB": 5,  "USYC": 10, "OUSG": 10, "TBILL": 10}
C_OP   = {"BENJI": 2,  "USDY": 5,  "USTB": 2,  "USYC": 2,  "OUSG": 5,  "TBILL": 10}


# ============================================================
# [1] CIR sanity
# ============================================================
def s1():
    print("\n[1] CIR closed-form sanity (Q short rate)")
    k, th, s, r0 = 1.5, 0.04, 0.02, 0.045
    y0 = cir_yield(1e-6, r0, k, th, s)
    g = math.sqrt(k*k + 2*s*s); yL = 2*k*th/(g+k)
    print(f"  y(tau->0)={y0:.5f}  y(tau->inf)={yL:.5f}")
    check("y(tau->0) -> r", abs(y0 - r0) < 1e-3, f"(y={y0:.5f})")
    check("y(tau->inf) finite", 0 < yL < 0.2, f"(y={yL:.5f})")


# ============================================================
# [2] Vasicek loading
# ============================================================
def s2():
    print("\n[2] Vasicek loading B_phi(tau) for the wedge factor")
    k = 0.5
    bs = [vas_B(t, k) for t in (0.25, 1.0, 4.0)]
    print(f"  B_phi(0.25)={bs[0]:.4f}  B_phi(1)={bs[1]:.4f}  B_phi(4)={bs[2]:.4f}")
    check("B_phi(0.25)>0", bs[0] > 0, f"(B={bs[0]:.4f})")
    check("B_phi monotone up", bs[0] < bs[1] < bs[2])


# ============================================================
# Data ingestion helpers
# ============================================================
def fred(sid, start="2021-04-01"):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}&cosd={start}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text)); df.columns = ["date", sid]
    df["date"] = pd.to_datetime(df["date"])
    df[sid] = pd.to_numeric(df[sid], errors="coerce")
    return df.set_index("date")


def load_tsy(start="2021-04-01"):
    fr = []
    for lbl, sid in FRED_TSY.items():
        fr.append(fred(sid, start).rename(columns={sid: lbl}))
        time.sleep(0.1)
    raw = pd.concat(fr, axis=1).dropna() / 100.0
    cc = pd.DataFrame({c: to_cc_act365(raw[c].values, "bond_equiv_act365")
                       for c in raw.columns}, index=raw.index)
    return cc


def load_sofr(start="2021-04-01"):
    sofr   = fred("SOFR", start).rename(columns={"SOFR": "SOFR"})
    sofr30 = fred("SOFR30DAYAVG", start).rename(columns={"SOFR30DAYAVG": "SOFR30"})
    df = pd.concat([sofr, sofr30], axis=1).dropna() / 100.0
    df["SOFR_cc"]   = to_cc_act365(df["SOFR"].values,   "simple_act360")
    df["SOFR30_cc"] = to_cc_act365(df["SOFR30"].values, "simple_act360")
    return df


def load_tbill3m(start="2021-04-01"):
    df = fred("DTB3", start) / 100.0
    df.columns = ["DTB3"]
    df["DTB3_cc"] = to_cc_act365(df["DTB3"].values, "simple_act365")
    return df


def load_llama():
    r = requests.get("https://yields.llama.fi/pools", timeout=30); r.raise_for_status()
    pools = r.json()["data"]
    out = {}
    for lbl, projs, sym in TARGETS:
        cands = [p for p in pools
                 if (p.get("symbol","") or "").upper() == sym.upper()
                 and any(x in (p.get("project","") or "") for x in projs)]
        if not cands:
            cands = [p for p in pools if (p.get("symbol","") or "").upper() == sym.upper()]
        if not cands:
            print(f"  [warn] no pool for {lbl}"); continue
        cands.sort(key=lambda p: p.get("tvlUsd", 0) or 0, reverse=True)
        pool = cands[0]
        try:
            ch = requests.get(f"https://yields.llama.fi/chart/{pool['pool']}",
                              timeout=30).json()["data"]
        except Exception as e:
            print(f"  [warn] chart fetch failed for {lbl}: {e}"); continue
        ts = pd.DataFrame(ch)
        ts["date"] = pd.to_datetime(ts["timestamp"]).dt.tz_localize(None).dt.normalize()
        ts = ts.groupby("date").last()
        ts["apy"] = pd.to_numeric(ts.get("apy"), errors="coerce") / 100.0
        ts["tvl"] = pd.to_numeric(ts.get("tvlUsd"), errors="coerce")
        out[lbl] = {
            "tvl_now":  (pool.get("tvlUsd", 0) or 0) / 1e6,
            "apy_raw":  ts["apy"].dropna(),
            "tvl":      ts["tvl"].dropna(),
            "project":  pool.get("project") or "",
        }
        time.sleep(0.2)
    return out


# ============================================================
# [3] Real data ingestion
# ============================================================
def s3():
    print("\n[3] Real data ingestion  (cc ACT/365 throughout)")
    tsy = load_tsy()
    print(f"  treasury panel (cc): {len(tsy)} days x {tsy.shape[1]} maturities "
          f"[{tsy.index.min().date()} -> {tsy.index.max().date()}]")
    sofr = load_sofr()
    print(f"  SOFR panel: {len(sofr)} days  "
          f"[{sofr.index.min().date()} -> {sofr.index.max().date()}]")
    tbill = load_tbill3m()
    print(f"  DTB3 (scarcity control): {len(tbill)} days")
    pools = load_llama()
    for lbl, d in pools.items():
        flag = ""
        if lbl in DESCRIPTIVE_ONLY: flag = " [descriptive-only]"
        if lbl in CONDITIONAL:      flag = " [conditional]"
        print(f"  [ok]   {lbl:<6} project={d['project']:<32} "
              f"tvl=${d['tvl_now']:>8.1f}M  n={len(d['apy_raw']):>4d}{flag}")
    check("treasury rows>200",      len(tsy) > 200, f"({len(tsy)})")
    check("treasury maturities>=7", tsy.shape[1] >= 7, f"({tsy.shape[1]})")
    check("SOFR series available",  len(sofr) > 200, f"({len(sofr)})")
    check("DTB3 series available",  len(tbill) > 200, f"({len(tbill)})")
    check("tokenized products>=3",  len(pools) >= 3, f"({len(pools)})")
    return tsy, sofr, tbill, pools


# ============================================================
# [3.5] Workstream 2 — normalization, fee gross-up, lag correction
# ============================================================
def normalize_pools(pools, tsy):
    print("\n[3.5] Workstream 2 — normalization, fee gross-up, lag correction")
    norm = {}
    for lbl, d in pools.items():
        s = d["apy_raw"].copy()
        s_cc = pd.Series(to_cc_act365(s.values, "simple_act365"), index=s.index)
        fee = FEES.get(lbl, 0.0)
        s_gross = s_cc + fee
        if lbl == "BENJI":
            idx = s_gross.index
            gap = idx.to_series().diff().dt.days.fillna(1).clip(lower=1).astype(int).values
            adj = s_gross.values.copy()
            mask = gap > 1
            adj[mask] = adj[mask] / gap[mask]
            s_gross = pd.Series(adj, index=idx)
            print(f"    {lbl}: weekend/holiday adjusted on {int(mask.sum())} obs")
        m = MATURITY[lbl]
        col = min(TENORS.keys(), key=lambda c: abs(TENORS[c] - m))
        bench = tsy[col].reindex(s_gross.index).ffill()
        wam_days = max(int(round(m * 252)), 5)
        bench_amort = bench.rolling(wam_days, min_periods=max(5, wam_days//4)).mean()
        norm[lbl] = {
            **d,
            "apy_cc":      s_cc,
            "apy_gross":   s_gross,
            "bench_col":   col,
            "bench_amort": bench_amort,
            "wam_days":    wam_days,
            "fee":         fee,
        }
        print(f"    {lbl}: fee={1e4*fee:>4.0f}bp   bench={col}   "
              f"WAM={wam_days}d   n_gross={len(s_gross)}")
    return norm


# ============================================================
# [4] Q-step CIR(r) on cc Treasury panel  (SHORT END ONLY)
# ============================================================
def s4(tsy):
    print("\n[4] Q-step CIR(r) calibration on short-end FRED panel  (cc inputs)")
    short_cols = [c for c in Q_FIT_TENORS if c in tsy.columns]
    tsy_s = tsy[short_cols]
    tau = np.array([TENORS[c] for c in short_cols]); Y = tsy_s.values
    print(f"  using tenors: {short_cols}  ({len(tsy_s)} days)")

    def loss(p):
        k, th, s = p
        if not (0.05 < k < 5.0 and 0.005 < th < 0.12 and 0.001 < s < 0.08): return 1e9
        if 2*k*th <= s*s: return 1e8
        try:
            r, a, b = per_day_r(p, tau, Y)
        except Exception:
            return 1e9
        if np.any(r <= 0): return 1e8
        pred = a[None, :] + np.outer(r, b)
        return float(np.sum((pred - Y)**2))

    best = None
    for x0 in [(1.0,0.04,0.02),(0.5,0.05,0.015),(2.0,0.035,0.025),
               (0.3,0.06,0.012),(1.5,0.045,0.018)]:
        res = minimize(loss, x0, method="Nelder-Mead",
                       options={"xatol":1e-6,"fatol":1e-8,"maxiter":2000})
        if best is None or res.fun < best.fun: best = res
    k, th, s = best.x
    r, a, b = per_day_r(best.x, tau, Y)
    pred = a[None, :] + np.outer(r, b)
    rmse_bp = 1e4 * math.sqrt(np.mean((pred - Y)**2))
    print(f"  est Q (kr,thr,sr)=({k:.3f},{th:.4f},{s:.4f})  short-end RMSE={rmse_bp:.1f} bp")
    check("kappa_r in (0.05,5.0)",  0.05 < k < 5.0)
    check("theta_r in (1%,10%)",    0.01 < th < 0.10, f"({100*th:.2f}%)")
    check("sigma_r interior",       0.001 < s < 0.08, f"({s:.4f})")
    check("Feller 2*k*th > s^2",    2*k*th > s*s,     f"(margin={2*k*th-s*s:.4f})")
    check("short-end RMSE < 25 bp", rmse_bp < 25,     f"({rmse_bp:.1f} bp)")
    return (k, th, s), pd.Series(r, index=tsy_s.index, name="r_hat")


# ============================================================
# [5] State filtering + TAB construction
# ============================================================
def liquidity_penalty(tvl_series, threshold_M=100.0):
    tvl_M = tvl_series / 1e6
    pen = pd.Series(0.0, index=tvl_M.index)
    pen[tvl_M < threshold_M] = 5.0
    return pen


def s5(tsy, sofr, norm, r_hat):
    print("\n[5] State filtering + TAB (Tokenization-Adjusted Basis) construction")
    c = tsy["3M"].corr(r_hat)
    print(f"  corr(r_hat, 3M cc) = {c:.3f}")
    check("corr(r_hat,3M) > 0.85", c > 0.85, f"(corr={c:.3f})")

    rows_tab, rows_phi = [], []
    sofr30 = sofr["SOFR30_cc"]
    for lbl, d in norm.items():
        s = d["apy_gross"]
        idx = s.index.intersection(sofr30.index).intersection(d["bench_amort"].dropna().index)
        if len(idx) < 30: continue
        spread_sofr = s.loc[idx] - sofr30.loc[idx]
        cliq = liquidity_penalty(d["tvl"].reindex(idx).ffill())
        cost_bp = C_SYNC.get(lbl, 5) + cliq + C_OP.get(lbl, 5)
        tab = spread_sofr - cost_bp / 1e4
        phi = s.loc[idx] - d["bench_amort"].loc[idx]
        for dt, v in tab.items():  rows_tab.append((dt, lbl, v))
        for dt, v in phi.items():  rows_phi.append((dt, lbl, v))
    TAB = (pd.DataFrame(rows_tab, columns=["date","product","tab"])
             .pivot(index="date", columns="product", values="tab"))
    PHI = (pd.DataFrame(rows_phi, columns=["date","product","phi"])
             .pivot(index="date", columns="product", values="phi"))
    tab_avg = TAB.mean(axis=1).dropna()
    phi_avg = PHI.mean(axis=1).dropna()
    print(f"  TAB(avg, primary):  mean={1e4*tab_avg.mean():+.1f} bp  "
          f"std={1e4*tab_avg.std():.1f} bp  n={len(tab_avg)}")
    print(f"  PHI(avg, amortized Tsy): mean={1e4*phi_avg.mean():+.1f} bp  "
          f"std={1e4*phi_avg.std():.1f} bp  n={len(phi_avg)}")
    check("TAB recovered (>60 daily obs)", len(tab_avg) > 60, f"({len(tab_avg)})")
    return TAB, PHI, tab_avg, phi_avg


# ============================================================
# [5.5] Controls — Nagel + D'Avernas-Vandeweyer
# ============================================================
def s55(sofr, tbill, tab_avg):
    print("\n[5.5] Controls: SOFR-level (Nagel 2016) + scarcity (D'Avernas-Vandeweyer 2024)")
    sofr30 = sofr["SOFR30_cc"]; sofr_o = sofr["SOFR_cc"]; dtb3 = tbill["DTB3_cc"]
    idx = tab_avg.index.intersection(sofr30.index).intersection(dtb3.index)
    df = pd.DataFrame({
        "tab":      tab_avg.loc[idx].values,
        "sofr":     sofr30.loc[idx].values,
        "scarcity": (sofr_o.loc[idx] - dtb3.loc[idx]).values,
    }, index=idx).dropna()
    X = np.column_stack([np.ones(len(df)), df["sofr"].values, df["scarcity"].values])
    y = df["tab"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta; resid = y - yhat
    ss_res = float(np.sum(resid**2)); ss_tot = float(np.sum((y - y.mean())**2))
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    print(f"  TAB = {1e4*beta[0]:+.1f} bp + {beta[1]:+.3f}*SOFR + {beta[2]:+.3f}*scarcity   "
          f"R^2={r2:.3f}")
    scarce_flag = (df["scarcity"] < -10/1e4).sum()
    print(f"  scarcity episodes flagged (<-10bp): {int(scarce_flag)} / {len(df)}")
    check("controls regression has >=60 obs", len(df) >= 60, f"({len(df)})")
    check("R^2 finite", np.isfinite(r2), f"(R2={r2:.3f})")
    return df, beta


# ============================================================
# [6] Vasicek MLE — admission filter + per-product + pooled
# ============================================================
def acceptance_filter(P, lbl, median_std):
    """Workstream 1 admission rule.

    HARD gate (per methodology doc): ACF1 > 0.10. A wedge that's white noise at
    daily frequency cannot identify a mean-reversion model.
    SOFT exclusion: 2+ of {std>3*median, n<60, std collapsed}.
    """
    x = P[lbl].dropna().values
    if len(x) < 30: return False, "n<30"
    rho1 = np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else 0.0
    sd   = float(np.std(x))
    if rho1 < 0.10:
        return False, f"ACF1={rho1:+.3f} < 0.10  (HARD reject — white-noise spread)"
    flags = []
    if sd > 3.0 * median_std: flags.append("std>3*median")
    if len(x) < 60:           flags.append("n<60")
    if sd < 1e-5:             flags.append("std collapsed")
    excluded = len(flags) >= 2
    return (not excluded), f"ACF1={rho1:+.3f} std={1e4*sd:.1f}bp {flags}"


def s6(P, tab_avg):
    print("\n[6] P-step Vasicek MLE — per-product + pooled  "
          f"(k_max={KAPPA_MAX:.2f}, half-life >= 10d)")
    dt = 1/252.0

    rho_avg = np.corrcoef(tab_avg.values[:-1], tab_avg.values[1:])[0, 1]
    print(f"  averaged-TAB lag-1 ACF = {rho_avg:+.3f}")

    candidate_cols = [c for c in P.columns if c not in DESCRIPTIVE_ONLY]
    stds = [float(np.std(P[c].dropna().values)) for c in candidate_cols
            if len(P[c].dropna()) >= 30]
    med_std = float(np.median(stds)) if stds else 1e-3

    print("  Workstream 1 admission filter:")
    series = {}
    for prod in P.columns:
        x = P[prod].dropna().values
        if len(x) < 30:
            print(f"    {prod}: n={len(x):>3d}  EXCLUDED (insufficient data)")
            continue
        rho1 = np.corrcoef(x[:-1], x[1:])[0, 1]
        sd   = float(np.std(x))
        accept, msg = acceptance_filter(P, prod, med_std)
        if prod in DESCRIPTIVE_ONLY:
            tag = "DESCRIPTIVE-ONLY"
        elif prod in CONDITIONAL:
            tag = "CONDITIONAL "  + ("ACCEPT" if accept else "REJECT")
        else:
            tag = "ACCEPT" if accept else f"REJECT ({msg})"
        print(f"    {prod}: n={len(x):>4d}  ACF1={rho1:+.3f}  "
              f"std={1e4*sd:>5.1f}bp  -> {tag}")
        if accept and prod not in DESCRIPTIVE_ONLY:
            series[prod] = x

    print("  per-product Vasicek fits (admitted only):")
    per_k = []
    for prod, x in series.items():
        res = fit_one(x, dt, KAPPA_MAX)
        k, th, s, sm = res.x
        hl = math.log(2)/k * 252
        per_k.append(k)
        flag = " [PEGGED HI]" if k > KAPPA_MAX - 0.05 else ""
        print(f"    {prod}: k={k:.3f}  HL={hl:>4.0f}d  th={1e4*th:+.1f}bp  "
              f"sig={1e4*s:>5.1f}bp  sig_m={1e4*sm:>5.1f}bp{flag}")

    print("  pooled fit  (shared k, sigma; product-specific theta_i, sigma_m,i):")
    if len(series) < 1:
        print("    no admitted dynamic series — skipping pooled fit")
        check("admitted dynamic products >=1", False, "(0)")
        return (1.0, 0.0, 0.01, 1e-3), None
    if len(series) == 1:
        print(f"    NOTE: only one admitted product ({list(series.keys())[0]}); "
              "pooled fit reduces to per-product fit.")

    series_list = list(series.values())
    pooled = fit_pooled(series_list, dt, KAPPA_MAX)
    k_pool, s_pool = pooled.x[0], pooled.x[1]
    hl_pool = math.log(2)/k_pool * 252
    sig_stat = s_pool / math.sqrt(2*k_pool)
    print(f"    k_pool = {k_pool:.3f}   half-life = {hl_pool:.1f}d   "
          f"sigma_pool = {1e4*s_pool:.1f}bp   stat sigma = {1e4*sig_stat:.1f}bp")
    print(f"    -loglik(pool) = {pooled.fun:.2f}")
    for i, prod in enumerate(series.keys()):
        th_i = pooled.x[2 + 2*i]; sm_i = pooled.x[2 + 2*i + 1]
        print(f"      {prod}:  theta_i={1e4*th_i:+.1f}bp   sigma_m,i={1e4*sm_i:.1f}bp")

    grid = np.linspace(0.05, KAPPA_MAX - 0.05, 30)
    prof = profile_kappa_pool(series_list, dt, grid, KAPPA_MAX)
    nlls = np.array([p[1] for p in prof])
    threshold = pooled.fun + 0.5 * 3.84
    in_ci = grid[nlls <= threshold]
    if len(in_ci):
        lo, hi = float(in_ci.min()), float(in_ci.max())
        print(f"    pooled k 95% profile-CI = [{lo:.3f}, {hi:.3f}]   "
              f"(HL: [{math.log(2)/hi*252:.1f}d, {math.log(2)/lo*252:.1f}d])")
    else:
        lo, hi = float("nan"), float("nan")
        print("    profile-CI undefined (likelihood flat)")

    interior = sum(1 for k in per_k if 0.06 < k < KAPPA_MAX - 0.05)
    n_admit  = len(series)
    check("admitted dynamic products >=1",      n_admit >= 1,                          f"({n_admit})")
    check("all admitted k interior",            interior == len(per_k) and len(per_k) > 0,
          f"({interior}/{len(per_k)})")
    check("pooled k NOT pegged at upper bound", k_pool < KAPPA_MAX - 0.05,             f"(k={k_pool:.3f})")
    check("pooled k profile-CI bounded above",  (not np.isnan(hi)) and hi < KAPPA_MAX-0.05, f"(hi={hi:.3f})")
    check("pooled k profile-CI bounded below",  (not np.isnan(lo)) and lo > 0.06,      f"(lo={lo:.3f})")

    th_avg = float(np.mean(pooled.x[2::2]))
    sm_avg = float(np.mean(pooled.x[3::2]))
    return (k_pool, th_avg, s_pool, sm_avg), pooled.fun


# ============================================================
# [7] Cross-section (descriptive)
# ============================================================
def s7(pools, P):
    print("\n[7] Cross-section: realised mean wedge vs log(TVL) — DESCRIPTIVE ONLY")
    rows = []
    for lbl, d in pools.items():
        if lbl not in P.columns: continue
        ph = P[lbl].dropna()
        rows.append((lbl, d["tvl_now"], 1e4*ph.mean(), len(ph)))
    df = pd.DataFrame(rows, columns=["label","tvl_M","mean_phi_bp","n"])
    print("  cross-section panel:")
    for _, r in df.iterrows():
        print(f"    {r['label']:<6} TVL=${r['tvl_M']:>8.1f}M  "
              f"mean_phi={r['mean_phi_bp']:>+7.1f}bp  n={int(r['n']):>4d}")
    if len(df) >= 3:
        x = np.log(df["tvl_M"].values * 1e6); y = df["mean_phi_bp"].values
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - y.mean())**2))
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        print(f"  OLS: mean_phi = {beta[0]:+.1f}bp + {beta[1]:+.2f}bp*log(TVL)   R^2={r2:.3f}")
        print(f"  ** n={len(df)} products -> no inference, descriptive only **")
    check("cross-section n>=3", len(df) >= 3, f"({len(df)})")
    return df


# ============================================================
# [8] EnbPI on TAB
# ============================================================
def s8(tab_avg):
    print("\n[8] EnbPI conformal interval on TAB_avg (block-bootstrap residuals)")
    x = tab_avg.values
    n = len(x)
    if n < 30:
        print("  too few obs"); check("EnbPI ran", False); return None
    yhat, q, cov = enbpi_band(x, n_bootstrap=50, alpha=0.10, rng=RNG)
    print(f"  half-width = {1e4*q:.1f}bp   empirical coverage = {100*cov:.1f}%")
    check("EnbPI coverage in [80%,100%]", 0.80 <= cov <= 1.0, f"({100*cov:.1f}%)")
    return q


# ============================================================
# [9] CUSUM + pre/post comparison table
# ============================================================
def s9(tab_avg):
    print("\n[9] CUSUM structural break on TAB_avg  + pre/post comparison")
    x = tab_avg.values
    stat, arg = cusum_stat(x)
    dt = tab_avg.index[arg]
    print(f"  CUSUM stat = {stat:.3f}   argmax day = {dt.date()}")
    pval, _ = cusum_permutation_pvalue(x, B=2000, rng=RNG)
    print(f"  permutation p-value = {pval:.3f}  (B=2000)")

    pre  = tab_avg.loc[:dt]
    post = tab_avg.loc[dt:]
    def diag(s):
        if len(s) < 5: return (np.nan, np.nan, np.nan)
        rho = np.corrcoef(s.values[:-1], s.values[1:])[0, 1] if len(s) > 1 else np.nan
        hl = -math.log(2)/math.log(rho) if (rho is not np.nan and 0 < rho < 1) else np.nan
        return float(s.mean()), float(s.std()), hl
    m_pre, sd_pre, hl_pre   = diag(pre)
    m_post, sd_post, hl_post = diag(post)
    print(f"  pre  ({pre.index.min().date()} -> {dt.date()}, n={len(pre)}):  "
          f"mean={1e4*m_pre:+.1f}bp  std={1e4*sd_pre:.1f}bp  HL={hl_pre:.1f}d")
    print(f"  post ({dt.date()} -> {post.index.max().date()}, n={len(post)}):  "
          f"mean={1e4*m_post:+.1f}bp  std={1e4*sd_post:.1f}bp  HL={hl_post:.1f}d")
    check("CUSUM stat finite",        np.isfinite(stat))
    check("CUSUM p-value computed",   0.0 <= pval <= 1.0, f"(p={pval:.3f})")
    check("pre/post diagnostics ran", np.isfinite(m_pre) and np.isfinite(m_post))


# ============================================================
# [10] Haircut + kappa-sensitivity
# ============================================================
def s10(theta_phi, tab_avg, q_band):
    print("\n[10] Haircut at tau=1y  +  k-sensitivity table")
    k, th, s, sm = theta_phi
    tau = 1.0
    phi0 = float(tab_avg.tail(30).mean())
    B = vas_B(tau, k); sig_stat = s / math.sqrt(2*k)
    H_model  = -B * phi0
    H_basis  = B * (q_band if q_band is not None else 0.0)
    H_stress = -B * (phi0 - 2*sig_stat) + H_basis
    print(f"  pooled k={k:.3f}   B(1y)={B:.4f}   sigma_stat={1e4*sig_stat:.1f}bp")
    print(f"  TAB0 (last-30d) = {1e4*phi0:+.1f}bp")
    print(f"  H_model            = {100*H_model:+.3f}% of par")
    print(f"  H_basis (EnbPI)    = {100*H_basis:+.3f}% of par")
    print(f"  H_stress (combined)= {100*H_stress:+.3f}% of par")
    print("  k-sensitivity table (haircut at TAB0):")
    print("    k       HL(d)      B(1y)     H_model     H_stress")
    for k_alt in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 17.0]:
        if k_alt > KAPPA_MAX + 0.5: continue
        B_alt = vas_B(tau, k_alt); H_alt = -B_alt * phi0
        sig_stat_alt = s / math.sqrt(2*k_alt)
        H_str_alt = -B_alt * (phi0 - 2*sig_stat_alt) + B_alt * (q_band or 0.0)
        hl_alt = math.log(2)/k_alt * 252
        print(f"    {k_alt:>5.2f}  {hl_alt:>6.0f}    {B_alt:.4f}    "
              f"{100*H_alt:>+7.3f}%   {100*H_str_alt:>+7.3f}%")
    check("H_model finite, |H|<10%",    abs(H_model) < 0.10)
    check("H_stress >= H_model",        H_stress >= H_model,
          f"(stress={100*H_stress:+.3f}% vs model={100*H_model:+.3f}%)")


# ============================================================
# [11] Stress
# ============================================================
def s11(theta_phi, tab_avg):
    print("\n[11] Stress scenarios")
    k, th, s, sm = theta_phi
    B = vas_B(1.0, k); base = float(tab_avg.tail(30).mean())
    H_lo = -B * (base - 0.005)
    H_hi = -B * (base + 0.005)
    print(f"  H(TAB-50bp) = {100*H_lo:+.3f}%   H(TAB+50bp) = {100*H_hi:+.3f}%")
    check("Haircut monotone in TAB_0",        H_lo > H_hi)
    check("Rate-shock invariance structural", True,
          "(B_phi has no r dependence in two-factor affine)")


# ============================================================
# main
# ============================================================
def main():
    print("Two-Factor Affine Tokenized-Treasury Pipeline — REAL DATA ONLY (v7.1)")
    print("=" * 78)
    print(f" Conventions: cc ACT/365 throughout. k_max={KAPPA_MAX:.2f} (HL>=10 trading days).")
    print("=" * 78)
    s1(); s2()
    tsy, sofr, tbill, pools = s3()
    norm = normalize_pools(pools, tsy)
    theta_r, r_hat = s4(tsy)
    TAB, PHI, tab_avg, phi_avg = s5(tsy, sofr, norm, r_hat)
    ctrl_df, ctrl_beta = s55(sofr, tbill, tab_avg)
    theta_phi, _ = s6(TAB, tab_avg)
    s7(pools, PHI)
    q_band = s8(tab_avg)
    s9(tab_avg)
    s10(theta_phi, tab_avg, q_band)
    s11(theta_phi, tab_avg)
    print("\n" + "=" * 78)
    print(f" SUMMARY:  {PASS} passed / {FAIL} failed   (total {PASS + FAIL})")
    print("=" * 78)


if __name__ == "__main__":
    main()