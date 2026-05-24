#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
affine.py — pure mathematical kernels for the two-factor affine
tokenized-Treasury pipeline. No I/O, no globals beyond constants.

Contents
--------
* Convention helpers       :  to_cc_act365
* CIR(r) closed forms      :  cir_AB, cir_yield, per_day_r
* Vasicek(phi) closed form :  vas_B
* Vasicek Kalman MLE       :  vas_loglik_one, vas_loglik_pool,
                              fit_one, fit_pooled, profile_kappa_pool
* EnbPI conformal band     :  enbpi_band
* CUSUM structural break   :  cusum_stat, cusum_permutation_pvalue
* Constants                :  KAPPA_MAX
"""
import math
import numpy as np
from scipy.optimize import minimize

# kappa_max -> half-life >= 10 trading days
KAPPA_MAX = math.log(2) / (10.0 / 252.0)   # ~17.46


# ============================================================
# Convention normalization
# ============================================================
def to_cc_act365(rate, convention):
    """Map any quoted rate to annual continuously compounded ACT/365."""
    if convention == "continuous_act365":
        return rate
    if convention == "simple_act365":
        return np.log1p(rate)
    if convention == "simple_act360":
        return np.log1p(rate * 360.0 / 365.0)
    if convention == "bond_equiv_act365":
        return 2.0 * np.log1p(rate / 2.0)
    raise ValueError(f"unknown convention: {convention}")


# ============================================================
# CIR closed forms
# ============================================================
def cir_AB(tau, k, th, s):
    g = math.sqrt(k * k + 2 * s * s)
    e = math.exp(g * tau)
    den = (g + k) * (e - 1) + 2 * g
    B = 2 * (e - 1) / den
    A = (2 * g * math.exp((k + g) * tau / 2) / den) ** (2 * k * th / (s * s))
    return A, B


def cir_yield(tau, r, k, th, s):
    if tau < 1e-10:
        return r
    A, B = cir_AB(tau, k, th, s)
    return (B * r - math.log(A)) / tau


def per_day_r(theta, tau, Y):
    """Closed-form per-day OLS recovery of r_t given (k,th,s) and a curve panel."""
    k, th, s = theta
    a = np.zeros(len(tau))
    b = np.zeros(len(tau))
    for i, T in enumerate(tau):
        A, B = cir_AB(T, k, th, s)
        a[i] = -math.log(A) / T
        b[i] = B / T
    r = ((Y - a) @ b) / (b @ b)
    return r, a, b


# ============================================================
# Vasicek loading
# ============================================================
def vas_B(tau, k):
    if k < 1e-8:
        return tau
    return (1.0 - math.exp(-k * tau)) / k


# ============================================================
# Vasicek Kalman-filter log-likelihoods
# ============================================================
def vas_loglik_one(params, x, dt, k_max):
    k, th, s, sm = params
    if not (1e-3 < k  < k_max):  return 1e9
    if not (-0.10 < th < 0.10):  return 1e9
    if not (1e-5 < s  < 0.5):    return 1e9
    if not (1e-6 < sm < 0.1):    return 1e9
    a = math.exp(-k * dt)
    b = th * (1.0 - a)
    Q = s * s * (1.0 - a * a) / (2.0 * k)
    R = sm * sm
    z = th
    P = s * s / (2.0 * k)
    ll = 0.0
    for xt in x:
        zp = a * z + b
        Pp = a * a * P + Q
        nu = xt - zp
        S = Pp + R
        if S <= 0:
            return 1e9
        ll += 0.5 * (math.log(2 * math.pi * S) + nu * nu / S)
        K = Pp / S
        z = zp + K * nu
        P = (1.0 - K) * Pp
    return ll


def vas_loglik_pool(params, series_list, dt, k_max):
    k, s = params[0], params[1]
    if not (1e-3 < k < k_max): return 1e9
    if not (1e-5 < s < 0.5):   return 1e9
    a_ = math.exp(-k * dt)
    Q = s * s * (1.0 - a_ * a_) / (2.0 * k)
    P0 = s * s / (2.0 * k)
    total = 0.0
    for i, x in enumerate(series_list):
        th = params[2 + 2 * i]
        sm = params[2 + 2 * i + 1]
        if not (-0.10 < th < 0.10): return 1e9
        if not (1e-6 < sm < 0.1):   return 1e9
        b = th * (1.0 - a_)
        R = sm * sm
        z = th
        P = P0
        for xt in x:
            zp = a_ * z + b
            Pp = a_ * a_ * P + Q
            nu = xt - zp
            S = Pp + R
            if S <= 0:
                return 1e9
            total += 0.5 * (math.log(2 * math.pi * S) + nu * nu / S)
            K = Pp / S
            z = zp + K * nu
            P = (1.0 - K) * Pp
    return total


def fit_one(x, dt, k_max):
    starts = [(0.5, x.mean(), max(x.std(), 1e-3), 1e-3),
              (2.0, x.mean(), 0.01, 5e-4),
              (0.1, x.mean(), 0.005, 2e-3),
              (5.0, 0.0,      0.01, 1e-3)]
    best = None
    for x0 in starts:
        try:
            r = minimize(lambda p: vas_loglik_one(p, x, dt, k_max),
                         x0, method="Nelder-Mead",
                         options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 4000})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    return best


def fit_pooled(series_list, dt, k_max):
    starts = []
    for k0 in [0.3, 1.0, 3.0, 8.0]:
        for s0 in [0.005, 0.02]:
            x0 = [k0, s0]
            for x in series_list:
                x0.extend([float(np.mean(x)), max(np.std(x) * 0.5, 1e-3)])
            starts.append(x0)
    best = None
    for x0 in starts:
        try:
            r = minimize(lambda p: vas_loglik_pool(p, series_list, dt, k_max),
                         x0, method="Nelder-Mead",
                         options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 15000})
            if best is None or r.fun < best.fun:
                best = r
        except Exception:
            pass
    return best


def profile_kappa_pool(series_list, dt, k_grid, k_max):
    out = []
    for k_fix in k_grid:
        def f(p):
            full = [k_fix] + list(p)
            return vas_loglik_pool(full, series_list, dt, k_max + 5.0)
        starts = []
        for s0 in [0.005, 0.02]:
            x0 = [s0]
            for x in series_list:
                x0.extend([float(np.mean(x)), max(np.std(x) * 0.5, 1e-3)])
            starts.append(x0)
        best = None
        for x0 in starts:
            try:
                r = minimize(f, x0, method="Nelder-Mead",
                             options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 8000})
                if best is None or r.fun < best.fun:
                    best = r
            except Exception:
                pass
        out.append((k_fix, best.fun if best else np.inf))
    return out


# ============================================================
# EnbPI conformal band on a univariate series (AR(1) base learner)
# ============================================================
def enbpi_band(x, n_bootstrap=50, block=None, alpha=0.10, rng=None):
    """Block-bootstrap EnbPI half-width for a univariate series.

    Returns
    -------
    yhat     : ndarray  ensemble point predictions
    q        : float    half-width at level (1-alpha)
    coverage : float    empirical fraction within +/- q
    """
    if rng is None:
        rng = np.random.default_rng(7)
    x = np.asarray(x, dtype=float)
    n = len(x)
    if block is None:
        block = max(10, n // 20)
    preds = np.zeros((n_bootstrap, n))
    for bi in range(n_bootstrap):
        idx = []
        while len(idx) < n:
            start = int(rng.integers(0, n - block))
            idx.extend(range(start, start + block))
        idx = np.array(idx[:n])
        xb = x[idx]
        A = np.column_stack([np.ones(len(xb) - 1), xb[:-1]])
        c, *_ = np.linalg.lstsq(A, xb[1:], rcond=None)
        preds[bi, 0] = x[0]
        for t in range(1, n):
            preds[bi, t] = c[0] + c[1] * x[t - 1]
    yhat = preds.mean(axis=0)
    resid = np.abs(x - yhat)
    q = float(np.quantile(resid, 1.0 - alpha))
    coverage = float(np.mean(np.abs(x - yhat) <= q))
    return yhat, q, coverage


# ============================================================
# CUSUM structural break
# ============================================================
def cusum_stat(x):
    """CUSUM statistic and argmax index."""
    x = np.asarray(x, dtype=float)
    z = (x - x.mean()) / (x.std() + 1e-12)
    c = np.cumsum(z) / math.sqrt(len(x))
    stat = float(np.max(np.abs(c)))
    arg = int(np.argmax(np.abs(c)))
    return stat, arg


def cusum_permutation_pvalue(x, B=2000, rng=None):
    """Two-sided permutation p-value for the CUSUM statistic."""
    if rng is None:
        rng = np.random.default_rng(7)
    x = np.asarray(x, dtype=float)
    stat, _ = cusum_stat(x)
    null = np.empty(B)
    for i in range(B):
        xp = rng.permutation(x)
        zp = (xp - xp.mean()) / (xp.std() + 1e-12)
        null[i] = np.max(np.abs(np.cumsum(zp) / math.sqrt(len(xp))))
    return float(np.mean(null >= stat)), stat