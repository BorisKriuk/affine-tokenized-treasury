# affine-tokenized-treasury

A two-factor affine term-structure model for tokenized U.S. Treasury products:
identification, haircuts, and a March-2026 structural break — using only real,
public data (FRED + DefiLlama).

> **Status (v7.1, May 2026):** 30 tests pass on live data.

---

## What this is

The pipeline does six things, end-to-end, from raw public APIs:

1. **Ingests** the FRED Treasury curve (1M–10Y), SOFR, 3M T-bill (DTB3), and
   on-chain APY/TVL series for tokenized Treasury products from DefiLlama.
2. **Normalizes** every yield to continuously compounded ACT/365, grosses up
   management fees, and amortizes Treasury benchmarks to product WAM.
3. **Calibrates** a CIR(r) short-rate model under Q on the short-end FRED panel
   and filters \\(\\hat r_t\\) per day.
4. **Constructs** a daily Tokenization-Adjusted Basis (TAB) per product, nets
   SOFR-level (Nagel 2016) and bill-scarcity (D'Avernas–Vandeweyer 2024)
   controls, and fits a Vasicek wedge process under P (per-product + pooled
   MLE with Kalman recursion and profile-likelihood κ CI).
5. **Runs an admission filter** (Workstream 1): products whose daily wedge is
   white noise (ACF₁ < 0.10) are rejected — administratively pegged NAVs do
   not identify a mean-reversion model.
6. **Reports** EnbPI conformal bands on TAB, CUSUM structural-break detection
   with permutation p-values, and 1-year collateral haircuts with κ-sensitivity
   and TAB ± 50 bp stress.

## Headline findings

- Of 5 tokenized products, **only USDY** clears the daily price-discovery test
  (ACF₁ = +0.977). USYC's daily spread is white noise; OUSG / TBILL are
  amortized-cost vehicles and report descriptively only.
- USDY wedge: kappa = 2.37, half-life ≈ 74 trading days, long-run mean
  ≈ +15 bp, idiosyncratic volatility ≈ 8 bp.
- **CUSUM detects a discrete regime change on 2026-03-17** (permutation
  p-value 0.002, B = 2000): wedge volatility collapses 6.5× post-break.
- Funding stress + bill scarcity explain only 7 % of TAB variance — the
  remaining 93 % is tokenization-specific.
- Model haircut at τ = 1y for USDY: 0.026 % of par; combined stress haircut
  0.099 %.

## Repository layout

| File                          | Role                                                    |
| ----------------------------- | ------------------------------------------------------- |
| `affine.py`                   | Pure math: CIR/Vasicek closed forms, MLE, EnbPI, CUSUM. |
| `pipeline.py`                 | Data ingestion + 11 pipeline stages + `main()`.         |
| `requirements.txt`            | `numpy pandas scipy requests`.                          |

## Install & run

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python pipeline.py
```

---

## ✍️ Authors

<div align="center">

**Artem Alhamov** &nbsp;·&nbsp; **Boris Kriuk**

</div>
