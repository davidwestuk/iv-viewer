# IV Surface Viewer

A split-screen Dash application for visualising implied volatility smiles alongside editable, fittable SVI parameters.

---

## Overview

The app displays two panels side by side:

- **Left** — an editable AG Grid table with one row per expiry, showing the five raw SVI parameters for that expiry. A **Fit** button re-fits the selected expiry, and individual cells can be edited by hand.
- **Right** — a Plotly volatility smile chart for the selected expiry. It updates whenever a row is selected, a parameter is edited, or the Fit button is pressed.

The smile uses **puts for strikes at or below the forward** and **calls for strikes above the forward**, stitched into a continuous curve, with the fitted SVI curve overlaid as a dotted line.

---

## Installation

```bash
pip install dash dash-ag-grid plotly pandas numpy scipy
```

---

## Running

```bash
python iv_viewer.py
```

Then open `http://127.0.0.1:8050` in your browser.

---

## Input Data

Replace the sample data call near the top of the file with your own CSV:

```python
df = pd.read_csv("your_file.csv")
```

The file must contain these columns:

| Column    | Description                            |
|-----------|----------------------------------------|
| `expiry`  | Expiry date string (e.g. `2026-06-20`) |
| `strike`  | Option strike price                    |
| `type`    | `"call"` or `"put"`                    |
| `iv_bid`  | Implied volatility bid                 |
| `iv_ask`  | Implied volatility ask                 |
| `iv_mid`  | Implied volatility mid                 |
| `forward` | Forward price for that expiry          |

> **Note on time to expiry.** Fitting is done in total-variance space, `w = IV² · T`. The sample app stores year-fractions per expiry in `EXPIRY_T`. If you load your own data, populate `EXPIRY_T` with the correct year-fraction for each expiry label (or set `T = 1.0` to fit directly in IV²-space).

---

## Features

### Interactive grid

- The grid drives the chart: **click any row** to view that expiry's smile.
- The `a`, `b`, `ρ`, `m`, `σ` cells are **editable** (shown in gold). Double-click a cell, type a value, and on commit the SVI curve redraws immediately.
- The `expiry` column is locked.
- Non-numeric or malformed edits are caught safely — the SVI curve is simply hidden rather than crashing the callback.

### Fit button

The **FIT SELECTED** button re-fits the SVI parameters for the currently selected expiry using `scipy.optimize.minimize` (L-BFGS-B) on that expiry's smile, then writes the result back into the grid row. The chart redraws from the new values.

---

## SVI Model

The app uses the **raw SVI** parameterisation of total implied variance.

### Formula

```
w(k) = a + b · [ρ(k − m) + √((k − m)² + σ²)]
```

where:

- `k = log(K / F)` — log-moneyness
- `w = IV² · T` — total implied variance
- annualised IV is recovered as `IV = √(w / T)`

### Parameters

| Param | Role                                    |
|-------|-----------------------------------------|
| `a`   | Overall variance level                  |
| `b`   | Slope of the wings                      |
| `ρ`   | Skew (correlation parameter, ∈ (−1, 1)) |
| `m`   | ATM shift                               |
| `σ`   | Smile curvature                         |

The fit minimises the sum of squared errors between the model total variance `w(k)` and the observed `iv_mid² · T` across all smile strikes for that expiry.

---

## Synthetic Data

The bundled sample data is generated **from** an SVI smile rather than an ad-hoc parabola, so it behaves like a realistic surface:

- Each expiry has its own "true" SVI parameters (`TRUE_SVI`) with a sensible **term structure** — ATM variance grows with maturity, negative skew (`ρ`) flattens as expiry lengthens, and curvature (`σ`) widens.
- IVs come from evaluating `w(k)` per strike, converting via `√(w / T)`, then adding ~0.3% multiplicative observation noise.
- The bid/ask spread widens in the wings to mimic lower liquidity away from the money.
- Forwards vary slightly by expiry (100 → 102).

### Caveat on parameter recovery

The Fit button recovers a smile that matches the data tightly (IV RMSE on the order of 0.05–0.7 vol points), but the recovered `(a, b, ρ, m, σ)` will not exactly equal the generating values. This is inherent to **raw SVI**: over a limited strike range, several parameter sets produce near-identical smiles, so the parameterisation is only weakly identifiable. The fitted *curve* is reliable even when the individual parameters drift. Approaches that improve parameter recovery include a wider strike grid, the SSVI / natural parameterisation, or a two-stage fit that pins `ρ` and `m` first.

---

## Application Structure

| Function / callback                     | Purpose                                                                                                                       |
|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `svi_w(k, a, b, rho, m, sigma)`         | Evaluates raw SVI total variance at log-moneyness `k`. Defined near the top so both the data generator and the fitter use it. |
| `generate_sample_data()`                | Builds the synthetic surface from `TRUE_SVI`, `EXPIRY_T`, and `FORWARDS`. Replace with your own data source.                  |
| `fit_svi(strikes, iv_mid, forward, T)`  | Fits the five SVI parameters in total-variance space. Returns a dict of `{a, b, rho, m, sigma}`, or `NaN` on failure.         |
| `build_svi_table()`                     | Fits every expiry at startup to populate the grid.                                                                            |
| `fit_selected(...)` — callback          | Fit button → refits the selected expiry → updates `rowData`.                                                                  |
| `update_chart(...)` — callback          | Redraws on selection change, cell edit, or `rowData` rewrite; reads parameters straight from the grid.                        |
| clientside callback                     | Selects the first grid row on load for a visual highlight.                                                                    |
