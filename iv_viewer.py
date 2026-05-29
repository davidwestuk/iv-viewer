"""
Implied Volatility Viewer
=========================
Left  — Dash AG Grid with one row per expiry showing fitted SVI parameters.
Right — Volatility smile chart for the selected expiry (calls above fwd, puts below).

Dependencies
------------
    pip install dash dash-ag-grid plotly pandas numpy scipy
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import dash
from dash import dcc, html, Input, Output, State
import dash_ag_grid as dag
import plotly.graph_objects as go

# NOTE: dash-ag-grid has no "selectedRows" init prop. The first row is selected
# on load via the clientside callback at the bottom of this file, and the
# Python callback also falls back to the first expiry when nothing is selected.

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#080a0f"
PANEL    = "#0e1118"
PANEL2   = "#12151f"
BORDER   = "#1c2133"
ACCENT   = "#00ddb3"
PUT_CLR  = "#ff5e7a"
CALL_CLR = "#00ddb3"
TEXT     = "#b8c4d8"
TEXT_DIM = "#44506a"
FONT     = "'IBM Plex Mono', 'Fira Code', monospace"

# ─────────────────────────────────────────────────────────────────────────────
# Raw SVI total-variance smile:  w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
#   k = log(K / F),  w = sigma_IV^2 * T  (total variance)
# Defined up here so the synthetic data generator can use it too.
# ─────────────────────────────────────────────────────────────────────────────
def svi_w(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


# Year-fraction to each expiry — also used by the fitter so that fitted
# parameters live in the same total-variance space as the generating params.
EXPIRY_T = {
    "2026-06-20": 0.06,
    "2026-07-18": 0.14,
    "2026-09-19": 0.31,
    "2026-12-18": 0.55,
}


# ─────────────────────────────────────────────────────────────────────────────
# Sample data — generated from a realistic SVI smile per expiry.
#
# Each expiry has its own "true" SVI parameters with a sensible term structure:
#   * ATM variance grows with time (a increases),
#   * skew (rho) is negative and flattens with maturity,
#   * curvature (sigma) widens with maturity.
# We evaluate the smile in TOTAL variance, convert to an annualised IV per
# strike, add a little observation noise, then build a bid/ask around the mid.
# Because the data is generated from SVI, the "Fit" button recovers parameters
# close to these true values.
# ─────────────────────────────────────────────────────────────────────────────
TRUE_SVI = {
    # exp label   :  a       b      rho     m       sigma
    "2026-06-20": dict(a=0.0022, b=0.018, rho=-0.62, m=0.012, sigma=0.085),
    "2026-07-18": dict(a=0.0050, b=0.030, rho=-0.55, m=0.015, sigma=0.105),
    "2026-09-19": dict(a=0.0120, b=0.052, rho=-0.48, m=0.020, sigma=0.130),
    "2026-12-18": dict(a=0.0230, b=0.078, rho=-0.42, m=0.028, sigma=0.160),
}
FORWARDS = {
    "2026-06-20": 100.0,
    "2026-07-18": 100.5,
    "2026-09-19": 101.0,
    "2026-12-18": 102.0,
}


def generate_sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for exp, p in TRUE_SVI.items():
        T   = EXPIRY_T[exp]
        fwd = FORWARDS[exp]
        strikes = np.arange(70, 136, 5, dtype=float)
        for K in strikes:
            k = np.log(K / fwd)                     # log-moneyness
            w = max(svi_w(k, **p), 1e-6)            # total variance
            iv_true = np.sqrt(w / T)                # annualised IV

            # small multiplicative observation noise (~0.3% of IV)
            iv_obs = max(iv_true * (1.0 + rng.normal(0, 0.003)), 0.01)

            # bid/ask spread widens in the wings (lower liquidity there)
            half_spread = 0.0025 + 0.01 * abs(k) + rng.uniform(0, 0.0015)

            for opt in ("call", "put"):
                rows.append(dict(
                    expiry=exp, strike=K, type=opt,
                    iv_bid=round(iv_obs - half_spread, 4),
                    iv_ask=round(iv_obs + half_spread, 4),
                    iv_mid=round(iv_obs, 4),
                    forward=fwd,
                ))
    return pd.DataFrame(rows)


# Replace with: df = pd.read_csv("your_file.csv")
df = generate_sample_data()

required = {"expiry", "strike", "type", "iv_bid", "iv_ask", "iv_mid", "forward"}
assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

expiry_list = sorted(df["expiry"].unique())

# ─────────────────────────────────────────────────────────────────────────────
# SVI fit  (raw parameterisation: a, b, rho, m, sigma)
# ─────────────────────────────────────────────────────────────────────────────
def fit_svi(strikes: np.ndarray, iv_mid: np.ndarray, forward: float,
            T: float = 1.0) -> dict:
    """Fit SVI to mid IVs in total-variance space (w = iv^2 * T)."""
    log_m = np.log(strikes / forward)
    total_var = (iv_mid ** 2) * T

    def objective(p):
        a, b, rho, m, sigma = p
        w = svi_w(log_m, a, b, rho, m, sigma)
        return np.sum((w - total_var) ** 2)

    atm_var = float(np.interp(0.0, log_m, total_var))
    x0 = [atm_var * 0.9, max(atm_var, 0.01), -0.3, 0.0, 0.1]
    bounds = [(1e-8, None), (1e-8, 5), (-0.999, 0.999), (-1, 1), (1e-4, 2)]
    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-14})

    if res.success or res.fun < 1e-6:
        a, b, rho, m, sigma = res.x
        return dict(a=a, b=b, rho=rho, m=m, sigma=sigma)
    return dict(a=np.nan, b=np.nan, rho=np.nan, m=np.nan, sigma=np.nan)


def build_svi_table() -> list:
    rows = []
    for exp in expiry_list:
        sub = df[df["expiry"] == exp]
        fwd = sub["forward"].iloc[0]
        puts  = sub[(sub["type"] == "put")  & (sub["strike"] <= fwd)]
        calls = sub[(sub["type"] == "call") & (sub["strike"] >  fwd)]
        smile = pd.concat([puts, calls]).sort_values("strike")

        p = fit_svi(smile["strike"].values, smile["iv_mid"].values, fwd,
                    T=EXPIRY_T.get(exp, 1.0))

        def fmt(v, d=4):
            return round(float(v), d) if not np.isnan(v) else None

        rows.append(dict(
            expiry=exp,
            a=fmt(p["a"]),
            b=fmt(p["b"]),
            rho=fmt(p["rho"], 3),
            m=fmt(p["m"], 4),
            sigma=fmt(p["sigma"], 4),
        ))
    return rows


svi_rows = build_svi_table()

# ─────────────────────────────────────────────────────────────────────────────
# AG Grid column definitions
# ─────────────────────────────────────────────────────────────────────────────
_num_col = {
    "type": "numericColumn",
    "editable": True,
    "width": 80,
    "valueParser": {"function": "Number(params.newValue)"},
    "cellStyle": {
        "fontFamily": FONT,
        "fontSize": "12px",
        "color": "#f0d080",
        "backgroundColor": "transparent",
        "borderColor": BORDER,
        "paddingLeft": "12px",
    },
}

col_defs = [
    {"field": "expiry", "headerName": "EXPIRY", "width": 118, "pinned": "left", "editable": False},
    {"field": "a",      "headerName": "a",      **_num_col},
    {"field": "b",      "headerName": "b",      **_num_col},
    {"field": "rho",    "headerName": "ρ",      **_num_col},
    {"field": "m",      "headerName": "m",      **_num_col},
    {"field": "sigma",  "headerName": "σ",      **_num_col},
]

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "IV Surface Viewer"

app.layout = html.Div(
    style={
        "backgroundColor": BG,
        "minHeight": "100vh",
        "fontFamily": FONT,
        "color": TEXT,
        "display": "flex",
        "flexDirection": "column",
        "padding": "28px 32px",
        "boxSizing": "border-box",
    },
    children=[

        # ── Header ────────────────────────────────────────────────────────────
        html.Div(
            style={
                "display": "flex", "alignItems": "baseline", "gap": "14px",
                "marginBottom": "24px", "borderBottom": f"1px solid {BORDER}",
                "paddingBottom": "16px",
            },
            children=[
                html.Span("IV", style={"fontSize": "26px", "fontWeight": "700",
                                       "color": ACCENT, "letterSpacing": "-1px"}),
                html.Span("SURFACE VIEWER",
                          style={"fontSize": "12px", "letterSpacing": "6px", "color": TEXT_DIM}),
                html.Span("·", style={"color": BORDER}),
                html.Span("SVI raw parameterisation",
                          style={"fontSize": "11px", "color": TEXT_DIM}),
            ],
        ),

        # ── Two-column body ───────────────────────────────────────────────────
        html.Div(
            style={"display": "flex", "flex": "1", "gap": "20px", "alignItems": "stretch"},
            children=[

                # ── Left: SVI grid ────────────────────────────────────────────
                html.Div(
                    style={"width": "510px", "flexShrink": "0", "display": "flex",
                           "flexDirection": "column", "gap": "12px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center",
                                   "justifyContent": "space-between"},
                            children=[
                                html.Div("SVI PARAMETERS",
                                         style={"fontSize": "10px", "letterSpacing": "4px",
                                                "color": TEXT_DIM}),
                                html.Button(
                                    "FIT SELECTED",
                                    id="fit-btn",
                                    n_clicks=0,
                                    style={
                                        "fontFamily": FONT,
                                        "fontSize": "10px",
                                        "letterSpacing": "2px",
                                        "color": BG,
                                        "backgroundColor": ACCENT,
                                        "border": "none",
                                        "borderRadius": "4px",
                                        "padding": "6px 14px",
                                        "cursor": "pointer",
                                        "fontWeight": "700",
                                    },
                                ),
                            ],
                        ),

                        # Grid wrapper — CSS vars propagate into ag-grid shadow DOM
                        html.Div(
                            dag.AgGrid(
                                id="svi-grid",
                                rowData=svi_rows,
                                columnDefs=col_defs,
                                defaultColDef={
                                    "resizable": True,
                                    "sortable": True,
                                    "cellStyle": {
                                        "fontFamily": FONT,
                                        "fontSize": "12px",
                                        "color": TEXT,
                                        "backgroundColor": "transparent",
                                        "borderColor": BORDER,
                                        "paddingLeft": "12px",
                                    },
                                },
                                getRowId="params.data.expiry",
                                dashGridOptions={
                                    "rowSelection": "single",
                                    "rowHeight": 40,
                                    "headerHeight": 36,
                                    "suppressCellFocus": True,
                                    "animateRows": True,
                                },
                                style={"height": f"{36 + 40 * len(svi_rows) + 2}px"},
                                className="ag-theme-alpine-dark",
                            ),
                            style={
                                "border": f"1px solid {BORDER}",
                                "borderRadius": "6px",
                                "overflow": "hidden",
                                "--ag-background-color": PANEL,
                                "--ag-odd-row-background-color": PANEL2,
                                "--ag-header-background-color": PANEL2,
                                "--ag-header-foreground-color": ACCENT,
                                "--ag-border-color": BORDER,
                                "--ag-row-border-color": BORDER,
                                "--ag-selected-row-background-color": "#172038",
                                "--ag-font-family": FONT,
                                "--ag-font-size": "12px",
                                "--ag-foreground-color": TEXT,
                            },
                        ),

                        # Formula card
                        html.Div(
                            style={
                                "backgroundColor": PANEL2,
                                "border": f"1px solid {BORDER}",
                                "borderRadius": "6px",
                                "padding": "16px 18px",
                                "fontSize": "11px",
                                "color": TEXT_DIM,
                                "lineHeight": "2.0",
                            },
                            children=[
                                html.Div("RAW SVI",
                                         style={"color": ACCENT, "letterSpacing": "3px",
                                                "fontSize": "10px", "marginBottom": "8px"}),
                                html.Div("w(k) = a + b · [ρ(k−m) + √((k−m)² + σ²)]"),
                                html.Div("k = log(K/F)   w = σ²_IV · T",
                                         style={"fontSize": "10px", "marginTop": "2px"}),
                                html.Hr(style={"border": f"1px solid {BORDER}", "margin": "10px 0"}),
                                html.Div([
                                    html.Span("a", style={"color": TEXT}),
                                    html.Span("  level   "),
                                    html.Span("b", style={"color": TEXT}),
                                    html.Span("  wings slope   "),
                                    html.Span("ρ", style={"color": TEXT}),
                                    html.Span("  skew"),
                                ]),
                                html.Div([
                                    html.Span("m", style={"color": TEXT}),
                                    html.Span("  ATM shift   "),
                                    html.Span("σ", style={"color": TEXT}),
                                    html.Span("  smile curvature"),
                                ]),
                            ],
                        ),
                    ],
                ),

                # ── Right: smile chart ────────────────────────────────────────
                html.Div(
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "gap": "10px"},
                    children=[
                        html.Div(id="chart-label",
                                 style={"fontSize": "10px", "letterSpacing": "4px", "color": TEXT_DIM,
                                        "minHeight": "14px"}),
                        html.Div(
                            dcc.Graph(id="iv-chart", config={"displayModeBar": False},
                                      style={"height": "100%"}),
                            style={
                                "flex": "1",
                                "backgroundColor": PANEL,
                                "border": f"1px solid {BORDER}",
                                "borderRadius": "6px",
                                "overflow": "hidden",
                                "minHeight": "420px",
                            },
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "22px",
                                   "fontSize": "11px", "color": TEXT_DIM},
                            children=[
                                html.Span([html.Span("●", style={"color": PUT_CLR,  "marginRight": "6px"}),
                                           "Puts (K ≤ fwd)"]),
                                html.Span([html.Span("●", style={"color": CALL_CLR, "marginRight": "6px"}),
                                           "Calls (K > fwd)"]),
                                html.Span([html.Span("· · ·", style={"color": "#f0d080", "marginRight": "6px"}),
                                           "SVI fit"]),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _smile_for(exp):
    """Return (forward, puts, calls, smile) for an expiry."""
    subset  = df[df["expiry"] == exp].copy()
    forward = subset["forward"].iloc[0]
    puts  = subset[(subset["type"] == "put")  & (subset["strike"] <= forward)].sort_values("strike")
    calls = subset[(subset["type"] == "call") & (subset["strike"] >  forward)].sort_values("strike")
    smile = pd.concat([puts, calls]).sort_values("strike")
    return forward, puts, calls, smile


def _selected_expiry(selected_rows):
    return selected_rows[0]["expiry"] if selected_rows else expiry_list[0]


# ─────────────────────────────────────────────────────────────────────────────
# Fit button → refit selected expiry, write params back into the grid.
# Returns updated rowData; the chart callback then redraws from rowData.
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("svi-grid", "rowData"),
    Input("fit-btn", "n_clicks"),
    State("svi-grid", "rowData"),
    State("svi-grid", "selectedRows"),
    prevent_initial_call=True,
)
def fit_selected(_n_clicks, row_data, selected_rows):
    exp = _selected_expiry(selected_rows)
    forward, _, _, smile = _smile_for(exp)
    p = fit_svi(smile["strike"].values, smile["iv_mid"].values, forward,
                T=EXPIRY_T.get(exp, 1.0))

    def fmt(v, d=4):
        return round(float(v), d) if not np.isnan(v) else None

    for row in row_data:
        if row["expiry"] == exp:
            row["a"]     = fmt(p["a"])
            row["b"]     = fmt(p["b"])
            row["rho"]   = fmt(p["rho"], 3)
            row["m"]     = fmt(p["m"], 4)
            row["sigma"] = fmt(p["sigma"], 4)
            break
    return row_data


# ─────────────────────────────────────────────────────────────────────────────
# Chart callback — redraws whenever the selection changes OR a cell is edited
# OR the Fit button rewrites rowData. Parameters are read from the grid so that
# manual edits are reflected immediately.
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("iv-chart",    "figure"),
    Output("chart-label", "children"),
    Input("svi-grid",     "selectedRows"),
    Input("svi-grid",     "cellValueChanged"),
    Input("svi-grid",     "rowData"),
)
def update_chart(selected_rows, _cell_changed, row_data):
    exp = _selected_expiry(selected_rows)
    forward, puts, calls, smile = _smile_for(exp)

    # Read SVI params from the (possibly user-edited) grid row
    svi_p = next((r for r in (row_data or []) if r["expiry"] == exp), None)
    T = EXPIRY_T.get(exp, 1.0)
    k_grid = np.linspace(smile["strike"].min(), smile["strike"].max(), 300)
    log_k  = np.log(k_grid / forward)
    svi_iv = None
    if svi_p and svi_p.get("a") is not None:
        try:
            w = svi_w(log_k, float(svi_p["a"]), float(svi_p["b"]),
                      float(svi_p["rho"]), float(svi_p["m"]), float(svi_p["sigma"]))
            svi_iv = np.sqrt(np.maximum(w, 0) / T)   # total variance → annualised IV
        except (TypeError, ValueError):
            svi_iv = None

    fig = go.Figure()

    # bid-ask bands
    for grp, clr in [(puts, PUT_CLR), (calls, CALL_CLR)]:
        if grp.empty:
            continue
        fig.add_trace(go.Scatter(
            x=pd.concat([grp["strike"], grp["strike"].iloc[::-1]]),
            y=pd.concat([grp["iv_ask"], grp["iv_bid"].iloc[::-1]]),
            fill="toself", fillcolor=f"{clr}1a",
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip", showlegend=False,
        ))

    # put mids
    fig.add_trace(go.Scatter(
        x=puts["strike"], y=puts["iv_mid"],
        mode="lines+markers",
        line=dict(color=PUT_CLR, width=2),
        marker=dict(size=6, color=PUT_CLR, line=dict(color=BG, width=1)),
        hovertemplate="<b>Put</b>  K=%{x}  IV=%{y:.2%}<extra></extra>",
    ))

    # call mids
    fig.add_trace(go.Scatter(
        x=calls["strike"], y=calls["iv_mid"],
        mode="lines+markers",
        line=dict(color=CALL_CLR, width=2),
        marker=dict(size=6, color=CALL_CLR, line=dict(color=BG, width=1)),
        hovertemplate="<b>Call</b>  K=%{x}  IV=%{y:.2%}<extra></extra>",
    ))

    # SVI fit
    if svi_iv is not None:
        fig.add_trace(go.Scatter(
            x=k_grid, y=svi_iv,
            mode="lines",
            line=dict(color="#f0d080", width=1.5, dash="dot"),
            hovertemplate="SVI  K=%{x:.1f}  IV=%{y:.2%}<extra></extra>",
        ))

    # forward line
    fig.add_vline(
        x=forward, line_width=1, line_dash="dot", line_color=BORDER,
        annotation_text=f"F {forward:.2f}",
        annotation_font=dict(color=TEXT_DIM, size=10, family=FONT),
        annotation_position="top right",
    )

    fig.update_layout(
        plot_bgcolor=PANEL,
        paper_bgcolor=PANEL,
        font=dict(family=FONT, color=TEXT, size=11),
        margin=dict(l=58, r=24, t=28, b=52),
        xaxis=dict(
            title=dict(text="Strike", font=dict(size=11, color=TEXT_DIM)),
            gridcolor=BORDER, zerolinecolor=BORDER, tickfont=dict(size=10),
        ),
        yaxis=dict(
            title=dict(text="Implied Volatility", font=dict(size=11, color=TEXT_DIM)),
            gridcolor=BORDER, zerolinecolor=BORDER,
            tickfont=dict(size=10), tickformat=".0%",
        ),
        hoverlabel=dict(bgcolor=PANEL2, bordercolor=BORDER,
                        font=dict(family=FONT, size=11, color=TEXT)),
        showlegend=False,
    )

    label = f"SMILE  ·  {exp}  ·  FWD {forward:.4f}"
    return fig, label


# ─────────────────────────────────────────────────────────────────────────────
# Select the first grid row on initial load (visual highlight).
# The grid's "selectedRows" is read-only as a prop, so we set it via the
# grid API from the client once the grid has rendered.
# ─────────────────────────────────────────────────────────────────────────────
app.clientside_callback(
    """
    function(rowData) {
        if (!rowData || rowData.length === 0) {
            return window.dash_clientside.no_update;
        }
        // Defer until the grid API is available, then select the first node.
        const trySelect = function(attempt) {
            const gridDiv = document.querySelector('#svi-grid');
            const api = gridDiv && gridDiv.gridApi;
            if (api) {
                const node = api.getDisplayedRowAtIndex(0);
                if (node) { node.setSelected(true); }
            } else if (attempt < 20) {
                setTimeout(function() { trySelect(attempt + 1); }, 100);
            }
        };
        trySelect(0);
        return window.dash_clientside.no_update;
    }
    """,
    Output("svi-grid", "id"),
    Input("svi-grid", "rowData"),
)


if __name__ == "__main__":
    app.run(debug=True)
