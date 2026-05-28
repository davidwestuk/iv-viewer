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
from dash import dcc, html, Input, Output
import dash_ag_grid as dag
import plotly.graph_objects as go

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
# Sample data
# ─────────────────────────────────────────────────────────────────────────────
def generate_sample_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    expiries = ["2026-06-20", "2026-07-18", "2026-09-19", "2026-12-18"]
    rows = []
    for exp in expiries:
        fwd = 100.0
        strikes = np.arange(70, 136, 5, dtype=float)
        for k in strikes:
            m = (k - fwd) / fwd
            base_iv = 0.20 + 0.15 * m**2 - 0.02 * m
            for opt in ("call", "put"):
                spread = rng.uniform(0.005, 0.015)
                mid = max(0.05, base_iv + rng.normal(0, 0.003))
                rows.append(dict(
                    expiry=exp, strike=k, type=opt,
                    iv_bid=round(mid - spread / 2, 4),
                    iv_ask=round(mid + spread / 2, 4),
                    iv_mid=round(mid, 4),
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
# Total variance w(k) = a + b*(rho*(k-m) + sqrt((k-m)^2 + sigma^2))
# ─────────────────────────────────────────────────────────────────────────────
def svi_w(k, a, b, rho, m, sigma):
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + sigma ** 2))


def fit_svi(strikes: np.ndarray, iv_mid: np.ndarray, forward: float) -> dict:
    """Fit SVI to mid IVs; returns parameter dict (or NaNs on failure)."""
    log_m = np.log(strikes / forward)
    total_var = iv_mid ** 2

    def objective(p):
        a, b, rho, m, sigma = p
        w = svi_w(log_m, a, b, rho, m, sigma)
        return np.sum((w - total_var) ** 2)

    atm_var = float(np.interp(0.0, log_m, total_var))
    x0 = [atm_var * 0.9, 0.1, -0.3, 0.0, 0.2]
    bounds = [(1e-6, None), (1e-6, 2), (-0.999, 0.999), (-1, 1), (1e-4, 2)]
    res = minimize(objective, x0, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 2000, "ftol": 1e-12})

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

        p = fit_svi(smile["strike"].values, smile["iv_mid"].values, fwd)

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
col_defs = [
    {"field": "expiry", "headerName": "EXPIRY", "width": 118, "pinned": "left"},
    {"field": "a",      "headerName": "a",      "width": 80, "type": "numericColumn"},
    {"field": "b",      "headerName": "b",      "width": 80, "type": "numericColumn"},
    {"field": "rho",    "headerName": "ρ",      "width": 80, "type": "numericColumn"},
    {"field": "m",      "headerName": "m",      "width": 80, "type": "numericColumn"},
    {"field": "sigma",  "headerName": "σ",      "width": 80, "type": "numericColumn"},
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
                        html.Div("SVI PARAMETERS",
                                 style={"fontSize": "10px", "letterSpacing": "4px", "color": TEXT_DIM}),

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
                                rowSelection="single",
                                selectedRows=[svi_rows[0]],
                                dashGridOptions={
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
                                html.Div("k = log(K/F)   w = σ²_IV",
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
# Callback
# ─────────────────────────────────────────────────────────────────────────────
@app.callback(
    Output("iv-chart",    "figure"),
    Output("chart-label", "children"),
    Input("svi-grid",     "selectedRows"),
)
def update_chart(selected_rows):
    exp = selected_rows[0]["expiry"] if selected_rows else expiry_list[0]

    subset  = df[df["expiry"] == exp].copy()
    forward = subset["forward"].iloc[0]

    puts  = subset[(subset["type"] == "put")  & (subset["strike"] <= forward)].sort_values("strike")
    calls = subset[(subset["type"] == "call") & (subset["strike"] >  forward)].sort_values("strike")
    smile = pd.concat([puts, calls]).sort_values("strike")

    # SVI curve
    svi_p = next((r for r in svi_rows if r["expiry"] == exp), None)
    k_grid = np.linspace(smile["strike"].min(), smile["strike"].max(), 300)
    log_k  = np.log(k_grid / forward)
    svi_iv = None
    if svi_p and svi_p["a"] is not None:
        w = svi_w(log_k, svi_p["a"], svi_p["b"], svi_p["rho"], svi_p["m"], svi_p["sigma"])
        svi_iv = np.sqrt(np.maximum(w, 0))

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


if __name__ == "__main__":
    app.run(debug=True)
