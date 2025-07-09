"""
Dash UI components & utilities for Option-Price app
– 2025-07-08 polished version –

* Single “Submit” button (at the bottom).
* “Chart Period” selector lives inside the main input stack.
* Risk-free and Dividend inputs auto-prefill with current market data.
* get_chart(symbol, period="3mo") supports the period kw-arg.
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from functools import lru_cache
from datetime import date
import numpy as np

from pages.options.functions import (
    BSMGreeks,                 # class alias provided in functions.py
    get_vol_matrix,
    get_risk_free_rate,
    get_dividend_yield,
)

# -------------------- constants --------------------
DEFAULT_PERIOD = "3mo"
PERIOD_OPTIONS = [
    {"label": "1 M", "value": "1mo"},
    {"label": "3 M", "value": "3mo"},
    {"label": "6 M", "value": "6mo"},
    {"label": "1 Y", "value": "1y"},
    {"label": "5 Y", "value": "5y"},
    {"label": "Max", "value": "max"},
]

# -------------------- inputs -----------------------
op_ticker_input = dbc.Input(id="op_ticker_input", placeholder="AAPL", type="text")
op_strike_input = dbc.Input(id="op_strike_input", placeholder="200", type="number", min=0)
op_exp_input    = dcc.DatePickerSingle(id="op_exp_input", date=date.today())
op_type_select  = dbc.Select(
    id="op_type_select",
    options=[{"label": "Call", "value": "call"}, {"label": "Put", "value": "put"}],
    value="call",
)

# Price-history / chart-period dropdown
op_period_select = dbc.Select(
    id="price_period_select",
    options=PERIOD_OPTIONS,
    value="1y",
)

# Advanced: risk-free & dividend – pre-filled but editable (pct)
op_r_input = dbc.Input(
    id="op_r_input",
    type="number",
    value=f"{get_risk_free_rate()*100:.2f}",
    step=0.01,
    placeholder="Risk-free %",
)

op_q_input = dbc.Input(
    id="op_q_input",
    type="number",
    value=f"{get_dividend_yield('AAPL')*100:.2f}",
    step=0.01,
    placeholder="Dividend %",
)

op_submit_btn = dbc.Button(
    "Submit",
    id="submit",
    n_clicks=0,
    color="primary",
    class_name="mt-3",
    style={"width": "100%"},
)

# ------------ historical price helper --------------
@lru_cache(maxsize=128)
def _history(symbol: str, period: str = DEFAULT_PERIOD):
    return yf.Ticker(symbol).history(period=period).reset_index()


def get_chart(symbol: str, period: str = DEFAULT_PERIOD) -> go.Figure:
    """Return a candlestick chart for *symbol* over *period*."""
    df = _history(symbol, period)
    if df.empty:
        return go.Figure(layout=dict(title="No price data"))

    fig = go.Figure(
        go.Candlestick(
            x=df["Date"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
        )
    )
    fig.update_layout(
        title=f"{symbol.upper()} Price – {period}",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
        height=400,
        margin=dict(t=40, r=20, l=60, b=40),
    )
    return fig

# -------------------- Greeks plot -------------------
def get_greeks_plot(S, T, sigma, _type="call", r=0.05, q=0.0) -> go.Figure:
    greeks = BSMGreeks
    K = np.linspace(S * 0.5, S * 1.5, 100)
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=("Δ Delta", "Γ Gamma", "ν Vega", "Θ Theta", "ρ Rho"),
        vertical_spacing=0.08, horizontal_spacing=0.10,
    )

    traces = [
        ("Δ", greeks.delta_vec(S, K, sigma, T, _type, r, q), (1, 1)),
        ("Γ", greeks.gamma_vec(S, K, sigma, T, r, q),        (1, 2)),
        ("ν", greeks.vega_vec(S, K, sigma, T, r, q),         (2, 1)),
        ("Θ", greeks.theta_vec(S, K, sigma, T, _type, r, q), (2, 2)),
        ("ρ", greeks.rho_vec(S, K, sigma, T, _type, r, q),   (3, 1)),
    ]
    for name, data, (row, col) in traces:
        fig.add_trace(go.Scatter(x=K, y=data, name=name, showlegend=False), row=row, col=col)

    fig.update_layout(height=850, template="plotly_white")
    return fig

# -------------------- IV surface --------------------
def get_vol_plot(symbol: str) -> go.Figure:
    vol_df = get_vol_matrix(symbol)  # strikes × expiries
    x_strikes = vol_df.index.values.astype(float)
    y_expiries = [d for d in vol_df.columns]
    z = vol_df.values.T

    fig = go.Figure(
        data=[go.Surface(x=x_strikes, y=list(range(len(y_expiries))), z=z, colorscale="Viridis")]
    )
    fig.update_layout(
        title="Implied Volatility Surface",
        scene=dict(
            xaxis=dict(title="Strike"),
            yaxis=dict(title="Expiry", tickvals=list(range(len(y_expiries))), ticktext=y_expiries),
            zaxis=dict(title="IV"),
            aspectratio=dict(x=1.2, y=1, z=0.6),
        ),
        template="plotly_white",
        margin=dict(l=65, r=50, b=65, t=90),
    )
    return fig

# ----------------- grouped inputs ------------------
input_groups = html.Div(
    [
        dbc.InputGroup([dbc.InputGroupText("Ticker"), op_ticker_input],   class_name="mb-2"),
        dbc.InputGroup([dbc.InputGroupText("Strike"), op_strike_input],   class_name="mb-2"),
        dbc.InputGroup([dbc.InputGroupText("Expiration"), op_exp_input], class_name="mb-2"),
        dbc.InputGroup([dbc.InputGroupText("Chart Period"), op_period_select], class_name="mb-2"),
        dbc.InputGroup([dbc.InputGroupText("Type"), op_type_select],      class_name="mb-2"),
        dbc.InputGroup([dbc.InputGroupText("Risk-free %"), op_r_input],   class_name="mb-2"),
        dbc.InputGroup([dbc.InputGroupText("Dividend %"), op_q_input],    class_name="mb-3"),
        op_submit_btn,
    ],
    style={"width": "340px"},
    className="p-2",
)
