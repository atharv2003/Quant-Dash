"""landing.py – Home page for the Option‑Price Dash app
======================================================
Shows a hero section plus a mini price chart that the user can switch to
any ticker symbol they like.
"""

from datetime import datetime

import dash
from dash import html, dcc, Input, Output, callback, no_update
import dash_bootstrap_components as dbc

from pages.options.components import get_chart
from pages.options.functions import get_risk_free_rate

# ----------------------------------------------------------------------------
# Page registration
# ----------------------------------------------------------------------------

dash.register_page(
    __name__, path="/", title="Home – Option‑Price Analytics", name="Home"
)

# ----------------------------------------------------------------------------
# Layout builder
# ----------------------------------------------------------------------------

def layout():
    r = get_risk_free_rate()

    hero = dbc.Container(
        [
            html.H1("Option‑Price Analytics", className="display-4 fw-bold"),
            html.P(
                "Interactive Black‑Scholes analytics, volatility surfaces and "
                "Greeks visualisation – all in one place.",
                className="lead",
            ),
            html.Hr(className="my-2"),
            dbc.Button(
                "Launch Option Pricer →", href="/options", color="primary", size="lg"
            ),
        ],
        className="py-5 text-center",
    )

    # --- user‑selectable ticker -------------------------------------------
    ticker_input = dbc.Input(
        id="landing_ticker", type="text", value="AAPL", placeholder="Ticker", debounce=True,
        style={"width": "120px"}, className="me-2",
    )

    metrics = dbc.Row(
        [
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("Risk‑free (13‑week T‑Bill)"),
                        dbc.CardBody(html.H4(f"{r*100:.2f} %", className="card-title mb-0")),
                    ],
                    className="shadow-sm h-100",
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(
                            dbc.Row(
                                [
                                    dbc.Col("Price chart – 1 mo", className="pe-0"),
                                    dbc.Col(ticker_input, width="auto"),
                                ],
                                className="g-1 align-items-center",
                            )
                        ),
                        dbc.CardBody(
                            dcc.Graph(
                                id="landing_price_chart",
                                style={"height": "230px"},
                                config={"displayModeBar": False},
                            )
                        ),
                    ],
                    className="shadow-sm h-100",
                ),
                width=9,
            ),
        ],
        className="gy-4 my-4 justify-content-center",
    )

    footer = html.Footer(
        dbc.Container(
            [
                html.Hr(),
                html.Small(
                    f"© {datetime.now().year} Option‑Price Demo – Built with Dash & Yahoo Finance",
                    className="text-muted",
                ),
            ],
            className="py-2 text-center",
        )
    )

    return html.Div([hero, metrics, footer], className="page-container")

# ----------------------------------------------------------------------------
# Callback – update mini chart when user types a ticker
# ----------------------------------------------------------------------------

@callback(
    Output("landing_price_chart", "figure"),
    Input("landing_ticker", "value"),
)

def update_chart(symbol: str):
    if not symbol:
        return no_update
    try:
        return get_chart(symbol.upper(), period="1mo")
    except Exception:
        # graceful fallback if Yahoo fails or invalid ticker
        return get_chart("AAPL", period="1mo")
