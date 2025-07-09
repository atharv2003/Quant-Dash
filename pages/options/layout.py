"""
Option Pricer – page layout & callback
"""

from datetime import datetime

import dash
from dash import (
    Input, Output, State, dcc, html, callback, ctx, no_update
)
import dash_bootstrap_components as dbc

from pages.options.components import (
    input_groups,                # now includes Submit button
    get_chart,
    get_greeks_plot,
    get_vol_plot,
)
from pages.options.functions import (
    bs_formula,
    get_last_price,
    get_sigma,
    get_risk_free_rate as get_r,
)

dash.register_page(
    __name__, path="/options", title="Option Analytics", name="Option Analytics"
)

_layout = dbc.Container(
    [
        dbc.Row(dbc.Col(html.H2("Option Pricer"), className="mb-4")),
        dbc.Row(
            [
                dbc.Col(input_groups, width=3, className="border-end pe-4"),
                dbc.Col(
                    dcc.Loading(
                        [
                            html.H4(id="option_price", className="mb-3"),
                            dcc.Graph(id="stock_chart"),
                            dcc.Graph(id="greeks_chart"),
                            dcc.Graph(id="vol_chart"),
                        ],
                        type="circle",
                        color="#d9534f",
                    ),
                    width=9,
                ),
            ],
            className="gx-4",
        ),
    ],
    fluid=True,
    className="pt-4",
)

def layout():
    return _layout

# ----------------------- callback -----------------------
@callback(
    Output("option_price", "children"),
    Output("stock_chart", "figure"),
    Output("greeks_chart", "figure"),
    Output("vol_chart", "figure"),
    Input("submit", "n_clicks"),
    State("op_ticker_input", "value"),
    State("op_strike_input", "value"),
    State("op_exp_input", "date"),
    State("op_type_select", "value"),
    State("price_period_select", "value"),
    State("op_r_input", "value"),
    State("op_q_input", "value"),
    prevent_initial_call=True,
)
def price_option(_, symbol, strike, exp_date, option_type,
                price_period, r_user, q_user):
    if ctx.triggered_id != "submit":
        return no_update, no_update, no_update, no_update

    if None in (symbol, strike, exp_date, option_type):
        return "Invalid input!", no_update, no_update, no_update

    days = (datetime.strptime(exp_date, "%Y-%m-%d") - datetime.today()).days
    if days <= 0:
        return "Expiration must be future date", no_update, no_update, no_update

    try:
        S = get_last_price(symbol)
        sigma = get_sigma(symbol)
    except Exception as exc:
        return f"Market-data error: {exc}", no_update, no_update, no_update

    T = days / 365.0
    r = float(r_user) / 100.0 if r_user not in (None, "") else get_r()

    try:
        price = bs_formula(S, strike, T, sigma, option_type, r=r)
    except ValueError as exc:
        return f"Pricing error: {exc}", no_update, no_update, no_update

    text = f"{option_type.capitalize()} Price: ${price:,.4f} (σ={sigma:.2%}, r={r:.2%})"
    fig_price  = get_chart(symbol, period=price_period)
    fig_greeks = get_greeks_plot(S, T, sigma, option_type, r=r)
    fig_vol    = get_vol_plot(symbol)

    return text, fig_price, fig_greeks, fig_vol
