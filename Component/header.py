"""header.py – Responsive Navbar
================================
Clean ASCII IDs, built‑in hourly refresh badge, dark‑mode toggle.
"""

import dash
from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from importlib import import_module

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _get_risk_free_rate() -> str:
    """Return formatted 13‑week T‑bill yield."""
    try:
        funcs = import_module("pages.options.functions")
        r = funcs.get_risk_free_rate()
        return f"Rₑ {r*100:0.2f}%"
    except Exception:  # pragma: no cover
        return "Rₑ --.-%"


# ------------------------------------------------------------------
# Layout builder
# ------------------------------------------------------------------

def build_navbar() -> dbc.Navbar:
    """Construct the Navbar component lazily."""

    risk_free_badge = dbc.Badge(_get_risk_free_rate(), id="rf-badge", color="secondary", className="ms-2")

    theme_switch = dbc.Switch(id="theme-switch", label="Dark mode", value=False, className="ms-3")

    nav_items = [
        dbc.NavLink("Home", href="/", id="nav-home", className="mx-2"),
        dbc.DropdownMenu(
            label="Projects",
            nav=True,
            in_navbar=True,
            children=[
                dbc.DropdownMenuItem("Options", href="/options", id="nav-options"),
            ],
            className="mx-2",
        ),
    ]

    navbar = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    dbc.Row(
                        [
                            dbc.Col(dbc.NavbarBrand("Quant-Dash", className="ms-2 fw-bold")),
                        ],
                        align="center",
                        className="g-0",
                    ),
                    href="/",
                    style={"textDecoration": "none"},
                ),
                dbc.Nav(nav_items, navbar=True, className="ms-4"),
                dbc.Nav([risk_free_badge, theme_switch], className="ms-auto flex-row"),
                dcc.Interval(id="interval-rf", interval=60 * 60 * 1000, n_intervals=0),
            ],
            fluid=True,
        ),
        color="dark",
        dark=True,
        sticky="top",
        id="header",
    )
    return navbar


navbar = build_navbar()

# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------

@dash.callback(Output("rf-badge", "children"), Input("interval-rf", "n_intervals"))
def _refresh_rf(_):  # pragma: no cover
    return _get_risk_free_rate()
