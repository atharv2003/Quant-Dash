"""
app.py – Main entry-point for the Option-Price Dash application.
"""

import dash
import dash_bootstrap_components as dbc

# External assets ---------------------------------------------------
EXTERNAL_STYLESHEETS = [
    dbc.themes.BOOTSTRAP,
    dbc.icons.FONT_AWESOME,
]

# Dash app ----------------------------------------------------------
app = dash.Dash(
    __name__,
    assets_folder="Design",          # <-- loads Design/style.css automatically
    external_stylesheets=EXTERNAL_STYLESHEETS,
    use_pages=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server

# Delay import so callbacks see the instantiated Dash app
from Component.header import navbar  # noqa: E402

# Layout ------------------------------------------------------------
app.layout = dbc.Container(
    [
        navbar,
        dash.page_container,
    ],
    fluid=True,
    id="root-container",
    className="gx-0 px-0",
)

# Main --------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=5050)
