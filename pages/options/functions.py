"""functions.py – Quantitative routines for Option‑Price Dash app
================================================================
Vectorised Black‑Scholes‑Merton analytics, market‑data helpers, and
implied–volatility surface builder. Designed to work hand‑in‑hand with
`components.py` (UI layer).

This revision fixes integration bugs discovered during live testing:
-------------------------------------------------------------------
* **Ticker vs. symbol** – helpers now accept either a raw symbol *or* a
  pre‑built ``yfinance.Ticker`` instance.
* **`get_r()` alias** – added for legacy imports (`layout.py` &
  `header.py`).
* **Stricter docstrings + typing** – to silence IDE warnings.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Union

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# -----------------------------------------------------------------------------
# Type aliases
# -----------------------------------------------------------------------------
OptionType = Literal["call", "put"]
TickerLike = Union[str, yf.Ticker]

# -----------------------------------------------------------------------------
# Utility: normalised symbol access
# -----------------------------------------------------------------------------

def _to_symbol(obj: TickerLike) -> str:
    """Return the ticker *symbol* regardless of whether *obj* is a string or
    a ``yfinance.Ticker`` instance."""
    if hasattr(obj, "ticker"):
        return obj.ticker  # type: ignore[arg-type]
    return str(obj)

# -----------------------------------------------------------------------------
# Black‑Scholes‑Merton internals (vectorised)
# -----------------------------------------------------------------------------

def _d1(S, K, T, sigma, r=0.0, q=0.0):
    S, K, T, sigma, r, q = map(np.asarray, (S, K, T, sigma, r, q))
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (
        sigma * np.sqrt(T)
    )


def _d2(S, K, T, sigma, r=0.0, q=0.0):
    return _d1(S, K, T, sigma, r, q) - sigma * np.sqrt(T)

# -----------------------------------------------------------------------------
# Public pricing & Greeks interface
# -----------------------------------------------------------------------------

def bs_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    sigma: float | np.ndarray,
    option: OptionType = "call",
    r: float | np.ndarray = 0.0,
    q: float | np.ndarray = 0.0,
):
    """Black‑Scholes‑Merton option price (vectorised)."""
    d1 = _d1(S, K, T, sigma, r, q)
    d2 = d1 - sigma * np.sqrt(T)

    if option == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    if option == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    raise ValueError("option must be 'call' or 'put'")


class BSMGreeks:
    """
    Vectorised Greeks for the Black-Scholes-Merton model.

    Scalar methods (`delta`, `gamma`, …) accept floats.
    Vector helpers (`delta_vec`, `gamma_vec`, …) accept an
    *array-like* of strikes and return a NumPy array – perfect
    for plotting curves.
    """

    # ---------- scalar Greeks ---------- #
    @staticmethod
    def delta(S, K, sigma, T, option: OptionType = "call", r=0.0, q=0.0):
        d1 = _d1(S, K, T, sigma, r, q)
        if option == "call":
            return np.exp(-q * T) * norm.cdf(d1)
        return np.exp(-q * T) * (norm.cdf(d1) - 1)

    @staticmethod
    def gamma(S, K, sigma, T, r=0.0, q=0.0):
        d1 = _d1(S, K, T, sigma, r, q)
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S, K, sigma, T, r=0.0, q=0.0):
        d1 = _d1(S, K, T, sigma, r, q)
        return 0.01 * S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    @staticmethod
    def theta(S, K, sigma, T, option: OptionType = "call", r=0.0, q=0.0):
        d1 = _d1(S, K, T, sigma, r, q)
        d2 = d1 - sigma * np.sqrt(T)
        term1 = -S * norm.pdf(d1) * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
        if option == "call":
            term2 =  q * S * np.exp(-q * T) * norm.cdf(d1)
            term3 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = -q * S * np.exp(-q * T) * norm.cdf(-d1)
            term3 =  r * K * np.exp(-r * T) * norm.cdf(-d2)
        return (term1 + term2 + term3) / 365.0

    @staticmethod
    def rho(S, K, sigma, T, option: OptionType = "call", r=0.0, q=0.0):
        d2 = _d2(S, K, T, sigma, r, q)
        factor = 0.01 * K * T * np.exp(-r * T)
        return factor * norm.cdf(d2 if option == "call" else -d2)

    # ---------- vector helpers ---------- #
    @classmethod
    def delta_vec(cls, S, K_array, sigma, T, option="call", r=0.0, q=0.0):
        f = np.vectorize(lambda k: cls.delta(S, k, sigma, T, option, r, q))
        return f(K_array)

    @classmethod
    def gamma_vec(cls, S, K_array, sigma, T, r=0.0, q=0.0):
        f = np.vectorize(lambda k: cls.gamma(S, k, sigma, T, r, q))
        return f(K_array)

    @classmethod
    def vega_vec(cls, S, K_array, sigma, T, r=0.0, q=0.0):
        f = np.vectorize(lambda k: cls.vega(S, k, sigma, T, r, q))
        return f(K_array)

    @classmethod
    def theta_vec(cls, S, K_array, sigma, T, option="call", r=0.0, q=0.0):
        f = np.vectorize(lambda k: cls.theta(S, k, sigma, T, option, r, q))
        return f(K_array)

    @classmethod
    def rho_vec(cls, S, K_array, sigma, T, option="call", r=0.0, q=0.0):
        f = np.vectorize(lambda k: cls.rho(S, k, sigma, T, option, r, q))
        return f(K_array)


# Backwards-compat alias for legacy imports
BSM_Greeks = BSMGreeks


# -----------------------------------------------------------------------------
# Market‑data helpers (Yahoo Finance)
# -----------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _yf_ticker(symbol: str) -> yf.Ticker:  # pragma: no cover
    return yf.Ticker(symbol)


def _ticker(symbol_or_tk: TickerLike) -> yf.Ticker:
    """Return a cached ``yf.Ticker`` for *symbol_or_tk*."""
    if isinstance(symbol_or_tk, yf.Ticker):
        return symbol_or_tk
    return _yf_ticker(str(symbol_or_tk))


@lru_cache(maxsize=256)
def get_last_price(symbol_or_tk: TickerLike) -> float:  # pragma: no cover
    data = _ticker(symbol_or_tk).history(period="1d")
    return float(data["Close"].iloc[-1])


@lru_cache(maxsize=256)
def get_sigma(symbol_or_tk: TickerLike, lookback: str = "6mo") -> float:  # pragma: no cover
    data = _ticker(symbol_or_tk).history(period=lookback)
    log_r = np.log(data["Close"]).diff().dropna()
    return float(log_r.std() * np.sqrt(252))


@lru_cache(maxsize=16)
def get_risk_free_rate() -> float:  # pragma: no cover
    try:
        data = yf.Ticker("^IRX").history(period="5d")
        return float(data["Close"].iloc[-1] / 100.0)
    except Exception:
        return 0.04

# Legacy alias for back‑compat

def get_r() -> float:  # pragma: no cover
    """Alias retained for older imports."""
    return get_risk_free_rate()


@lru_cache(maxsize=256)
def get_dividend_yield(symbol_or_tk: TickerLike) -> float:  # pragma: no cover
    info = _ticker(symbol_or_tk).info
    return float(info.get("dividendYield") or 0.0)

# -----------------------------------------------------------------------------
# Implied‑volatility surface utilities
# -----------------------------------------------------------------------------

@lru_cache(maxsize=64)
def get_vol_surface(symbol: str, option: OptionType = "call") -> pd.DataFrame:  # pragma: no cover
    tk = _ticker(symbol)
    surface: dict[str, pd.Series] = {}
    for expiry in tk.options:
        chain = tk.option_chain(expiry)
        df = chain.calls if option == "call" else chain.puts
        surface[expiry] = df.set_index("strike")["impliedVolatility"]

    iv_df = pd.DataFrame(surface).sort_index()
    iv_df = (
        iv_df.interpolate(axis=0, method="linear")
        .fillna(method="bfill")
        .fillna(method="ffill")
    )

    last = get_last_price(symbol)
    return iv_df.query(f"{last*0.5} <= strike <= {last*1.5}")


def get_vol_matrix(ticker_or_symbol, option: OptionType = "call") -> pd.DataFrame:
    symbol = _to_symbol(ticker_or_symbol)
    return get_vol_surface(symbol, option)


# -----------------------------------------------------------------------------
# Thin wrapper matching previous API
# -----------------------------------------------------------------------------

def bs_formula(
    S: float,
    K: float,
    T: float,
    sigma: float,
    _type: str = "call",
    r: float | None = None,
    q: float = 0.0,
) -> float:
    if r is None:
        r = get_risk_free_rate()
    return float(bs_price(S, K, T, sigma, option=_type, r=r, q=q))


__all__ = [
    "bs_price",
    "bs_formula",
    "BSMGreeks",
    "get_last_price",
    "get_sigma",
    "get_vol_matrix",
    "get_risk_free_rate",
    "get_r",
    "get_dividend_yield",
]