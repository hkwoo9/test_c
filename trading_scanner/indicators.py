import numpy as np
import pandas as pd
from typing import Optional, Tuple


# ─── Primitives ───────────────────────────────────────────────────────────────

def _ema(series: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1)
    out = np.full(len(series), np.nan)
    for i, v in enumerate(series):
        if np.isnan(v):
            continue
        if np.isnan(out[i - 1]) if i > 0 else True:
            out[i] = v
        else:
            out[i] = alpha * v + (1 - alpha) * out[i - 1]
    return out


def _wilder_smooth(series: np.ndarray, period: int) -> np.ndarray:
    """Wilder's smoothing (alpha = 1/period), initialised with SMA."""
    alpha = 1.0 / period
    out = np.full(len(series), np.nan)
    first = period
    while first < len(series) and np.isnan(series[first - period:first]).any():
        first += 1
    if first >= len(series):
        return out
    out[first - 1] = np.nanmean(series[first - period:first])
    for i in range(first, len(series)):
        if not np.isnan(series[i]):
            out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _wilder_smooth(gain, period)
    avg_loss = _wilder_smooth(loss, period)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_close),
                    np.abs(low  - prev_close)))
    return _wilder_smooth(tr, period)


# ─── Supertrend ───────────────────────────────────────────────────────────────

def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> np.ndarray:
    """Returns trend array: +1 = bullish, -1 = bearish."""
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    hl2   = (high + low) / 2.0
    atr   = _atr(high, low, close, period)

    up = hl2 - multiplier * atr
    dn = hl2 + multiplier * atr
    n  = len(close)

    up_trail = up.copy()
    dn_trail = dn.copy()
    trend    = np.ones(n, dtype=int)

    for i in range(1, n):
        up_trail[i] = max(up[i], up_trail[i - 1]) if close[i - 1] > up_trail[i - 1] else up[i]
        dn_trail[i] = min(dn[i], dn_trail[i - 1]) if close[i - 1] < dn_trail[i - 1] else dn[i]

        if trend[i - 1] == -1 and close[i] > dn_trail[i - 1]:
            trend[i] = 1
        elif trend[i - 1] == 1 and close[i] < up_trail[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    return trend


# ─── Confluence helpers ───────────────────────────────────────────────────────

def _stoch_rsi(close: np.ndarray, rsi_len=14, stoch_len=14, k_sm=3, d_sm=3):
    rsi = _rsi(close, rsi_len)
    k_out = np.full(len(close), np.nan)
    d_out = np.full(len(close), np.nan)

    for i in range(stoch_len - 1, len(rsi)):
        window = rsi[i - stoch_len + 1:i + 1]
        lo, hi = np.nanmin(window), np.nanmax(window)
        rng = hi - lo
        k_out[i] = 100.0 * (rsi[i] - lo) / rng if rng > 0 else 0.0

    # k smoothing
    k_sm_out = np.full(len(close), np.nan)
    for i in range(k_sm - 1, len(k_out)):
        w = k_out[i - k_sm + 1:i + 1]
        if not np.isnan(w).any():
            k_sm_out[i] = np.mean(w)

    # d smoothing
    for i in range(d_sm - 1, len(k_sm_out)):
        w = k_sm_out[i - d_sm + 1:i + 1]
        if not np.isnan(w).any():
            d_out[i] = np.mean(w)

    return k_sm_out, d_out


def _find_pivot_lo(low: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(low)
    pivots = np.full(n, np.nan)
    for i in range(left, n - right):
        window = low[i - left:i + right + 1]
        if low[i] == np.min(window) and low[i] < np.min(np.concatenate([low[i - left:i], low[i + 1:i + right + 1]])):
            pivots[i] = low[i]
    return pivots


def _find_pivot_hi(high: np.ndarray, left: int, right: int) -> np.ndarray:
    n = len(high)
    pivots = np.full(n, np.nan)
    for i in range(left, n - right):
        window = high[i - left:i + right + 1]
        if high[i] == np.max(window) and high[i] > np.max(np.concatenate([high[i - left:i], high[i + 1:i + right + 1]])):
            pivots[i] = high[i]
    return pivots


def _liquidity_signals(df: pd.DataFrame, left=3, right=3, sweep_lb=10, tol=0.15
                       ) -> Tuple[bool, bool, bool, bool]:
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    n     = len(df)

    pl_arr = _find_pivot_lo(low,  left, right)
    ph_arr = _find_pivot_hi(high, left, right)

    last_pl = np.nan
    last_ph = np.nan
    ssl_bar = -9999
    bsl_bar = -9999

    for i in range(n):
        if not np.isnan(pl_arr[i]):
            last_pl = pl_arr[i]
        if not np.isnan(ph_arr[i]):
            last_ph = ph_arr[i]

        if not np.isnan(last_pl) and low[i] < last_pl and close[i] > last_pl:
            ssl_bar = i
        if not np.isnan(last_ph) and high[i] > last_ph and close[i] < last_ph:
            bsl_bar = i

    last = n - 1
    ssl_active = (last - ssl_bar) <= sweep_lb
    bsl_active = (last - bsl_bar) <= sweep_lb

    near_lo = False
    near_hi = False
    if not np.isnan(last_pl) and last_pl > 0:
        near_lo = abs(close[-1] - last_pl) / last_pl * 100 <= tol
    if not np.isnan(last_ph) and last_ph > 0:
        near_hi = abs(close[-1] - last_ph) / last_ph * 100 <= tol

    return ssl_active, bsl_active, near_lo, near_hi


# ─── Confluence (8-factor) ────────────────────────────────────────────────────

def confluence_score(
    df_15m:  pd.DataFrame,
    df_daily: pd.DataFrame,
    df_btc_15m: Optional[pd.DataFrame] = None,
    btc_dom_history: Optional[list] = None,
    is_btc: bool = False,
    min_conf: int = 6,
) -> Tuple[bool, int, dict]:
    """
    Returns (is_long_signal, score, detail_dict).
    Each detail value is True/False.
    """
    if len(df_15m) < 60:
        return False, 0, {}

    close  = df_15m["close"].values
    high   = df_15m["high"].values
    low    = df_15m["low"].values

    details: dict[str, bool] = {}

    # 1 ── EMA Trend (20 > 50)
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    c1 = bool(ema20[-1] > ema50[-1])
    details["EMA Trend"] = c1

    # 2 ── Stochastic RSI: K > D (상태 기반) AND K < 35 (반등 초기만 포착)
    k, d = _stoch_rsi(close)
    if not np.isnan(k[-1]) and not np.isnan(d[-1]):
        c2 = bool(k[-1] > d[-1] and k[-1] < 35)
    else:
        c2 = False
    details["StochRSI"] = c2

    # 3 ── MACD: MACD > Signal AND 히스토그램 양수 (상태 기반)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd  = ema12 - ema26
    sig   = _ema(macd, 9)
    hist  = macd - sig
    c3 = bool(macd[-1] > sig[-1] and hist[-1] > 0)
    details["MACD"] = c3

    # 4 ── Liquidity: SSL sweep(20봉) or near pivot low(0.8%)
    ssl_active, _, near_lo, _ = _liquidity_signals(df_15m, sweep_lb=20, tol=0.8)
    c4 = bool(ssl_active or near_lo)
    details["Liquidity"] = c4

    # 5 ── Squeeze + Keltner Breakout
    bb_len, bb_mult = 20, 2.0
    kc_len, kc_mult, sq_look = 20, 1.5, 10

    bb_basis = pd.Series(close).rolling(bb_len).mean().values
    bb_std   = pd.Series(close).rolling(bb_len).std(ddof=0).values
    bb_upper = bb_basis + bb_mult * bb_std
    bb_lower = bb_basis - bb_mult * bb_std

    kc_ema   = _ema(close, kc_len)
    kc_atr   = _atr(high, low, close, kc_len)
    kc_upper = kc_ema + kc_mult * kc_atr
    kc_lower = kc_ema - kc_mult * kc_atr

    in_sq    = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    recent_sq = bool(np.any(in_sq[-(sq_look + 1):]))
    kc_break  = recent_sq and (close[-1] > kc_upper[-1]) and (close[-2] <= kc_upper[-2])
    c5 = bool(kc_break)
    details["Squeeze/KC"] = c5

    # 6 ── HTF Daily EMA confirm
    if len(df_daily) >= 50:
        dc     = df_daily["close"].values
        d_ema20 = _ema(dc, 20)
        d_ema50 = _ema(dc, 50)
        c6 = bool(d_ema20[-1] > d_ema50[-1])
    else:
        c6 = True
    details["HTF Daily"] = c6

    # 7 ── BTC Dominance + Relative Strength
    if is_btc:
        c7 = True
    elif btc_dom_history and len(btc_dom_history) >= 2 and df_btc_15m is not None and len(df_btc_15m) >= 50:
        dom_arr   = np.array(btc_dom_history, dtype=float)
        dom_sma   = np.mean(dom_arr[-50:]) if len(dom_arr) >= 50 else np.mean(dom_arr)
        dom_falling = float(dom_arr[-1]) < dom_sma

        btc_cl   = df_btc_15m.set_index("time")["close"]
        coin_cl  = df_15m.set_index("time")["close"]
        aligned  = btc_cl.reindex(coin_cl.index, method="nearest")
        rel      = (coin_cl / aligned.values).fillna(method="ffill").dropna().values
        rel_e20  = _ema(rel, 20)
        rel_e50  = _ema(rel, 50)
        rel_up   = bool(rel_e20[-1] > rel_e50[-1])
        c7 = bool(rel_up)  # domFalling 제거: BTC 시즌에도 상대강도만으로 판단
    else:
        c7 = True
    details["Dominance/RS"] = c7

    # 8 ── Bollinger Basis Reclaim
    bb_look = 10
    touched_lower = bool(np.any(low[-(bb_look + 1):] <= bb_lower[-(bb_look + 1):]))
    c8 = bool(touched_lower and close[-1] > bb_basis[-1])
    details["BB Reclaim"] = c8

    score   = sum(details.values())
    is_long = score >= min_conf
    return is_long, score, details
