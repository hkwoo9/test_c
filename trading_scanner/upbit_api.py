import asyncio
import aiohttp
import pandas as pd
from typing import Optional

UPBIT_BASE = "https://api.upbit.com/v1"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"

_rate_sem = asyncio.Semaphore(8)


async def _get(session: aiohttp.ClientSession, url: str, params: dict = None) -> Optional[list | dict]:
    async with _rate_sem:
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 429:
                    await asyncio.sleep(1)
                    return None
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception:
            return None


async def get_krw_markets(session: aiohttp.ClientSession) -> list[str]:
    data = await _get(session, f"{UPBIT_BASE}/market/all", {"isDetails": "false"})
    if not data:
        return []
    return [m["market"] for m in data if m["market"].startswith("KRW-")]


def _to_df(data: list) -> Optional[pd.DataFrame]:
    if not data:
        return None
    df = pd.DataFrame(data)
    df = df.rename(columns={
        "candle_date_time_utc": "time",
        "opening_price": "open",
        "high_price": "high",
        "low_price": "low",
        "trade_price": "close",
        "candle_acc_trade_volume": "volume",
    })
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


async def get_candles_15m(session: aiohttp.ClientSession, market: str, count: int = 210) -> Optional[pd.DataFrame]:
    data = await _get(session, f"{UPBIT_BASE}/candles/minutes/15", {"market": market, "count": count})
    return _to_df(data)


async def get_candles_daily(session: aiohttp.ClientSession, market: str, count: int = 100) -> Optional[pd.DataFrame]:
    data = await _get(session, f"{UPBIT_BASE}/candles/days", {"market": market, "count": count})
    return _to_df(data)


async def get_btc_dominance(session: aiohttp.ClientSession) -> Optional[float]:
    data = await _get(session, f"{COINGECKO_BASE}/global")
    if not data:
        return None
    try:
        return float(data["data"]["market_cap_percentage"]["btc"])
    except Exception:
        return None
