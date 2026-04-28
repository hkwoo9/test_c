"""
Background scanner: every SCAN_INTERVAL seconds fetches 15m + daily candles
for all KRW Upbit markets and checks both indicators.
"""
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from indicators import confluence_score, supertrend
from upbit_api import (
    get_btc_dominance,
    get_candles_15m,
    get_candles_daily,
    get_krw_markets,
)

logger = logging.getLogger(__name__)

SCAN_INTERVAL   = 300        # 5 minutes
MIN_CONFLUENCE  = 6
REQUEST_DELAY   = 0.12       # ~8 req/s to stay under Upbit limit


class Scanner:
    def __init__(self):
        self.signals:      list  = []
        self.last_scan:    Optional[str] = None
        self.is_scanning:  bool  = False
        self._dom_history: list  = []   # rolling BTC dominance readings
        self._subscribers: set   = set()

    def subscribe(self, queue: asyncio.Queue):
        self._subscribers.add(queue)

    def unsubscribe(self, queue: asyncio.Queue):
        self._subscribers.discard(queue)

    async def _push(self, payload: dict):
        dead = set()
        for q in self._subscribers:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                dead.add(q)
        self._subscribers -= dead

    async def _scan_once(self):
        self.is_scanning = True
        signals = []

        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
            # BTC dominance
            dom = await get_btc_dominance(session)
            if dom is not None:
                self._dom_history.append(dom)
                if len(self._dom_history) > 200:
                    self._dom_history = self._dom_history[-200:]

            markets = await get_krw_markets(session)
            if not markets:
                logger.warning("No markets fetched")
                self.is_scanning = False
                return

            # BTC 15m candles for relative-strength calculation
            df_btc = await get_candles_15m(session, "KRW-BTC", count=210)
            await asyncio.sleep(REQUEST_DELAY)

            total   = len(markets)
            found   = 0

            for idx, market in enumerate(markets):
                try:
                    df_15m = await get_candles_15m(session, market)
                    await asyncio.sleep(REQUEST_DELAY)
                    df_day = await get_candles_daily(session, market)
                    await asyncio.sleep(REQUEST_DELAY)

                    if df_15m is None or df_day is None or len(df_15m) < 60:
                        continue

                    is_btc = market == "KRW-BTC"

                    # ── Supertrend check ──────────────────────────────────
                    trend = supertrend(df_15m, period=10, multiplier=3.0)
                    st_buy = bool(trend[-1] == 1)

                    # ── Confluence check ──────────────────────────────────
                    is_long, score, detail = confluence_score(
                        df_15m,
                        df_day,
                        df_btc_15m=df_btc,
                        btc_dom_history=self._dom_history,
                        is_btc=is_btc,
                        min_conf=MIN_CONFLUENCE,
                    )

                    if st_buy and is_long:
                        coin_name = market.replace("KRW-", "")
                        price     = float(df_15m["close"].iloc[-1])
                        prev24    = float(df_15m["close"].iloc[max(0, len(df_15m) - 96)])
                        chg_pct   = (price - prev24) / prev24 * 100 if prev24 > 0 else 0.0

                        sig = {
                            "market":      market,
                            "coin":        coin_name,
                            "price":       price,
                            "change_pct":  round(chg_pct, 2),
                            "supertrend":  st_buy,
                            "confluence":  score,
                            "details":     detail,
                            "signal_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                        }
                        signals.append(sig)
                        found += 1
                        logger.info(f"SIGNAL  {market:<14}  conf={score}/8  ST=Buy")

                    if (idx + 1) % 20 == 0:
                        pct = (idx + 1) / total * 100
                        logger.info(f"Scanned {idx + 1}/{total} ({pct:.0f}%)  signals so far: {found}")
                        await self._push({"type": "progress", "scanned": idx + 1, "total": total})

                except Exception as e:
                    logger.debug(f"Error scanning {market}: {e}")

        # sort by confluence desc, then change_pct desc
        signals.sort(key=lambda x: (-x["confluence"], -x["change_pct"]))
        self.signals   = signals
        self.last_scan = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.is_scanning = False

        await self._push({"type": "signals", "signals": signals, "last_scan": self.last_scan})
        logger.info(f"Scan complete. {len(signals)} buy signal(s) found.")

    async def run_forever(self):
        while True:
            try:
                logger.info("Starting scan cycle...")
                await self._scan_once()
            except Exception as e:
                logger.error(f"Scan error: {e}")
                self.is_scanning = False
            await asyncio.sleep(SCAN_INTERVAL)


scanner = Scanner()
