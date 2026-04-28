import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

from indicators import confluence_score, supertrend, _atr
from paper_trader import paper_trader
from upbit_api import (
    get_btc_dominance,
    get_candles_15m,
    get_candles_daily,
    get_krw_markets,
)

logger = logging.getLogger(__name__)

SCAN_INTERVAL  = 300    # 5분
MIN_CONFLUENCE = 6
REQUEST_DELAY  = 0.12   # ~8 req/s


class Scanner:
    def __init__(self):
        self.signals:     list          = []
        self.last_scan:   Optional[str] = None
        self.is_scanning: bool          = False
        self._dom_history: list         = []
        self._subscribers: set          = set()

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
        closed_trades = []

        connector = aiohttp.TCPConnector(limit=20)
        async with aiohttp.ClientSession(connector=connector) as session:
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

            df_btc = await get_candles_15m(session, "KRW-BTC", count=210)
            await asyncio.sleep(REQUEST_DELAY)

            total = len(markets)
            found = 0

            for idx, market in enumerate(markets):
                try:
                    df_15m = await get_candles_15m(session, market)
                    await asyncio.sleep(REQUEST_DELAY)
                    df_day = await get_candles_daily(session, market)
                    await asyncio.sleep(REQUEST_DELAY)

                    if df_15m is None or df_day is None or len(df_15m) < 60:
                        continue

                    is_btc        = market == "KRW-BTC"
                    current_price = float(df_15m["close"].iloc[-1])

                    trend  = supertrend(df_15m, period=10, multiplier=2.5)
                    st_buy = bool(trend[-1] == 1)

                    is_long, score, detail = confluence_score(
                        df_15m, df_day,
                        df_btc_15m=df_btc,
                        btc_dom_history=self._dom_history,
                        is_btc=is_btc,
                        min_conf=MIN_CONFLUENCE,
                    )

                    # ── 보유 포지션 청산 체크 ──────────────────────────────
                    if market in paper_trader.positions:
                        closed = paper_trader.check_exit(market, current_price, st_buy)
                        if closed:
                            closed_trades.append(closed)
                    else:
                        paper_trader.update_price(market, current_price)

                    # ── 매수 신호 → 진입 시도 ─────────────────────────────
                    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

                    if st_buy and is_long and market not in paper_trader.positions:
                        atr_val = float(_atr(
                            df_15m["high"].values,
                            df_15m["low"].values,
                            df_15m["close"].values,
                            14,
                        )[-1])
                        paper_trader.try_entry(market, current_price, atr_val, score, now_str)

                    # ── 신호 목록 수집 ────────────────────────────────────
                    if st_buy and is_long:
                        coin    = market.replace("KRW-", "")
                        prev24  = float(df_15m["close"].iloc[max(0, len(df_15m) - 96)])
                        chg_pct = (current_price - prev24) / prev24 * 100 if prev24 > 0 else 0.0
                        signals.append({
                            "market":      market,
                            "coin":        coin,
                            "price":       current_price,
                            "change_pct":  round(chg_pct, 2),
                            "supertrend":  st_buy,
                            "confluence":  score,
                            "details":     detail,
                            "signal_time": now_str,
                            "in_position": market in paper_trader.positions,
                        })
                        found += 1
                        logger.info(f"SIGNAL  {market:<14}  conf={score}/8")

                    if (idx + 1) % 20 == 0:
                        pct = (idx + 1) / total * 100
                        logger.info(f"Scanned {idx+1}/{total} ({pct:.0f}%)  signals: {found}")
                        await self._push({"type": "progress", "scanned": idx + 1, "total": total})

                except Exception as e:
                    logger.debug(f"Error scanning {market}: {e}")

        signals.sort(key=lambda x: (-x["confluence"], -x["change_pct"]))
        self.signals   = signals
        self.last_scan = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        self.is_scanning = False

        await self._push({
            "type":          "signals",
            "signals":       signals,
            "last_scan":     self.last_scan,
            "portfolio":     paper_trader.get_summary(),
            "positions":     paper_trader.get_positions(),
            "closed_trades": closed_trades,
        })
        logger.info(f"Scan complete. signals={len(signals)}  closed={len(closed_trades)}")

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
