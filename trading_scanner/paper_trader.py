import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PORTFOLIO_FILE  = Path(__file__).parent / "portfolio.json"
TRADES_FILE     = Path(__file__).parent / "trades.csv"
INITIAL_CAPITAL = 10_000_000  # 초기 가상 자본 (KRW)
MAX_POSITIONS   = 10
COOLDOWN_SECS   = 3600        # 동일 코인 재진입 쿨다운 1시간

TRADES_HEADER = [
    "trade_id", "market", "coin",
    "entry_time", "entry_price",
    "exit_time",  "exit_price",
    "quantity",   "invest_krw",
    "pnl_krw",    "pnl_pct",
    "exit_reason","confluence",
]


class PaperTrader:
    def __init__(self):
        self.cash:      float = float(INITIAL_CAPITAL)
        self.positions: dict  = {}
        self._trade_counter: int = 0
        self._cooldown: dict  = {}  # market -> exit datetime
        self._load_state()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_state(self):
        if PORTFOLIO_FILE.exists():
            data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
            self.cash           = float(data.get("cash", INITIAL_CAPITAL))
            self.positions      = data.get("positions", {})
            self._trade_counter = int(data.get("trade_counter", 0))
            for market, ts in data.get("cooldown", {}).items():
                self._cooldown[market] = datetime.fromisoformat(ts)
        else:
            self._save_state()

        if not TRADES_FILE.exists():
            with open(TRADES_FILE, "w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, fieldnames=TRADES_HEADER).writeheader()

    def _save_state(self):
        PORTFOLIO_FILE.write_text(
            json.dumps(
                {
                    "cash":          self.cash,
                    "positions":     self.positions,
                    "trade_counter": self._trade_counter,
                    "cooldown":      {m: dt.isoformat() for m, dt in self._cooldown.items()},
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    def _append_trade(self, trade: dict):
        with open(TRADES_FILE, "a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, fieldnames=TRADES_HEADER).writerow(trade)

    # ── Portfolio state ───────────────────────────────────────────────────────

    @property
    def total_assets(self) -> float:
        pos_value = sum(p["quantity"] * p["current_price"] for p in self.positions.values())
        return self.cash + pos_value

    def get_summary(self) -> dict:
        total = self.total_assets
        pnl   = total - INITIAL_CAPITAL
        return {
            "initial_capital": INITIAL_CAPITAL,
            "total_assets":    round(total),
            "cash":            round(self.cash),
            "pnl_krw":         round(pnl),
            "pnl_pct":         round(pnl / INITIAL_CAPITAL * 100, 2),
            "position_count":  len(self.positions),
            "max_positions":   MAX_POSITIONS,
        }

    def get_positions(self) -> list:
        rows = []
        for p in self.positions.values():
            ep  = p["entry_price"]
            cp  = p["current_price"]
            pnl_pct = (cp - ep) / ep * 100 if ep else 0
            pnl_krw = (cp - ep) * p["quantity"]
            rows.append({**p,
                "pnl_krw": round(pnl_krw),
                "pnl_pct": round(pnl_pct, 2),
            })
        return rows

    # ── Entry ─────────────────────────────────────────────────────────────────

    def try_entry(self, market: str, price: float, atr: float,
                  confluence: int, entry_time: str) -> bool:
        if len(self.positions) >= MAX_POSITIONS:
            return False
        if market in self.positions:
            return False
        if price <= 0 or atr <= 0:
            return False

        # 동일 코인 재진입 쿨다운 체크
        if market in self._cooldown:
            elapsed = (datetime.now(timezone.utc) - self._cooldown[market]).total_seconds()
            if elapsed < COOLDOWN_SECS:
                remaining = int((COOLDOWN_SECS - elapsed) / 60)
                logger.debug(f"COOLDOWN {market}  {remaining}분 남음")
                return False

        invest = min(self.total_assets / MAX_POSITIONS, self.cash)
        if invest < 1:
            return False

        quantity    = invest / price
        stop_dist   = max(atr * 1.5, price * 0.02)  # 최소 2% 손절 보장
        stop_loss   = price - stop_dist
        take_profit = price + stop_dist * 2.0        # RR=2.0 유지

        self.cash -= invest
        self.positions[market] = {
            "market":        market,
            "coin":          market.replace("KRW-", ""),
            "entry_price":   price,
            "quantity":      quantity,
            "invest_krw":    round(invest),
            "current_price": price,
            "stop_loss":     stop_loss,
            "take_profit":   take_profit,
            "entry_time":    entry_time,
            "atr":           atr,
            "confluence":    confluence,
            "st_bear_count": 0,
        }
        self._save_state()
        logger.info(
            f"ENTRY  {market:<14}  price={price:,.0f}  "
            f"invest={invest:,.0f}원  SL={stop_loss:,.0f}  TP={take_profit:,.0f}"
        )
        return True

    # ── Exit ──────────────────────────────────────────────────────────────────

    def update_price(self, market: str, current_price: float):
        if market in self.positions:
            self.positions[market]["current_price"] = current_price

    def check_exit(self, market: str, current_price: float, supertrend_bull: bool) -> dict | None:
        if market not in self.positions:
            return None

        self.positions[market]["current_price"] = current_price

        reason = None

        if not supertrend_bull:
            # ST 약세 카운트 누적 → 2회 연속일 때만 청산
            cnt = self.positions[market].get("st_bear_count", 0) + 1
            self.positions[market]["st_bear_count"] = cnt
            if cnt >= 6:  # 5분 스캔 × 6 = 30분 ≈ 15분봉 2봉 확인
                reason = "Supertrend 반전"
        else:
            # ST가 다시 강세로 돌아오면 카운트 초기화
            self.positions[market]["st_bear_count"] = 0

        if reason is None:
            if current_price <= self.positions[market]["stop_loss"]:
                reason = "손절"
            elif current_price >= self.positions[market]["take_profit"]:
                reason = "익절"

        if reason:
            return self._close_position(market, current_price, reason)

        self._save_state()
        return None

    def _close_position(self, market: str, exit_price: float, reason: str) -> dict:
        pos      = self.positions.pop(market)
        proceeds = pos["quantity"] * exit_price
        pnl_krw  = proceeds - pos["invest_krw"]
        pnl_pct  = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100

        self.cash += proceeds
        self._trade_counter += 1
        self._cooldown[market] = datetime.now(timezone.utc)  # 쿨다운 시작

        trade = {
            "trade_id":    self._trade_counter,
            "market":      market,
            "coin":        pos["coin"],
            "entry_time":  pos["entry_time"],
            "entry_price": round(pos["entry_price"], 8),
            "exit_time":   datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "exit_price":  round(exit_price, 8),
            "quantity":    round(pos["quantity"], 8),
            "invest_krw":  round(pos["invest_krw"]),
            "pnl_krw":     round(pnl_krw),
            "pnl_pct":     round(pnl_pct, 2),
            "exit_reason": reason,
            "confluence":  pos["confluence"],
        }
        self._append_trade(trade)
        self._save_state()
        logger.info(
            f"EXIT   {market:<14}  {reason:<12}  "
            f"pnl={pnl_krw:+,.0f}원 ({pnl_pct:+.2f}%)"
        )
        return trade


paper_trader = PaperTrader()
