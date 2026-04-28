import asyncio
import csv
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from paper_trader import TRADES_FILE, paper_trader
from scanner import scanner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)

app = FastAPI(title="Upbit Trading Signal Scanner")

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
async def startup():
    asyncio.create_task(scanner.run_forever())
    asyncio.create_task(scanner.run_price_stream())


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/signals")
async def get_signals():
    return {
        "signals":   scanner.signals,
        "last_scan": scanner.last_scan,
        "scanning":  scanner.is_scanning,
        "portfolio": paper_trader.get_summary(),
        "positions": paper_trader.get_positions(),
    }


@app.get("/api/trades")
async def get_trades():
    if not TRADES_FILE.exists():
        return []
    with open(TRADES_FILE, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return list(reversed(rows))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    scanner.subscribe(queue)

    await ws.send_text(json.dumps({
        "type":      "signals",
        "signals":   scanner.signals,
        "last_scan": scanner.last_scan,
        "scanning":  scanner.is_scanning,
        "portfolio": paper_trader.get_summary(),
        "positions": paper_trader.get_positions(),
    }))

    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=20)
                await ws.send_text(json.dumps(msg))
            except asyncio.TimeoutError:
                # keep-alive ping so browser doesn't close the connection
                await ws.send_text(json.dumps({"type": "ping"}))
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        scanner.unsubscribe(queue)
