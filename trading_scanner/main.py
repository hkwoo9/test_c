import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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


@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/signals")
async def get_signals():
    return {
        "signals":   scanner.signals,
        "last_scan": scanner.last_scan,
        "scanning":  scanner.is_scanning,
    }


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=50)
    scanner.subscribe(queue)

    # send current state immediately
    await ws.send_text(json.dumps({
        "type":      "signals",
        "signals":   scanner.signals,
        "last_scan": scanner.last_scan,
        "scanning":  scanner.is_scanning,
    }))

    try:
        while True:
            msg = await asyncio.wait_for(queue.get(), timeout=30)
            await ws.send_text(json.dumps(msg))
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception:
        pass
    finally:
        scanner.unsubscribe(queue)
