import asyncio
import logging
import os
import secrets
import socket
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .alerts import AlertManager
from .config import camera_ready, load_config
from .db import EventDB
from .pipeline import StreamWorker
from .supervisor import Supervisor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
cfg = load_config()
os.makedirs(cfg.server.snapshots_dir, exist_ok=True)
db = EventDB(cfg.server.db_path)
alerts = AlertManager(cfg)
workers: dict[str, StreamWorker] = {}
HERE = os.path.dirname(__file__)


def require_key(x_api_key: str | None = Header(default=None)):
    key = os.getenv("API_KEY")
    if key and not secrets.compare_digest(x_api_key or "", key):
        raise HTTPException(401, "invalid or missing x-api-key")


@asynccontextmanager
async def lifespan(app: FastAPI):
    for cam in cfg.cameras:
        workers[cam.id] = StreamWorker(cam, cfg, db, alerts)
        if getattr(cam, "autostart", False) and camera_ready(cam):
            workers[cam.id].start()
    sup = Supervisor(cfg, workers, db, alerts)
    sup.announce_start(socket.gethostname())
    task = asyncio.create_task(sup.run())
    yield
    task.cancel()
    for w in workers.values():
        w.stop()


app = FastAPI(title="SafeGuard AI", version="1.0", lifespan=lifespan)
app.mount("/snapshots", StaticFiles(directory=cfg.server.snapshots_dir), name="snapshots")


def _worker(cid: str) -> StreamWorker:
    if cid not in workers:
        raise HTTPException(404, "unknown camera")
    return workers[cid]


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(os.path.join(HERE, "static", "index.html"))


@app.get("/health")
def health():
    return {"ok": True, "cameras_running": sum(w.running for w in workers.values())}


@app.get("/api/cameras")
def cameras():
    return [w.status() for w in workers.values()]


@app.post("/api/cameras/{cid}/start", dependencies=[Depends(require_key)])
def start(cid: str):
    _worker(cid).start()
    return _worker(cid).status()


@app.post("/api/cameras/{cid}/stop", dependencies=[Depends(require_key)])
def stop(cid: str):
    _worker(cid).stop()
    return _worker(cid).status()


@app.get("/api/events")
def events(limit: int = 50, camera_id: str | None = None):
    return db.recent(min(limit, 500), camera_id)


@app.post("/api/siren/{state}", dependencies=[Depends(require_key)])
def siren(state: str):
    if state not in ("on", "off"):
        raise HTTPException(400, "state must be on|off")
    return {"sent": alerts.siren(state == "on")}


@app.post("/api/test-alert", dependencies=[Depends(require_key)])
def test_alert():
    alerts.notify("SafeGuard test message. If you can read this, Telegram alerts work.")
    return {"sent": True}


@app.get("/stream/{cid}")
def stream(cid: str):
    w = _worker(cid)

    def gen():
        while True:
            if w.latest_jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + w.latest_jpeg + b"\r\n"
            time.sleep(0.1)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")
