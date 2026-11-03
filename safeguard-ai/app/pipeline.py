"""One StreamWorker per camera: grab -> YOLO track -> detector -> annotate -> events/alerts."""
import logging
import os
import threading
import time

import cv2
import numpy as np

from .config import camera_ready
from .detectors.drowning import DrowningDetector
from .detectors.fall import FallClassifier, FallDetector

log = logging.getLogger("pipeline")


class FrameReader:
    """Reads continuously in its own thread and keeps ONLY the latest frame (no RTSP lag build-up)."""

    def __init__(self, source, loop=False):
        self.source, self.loop = source, loop
        self.is_file = isinstance(source, str) and not source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.frame, self.seq, self.connected, self.ended = None, 0, False, False
        self._lock, self._stop = threading.Lock(), threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()

    def get(self):
        with self._lock:
            return self.frame, self.seq

    def _run(self):
        while not self._stop.is_set():
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                self.connected = False
                self._fails = getattr(self, "_fails", 0) + 1
                if self._fails % 10 == 1:  # first failure, then about every 30 seconds (not every 3)
                    log.warning("cannot open the camera (attempt %d), still retrying. Check IP Webcam is running, "
                                "same Wi-Fi, and that the link ends in /video. Test with: python check_camera.py", self._fails)
                time.sleep(3)
                continue
            self._fails = 0
            self.connected = True
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            while not self._stop.is_set():
                ok, fr = cap.read()
                if not ok:
                    break
                with self._lock:
                    self.frame, self.seq = fr, self.seq + 1
                if self.is_file:
                    time.sleep(1 / fps)  # play files at real-time speed
            cap.release()
            self.connected = False
            if self.is_file and not self.loop:
                self.ended = True
                return
            time.sleep(2)  # reconnect delay


class StreamWorker:
    def __init__(self, cam, cfg, db, alerts):
        self.cam, self.cfg, self.db, self.alerts = cam, cfg, db, alerts
        self.id, self.mode = cam.id, cam.mode
        self.latest_jpeg = None
        self.fps, self.running, self.last_event, self.error = 0.0, False, None, None
        self.started_at, self.manual_stop = 0.0, False
        self._stop, self._thread, self._reader = threading.Event(), None, None
        self._last_frame = 0.0
        os.makedirs(cfg.server.snapshots_dir, exist_ok=True)

    def start(self):
        if self.running:
            return
        if not camera_ready(self.cam):
            self.error = "No camera link set yet. Add it to .env (HOME_CAM_URL) and restart."
            return
        self._stop.clear()
        self.running, self.error = True, None
        self.started_at, self.manual_stop = time.time(), False
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"cam-{self.id}")
        self._thread.start()

    def stop(self):
        self.manual_stop = True  # the supervisor must not restart a camera someone stopped on purpose
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        self.running = False

    def status(self):
        return {"id": self.id, "name": self.cam.name, "mode": self.mode, "running": self.running,
                "connected": bool(self._reader and self._reader.connected), "fps": round(self.fps, 1),
                "last_frame_age": round(time.time() - self._last_frame, 1) if self._last_frame else None,
                "last_event": self.last_event, "error": self.error}

    def _run(self):
        try:
            from ultralytics import YOLO
            d = self.cfg.detection
            if self.mode == "land":
                clf = FallClassifier(self.cfg.fall.lstm_path) if self.cfg.fall.use_lstm else None
                self.det, weights, self.classes = FallDetector(self.cfg.fall, clf), d.pose_weights, [0]
            else:
                self.det = DrowningDetector(self.cfg.drowning, getattr(self.cam, "pool_polygon", None))
                weights, self.classes = d.pool_weights, getattr(d, "pool_classes", None)
            self.model = YOLO(weights)
            self._reader = FrameReader(self.cam.source, loop=bool(getattr(self.cam, "loop", False)))
            self._reader.start()
            interval, last_seq, next_t = 1.0 / d.target_fps, -1, 0.0
            while not self._stop.is_set():
                now = time.time()
                frame, seq = self._reader.get()
                if now < next_t or frame is None or seq == last_seq:
                    if self._reader.ended:
                        break
                    time.sleep(0.005)
                    continue
                last_seq, next_t = seq, now + interval
                t0 = time.time()
                self._process(frame, now)
                self.fps = 0.8 * self.fps + 0.2 / max(time.time() - t0, 1e-3) if self.fps else 1 / max(time.time() - t0, 1e-3)
        except Exception as e:
            log.exception("worker %s crashed", self.id)
            self.error = str(e)
        finally:
            if self._reader:
                self._reader.stop()
            self.running = False

    def _process(self, frame, t):
        d = self.cfg.detection
        res = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=d.conf, imgsz=d.imgsz,
                               classes=self.classes, device=d.device, verbose=False)[0]
        b = res.boxes
        if b is not None and len(b) and b.id is not None:
            boxes, ids = b.xyxy.cpu().numpy(), b.id.int().cpu().tolist()
            kpts = res.keypoints.data.cpu().numpy() if (self.mode == "land" and res.keypoints is not None) else None
        else:
            boxes, ids, kpts = np.zeros((0, 4)), [], None

        if self.mode == "land":
            events, ann = self.det.update(ids, boxes, kpts, t, frame)
        else:
            events, ann = self.det.update(frame, boxes, ids, t)

        vis = frame.copy()
        poly = getattr(self.cam, "pool_polygon", None)
        if poly:
            cv2.polylines(vis, [np.array(poly, np.int32)], True, (200, 170, 40), 2)
        for box, label, color in ann:
            x1, y1, x2, y2 = (int(v) for v in box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(vis, f"{self.cam.name}  {time.strftime('%H:%M:%S')}  {self.fps:.0f}fps", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            self.latest_jpeg = jpg.tobytes()
        self._last_frame = time.time()

        for ev in events:
            ev.update(camera_id=self.id, camera=self.cam.name, mode=self.mode, ts=t)
            name = f"{int(t)}_{self.id}_{ev['type']}_{ev['level']}.jpg"
            path = os.path.join(self.cfg.server.snapshots_dir, name)
            cv2.imwrite(path, vis)
            ev["snapshot"] = name
            self.db.add(ev)
            self.alerts.dispatch(ev, path)
            self.last_event = {k: ev[k] for k in ("ts", "type", "level", "detail")}
            log.warning("EVENT %s", ev)
