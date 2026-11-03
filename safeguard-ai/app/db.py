import os
import sqlite3
import threading


class EventDB:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.c = sqlite3.connect(path, check_same_thread=False)
        self.c.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.c.execute(
                """CREATE TABLE IF NOT EXISTS events(
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, camera_id TEXT, camera TEXT,
                type TEXT, level TEXT, track_id INTEGER, confidence REAL, detail TEXT, snapshot TEXT)"""
            )
            self.c.commit()

    def add(self, ev: dict):
        with self.lock:
            self.c.execute(
                "INSERT INTO events(ts,camera_id,camera,type,level,track_id,confidence,detail,snapshot) VALUES(?,?,?,?,?,?,?,?,?)",
                (ev["ts"], ev["camera_id"], ev["camera"], ev["type"], ev["level"], ev.get("track_id"),
                 ev.get("confidence"), ev.get("detail"), ev.get("snapshot")),
            )
            self.c.commit()

    def recent(self, limit: int = 50, camera_id: str | None = None):
        q, args = "SELECT * FROM events", []
        if camera_id:
            q += " WHERE camera_id=?"
            args.append(camera_id)
        q += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self.lock:
            return [dict(r) for r in self.c.execute(q, args).fetchall()]
