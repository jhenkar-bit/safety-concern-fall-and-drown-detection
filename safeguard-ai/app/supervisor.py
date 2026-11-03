"""Keeps the always-on service honest: restarts crashed cameras, reports offline cameras,
and sends a daily 'still running' message. Without this, a dead camera would fail silently."""
import asyncio
import logging
import time

from .config import camera_ready

log = logging.getLogger("supervisor")


class Supervisor:
    def __init__(self, cfg, workers, db, alerts):
        a = cfg.alerts
        self.cams = [c for c in cfg.cameras if getattr(c, "autostart", False) and camera_ready(c)]
        self.offline_after = getattr(a, "offline_after_s", 90)
        self.hb_hour = getattr(a, "heartbeat_hour", 9)  # null in config.yaml = no daily message
        self.workers, self.db, self.alerts = workers, db, alerts
        self.state = {c.id: {"offline": False, "restarts": 0, "next_try": 0.0} for c in self.cams}
        self.last_beat_day = None

    @staticmethod
    def _day(now):
        lt = time.localtime(now)
        return (lt.tm_year, lt.tm_yday), lt.tm_hour

    def announce_start(self, host: str, now=None):
        now = now or time.time()
        if self.cams:
            self.alerts.notify(f"SafeGuard started on {host}. Watching: {', '.join(c.name for c in self.cams)}.")
        else:
            self.alerts.notify(f"SafeGuard started on {host}, but no camera link is set yet, so nothing is being watched. "
                               f"Add HOME_CAM_URL to the .env file and restart.")
        day, hour = self._day(now)
        if self.hb_hour is not None and hour >= self.hb_hour:
            self.last_beat_day = day  # the start message counts as today's check-in

    def tick(self, now=None):
        now = now or time.time()
        for cam in self.cams:
            w, st = self.workers[cam.id], self.state[cam.id]
            if getattr(w, "manual_stop", False):
                continue  # someone stopped it on purpose from the dashboard
            if not w.running:
                if not st["offline"]:
                    st["offline"] = True
                    self.alerts.notify(f"Camera '{cam.name}' stopped ({w.error or 'unknown reason'}). "
                                       f"Restarting automatically. Monitoring is NOT working until then.")
                if now >= st["next_try"]:
                    st["restarts"] += 1
                    st["next_try"] = now + min(300, 15 * st["restarts"])  # back off up to 5 minutes
                    w.start()
                continue
            age = w.status()["last_frame_age"]
            stale = (now - w.started_at > self.offline_after + 30) if age is None else age > self.offline_after
            if stale and not st["offline"]:
                st["offline"] = True
                self.alerts.notify(f"Camera '{cam.name}' is OFFLINE: no video for {self.offline_after} seconds. "
                                   f"Monitoring is NOT working. Check the camera power and Wi-Fi.")
            elif not stale and st["offline"]:
                st["offline"], st["restarts"] = False, 0
                self.alerts.notify(f"Camera '{cam.name}' is back online. Monitoring has resumed.")
        self._heartbeat(now)

    def _heartbeat(self, now):
        if self.hb_hour is None:
            return
        day, hour = self._day(now)
        if day != self.last_beat_day and hour >= self.hb_hour:
            self.last_beat_day = day
            online = sum(1 for c in self.cams if self.workers[c.id].running and not self.state[c.id]["offline"])
            alarms = len([e for e in self.db.recent(500) if e["level"] == "alarm" and e["ts"] > now - 86400])
            self.alerts.notify(f"SafeGuard is running. Cameras online: {online} of {len(self.cams)}. "
                               f"Alarms in the last 24 hours: {alarms}.")

    async def run(self):
        while True:
            try:
                self.tick()
            except Exception:
                log.exception("supervisor check failed")
            await asyncio.sleep(30)
