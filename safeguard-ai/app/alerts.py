import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from .sound import play_local, stop_local

log = logging.getLogger("alerts")


class AlertManager:
    """Telegram (photo + text), siren on this PC's speakers, and an optional HTTP hardware siren."""

    def __init__(self, cfg):
        a = cfg.alerts
        self.cooldown = a.cooldown_s
        self.siren_seconds = a.siren_seconds
        self.siren_on_warning = a.siren_on_warning
        self.local_siren = getattr(a, "local_siren", True)
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat = os.getenv("TELEGRAM_CHAT_ID")
        self.siren_url = os.getenv("SIREN_URL")
        self.siren_key = os.getenv("SIREN_KEY", "")
        self._last: dict = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=4)

    def dispatch(self, ev: dict, snapshot: str | None = None) -> bool:
        key = (ev["camera_id"], ev["type"], ev["level"])
        with self._lock:
            if time.time() - self._last.get(key, 0) < self.cooldown:
                return False
            self._last[key] = time.time()
        self._pool.submit(self._deliver, ev, snapshot)
        return True

    def notify(self, text: str):
        """Plain Telegram message (status messages: started, camera offline, daily check-in)."""
        log.warning("NOTIFY %s", text)
        self._pool.submit(self._telegram, text, None)

    def _deliver(self, ev, snapshot):
        if ev["level"] == "alarm" or self.siren_on_warning:
            self.siren(True)  # siren first: it is the time-critical channel
        icon = "ALARM" if ev["level"] == "alarm" else "WARNING"
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ev["ts"]))
        text = (f"{icon}: {ev['type'].upper()}\nCamera: {ev['camera']}\n"
                f"Reason: {ev['detail']}\nConfidence: {ev['confidence']:.0%}\nTime: {when}")
        self._telegram(text, snapshot)

    def _telegram(self, text, photo_path):
        if not (self.token and self.chat):
            log.warning("Telegram not configured; alert only logged: %s", text.replace("\n", " | "))
            return
        base = f"https://api.telegram.org/bot{self.token}"
        for attempt in range(3):
            try:
                if photo_path and os.path.exists(photo_path):
                    with open(photo_path, "rb") as f:
                        r = requests.post(f"{base}/sendPhoto", data={"chat_id": self.chat, "caption": text},
                                          files={"photo": f}, timeout=10)
                else:
                    r = requests.post(f"{base}/sendMessage", data={"chat_id": self.chat, "text": text}, timeout=10)
                if r.ok:
                    return
                log.error("Telegram HTTP %s: %s", r.status_code, r.text[:200])
            except Exception as e:  # network errors must never kill the pipeline
                log.error("Telegram error: %s", e)
            time.sleep(2 ** attempt)

    def siren(self, on: bool = True, seconds: int | None = None) -> bool:
        """Ring (or silence) the siren on this PC and the hardware siren, if configured."""
        secs = int(seconds or self.siren_seconds)
        ok = False
        if on:
            ok = bool(self.local_siren and play_local(secs))
        else:
            stop_local()
            ok = True
        if self.siren_url:
            try:
                r = requests.post(self.siren_url, headers={"x-key": self.siren_key},
                                  json={"state": "on" if on else "off", "seconds": secs}, timeout=3)
                ok = ok or r.ok
            except Exception as e:
                log.error("Siren error: %s", e)
        elif not ok:
            log.warning("No siren available (no speakers and no SIREN_URL)")
        return ok
