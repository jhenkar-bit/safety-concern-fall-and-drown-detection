"""Tests for the always-on supervisor (fake cameras, no network):  python -m tests.test_supervisor"""
import time
from types import SimpleNamespace as NS

from app.supervisor import Supervisor


class FakeWorker:
    def __init__(self):
        self.running, self.error, self.started_at, self.manual_stop, self.age = True, None, 0.0, False, 1.0
        self.starts = 0

    def status(self):
        return {"last_frame_age": self.age}

    def start(self):
        self.running, self.starts, self.manual_stop = True, self.starts + 1, False


class FakeAlerts:
    def __init__(self):
        self.msgs = []

    def notify(self, text):
        self.msgs.append(text)


class FakeDB:
    def recent(self, limit=500):
        return [{"level": "alarm", "ts": time.time() - 100}, {"level": "alarm", "ts": time.time() - 999999}]


def make(hb=9):
    cfg = NS(alerts=NS(offline_after_s=90, heartbeat_hour=hb), cameras=[NS(id="c1", name="Hall", autostart=True, source="http://cam/video")])
    w, a = FakeWorker(), FakeAlerts()
    return Supervisor(cfg, {"c1": w}, FakeDB(), a), w, a


def test_offline_then_back_online_each_reported_once():
    s, w, a = make(hb=None)
    w.age = 200
    s.tick(); s.tick(); s.tick()
    assert len([m for m in a.msgs if "OFFLINE" in m]) == 1
    w.age = 1
    s.tick(); s.tick()
    assert len([m for m in a.msgs if "back online" in m]) == 1


def test_crashed_worker_is_restarted_and_reported():
    s, w, a = make(hb=None)
    w.running, w.error = False, "model failed"
    s.tick()
    assert w.starts == 1 and any("model failed" in m for m in a.msgs)


def test_restart_backs_off():
    s, w, a = make(hb=None)
    w.running = False
    now = time.time()
    s.tick(now)                 # restart 1
    w.running = False
    s.tick(now + 5)             # too soon: no second restart yet
    assert w.starts == 1
    s.tick(now + 40)            # after back-off
    assert w.starts == 2


def test_manual_stop_is_respected():
    s, w, a = make(hb=None)
    w.running, w.manual_stop = False, True
    s.tick()
    assert w.starts == 0 and a.msgs == []


def test_daily_heartbeat_once_per_day():
    s, w, a = make(hb=0)        # hour 0 = always due
    s.tick(); s.tick(); s.tick()
    beats = [m for m in a.msgs if "is running" in m]
    assert len(beats) == 1 and "1 of 1" in beats[0] and "last 24 hours: 1" in beats[0]


def test_camera_without_link_is_ignored():
    cfg = NS(alerts=NS(offline_after_s=90, heartbeat_hour=None),
             cameras=[NS(id="c1", name="Hall", autostart=True, source="${HOME_CAM_URL}"),
                      NS(id="c2", name="Yard", autostart=True, source="")])
    a = FakeAlerts()
    s = Supervisor(cfg, {"c1": FakeWorker(), "c2": FakeWorker()}, FakeDB(), a)
    s.announce_start("pc")
    s.tick(); s.tick()
    assert s.cams == [] and len(a.msgs) == 1 and "no camera link" in a.msgs[0]


def test_heartbeat_can_be_turned_off():
    s, w, a = make(hb=None)
    s.tick()
    assert a.msgs == []


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("PASS", n)
