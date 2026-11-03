"""Synthetic drowning-logic tests:  python -m tests.test_drowning   (needs opencv + numpy only)"""
from types import SimpleNamespace

import numpy as np

from app.detectors.drowning import DrowningDetector

CFG = SimpleNamespace(window_s=8, low_travel=0.6, vertical_ratio=1.15, low_motion=0.03, vertical_alarm_s=25,
                      motionless_alarm_s=15, submerge_alarm_s=10, warn_at=0.6, edge_margin_px=40)
POLY = [[100, 100], [1100, 100], [1100, 600], [100, 600]]
FRAME = np.zeros((720, 1280, 3), np.uint8)


def run(box_fn, secs, present=lambda t: True):
    det, evs, t = DrowningDetector(CFG, POLY), [], 0.0
    for _ in range(int(secs * 10)):
        if present(t):
            evs += det.update(FRAME, [box_fn(t)], [1], t)[0]
        else:
            evs += det.update(FRAME, [], [], t)[0]
        t += 0.1
    return evs


def test_upright_stationary_alarms():
    evs = run(lambda t: [500, 300, 560, 400], 35)  # tall box, never moves
    assert [e["level"] for e in evs] == ["warning", "alarm"] or [e["level"] for e in evs][-1] == "alarm", evs


def test_normal_swimmer_ok():
    evs = run(lambda t: [200 + 60 * t, 300, 320 + 60 * t, 350], 14)  # wide box, travelling ~1 m/s
    assert evs == []


def test_submerged_alarm():
    evs = run(lambda t: [500, 300, 620, 350], 25, present=lambda t: t < 5)  # vanishes mid-pool at 5 s
    assert any(e["level"] == "alarm" and "disappeared" in e["detail"] for e in evs), evs


def test_exit_at_edge_ok():
    evs = run(lambda t: [105, 300, 165, 350], 25, present=lambda t: t < 5)  # leaves near the pool edge
    assert evs == [], evs


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
