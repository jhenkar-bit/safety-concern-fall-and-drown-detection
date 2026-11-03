"""Synthetic tests of the decision logic (no camera / model needed):  python -m tests.test_logic"""
from types import SimpleNamespace

import numpy as np

from app.detectors.fall import FallDetector

FALL = SimpleNamespace(down_angle=55, up_angle=35, still_motion=0.25, confirm_s=3.0,
                       candidate_timeout_s=12, long_lie_s=10, drop_vy=1.0)
rng = np.random.default_rng(0)


def person(sh, hip, ank, jitter=1.0):
    k = np.ones((17, 3))
    mid = lambda p, q: ((np.array(p) + np.array(q)) / 2)
    for i in (0, 1, 2, 3, 4, 7, 8, 9, 10):  # head/arms near the shoulders
        k[i, :2] = sh
    for i in (13, 14):  # knees between hip and ankle
        k[i, :2] = mid(hip, ank)
    for (a, b), (x, y) in {(5, 6): sh, (11, 12): hip, (15, 16): ank}.items():
        k[a, :2] = [x - 15, y]
        k[b, :2] = [x + 15, y]
    k[:, :2] += rng.normal(0, jitter, (17, 2))
    box = [k[:, 0].min(), k[:, 1].min(), k[:, 0].max(), k[:, 1].max()]
    return np.array(box), k


STAND = ((500, 200), (500, 300), (500, 400))
LYING = ((350, 420), (450, 420), (550, 420))
SIT = ((500, 280), (500, 380), (500, 420))


def lerp(a, b, f):
    return tuple(tuple(np.array(p) * (1 - f) + np.array(q) * f) for p, q in zip(a, b))


FRAME = np.zeros((720, 1280, 3), np.uint8)
DROP = ((500, 350), (500, 450), (500, 470))  # hips drop fast, torso stays vertical


def shift(pose, dx):
    return tuple((x + dx, y) for x, y in pose)


def run(script, switch_id_at=None, frame=None):
    det, events, t = FallDetector(FALL), [], 0.0
    for pose, secs in script:
        for _ in range(int(secs * 10)):
            if pose is None:  # nobody in view
                ev, _ = det.update([], [], None, t, frame)
            else:
                box, k = person(*pose)
                tid = 2 if (switch_id_at is not None and t >= switch_id_at) else 1  # tracker gives a NEW id
                ev, _ = det.update([tid], [box], k[None], t, frame)
            events += ev
            t += 0.1
    return events


def test_fall_detected():
    steps = [(lerp(STAND, LYING, f), 0.1) for f in (0.33, 0.66)]
    assert len(run([(STAND, 2)] + steps + [(LYING, 6)])) == 1


def test_fall_with_id_switch():
    steps = [(lerp(STAND, LYING, f), 0.1) for f in (0.33, 0.66)]
    assert len(run([(STAND, 2)] + steps + [(LYING, 6)], switch_id_at=2.1)) == 1  # id changes mid-fall


def test_found_lying_alarms():
    assert len(run([(LYING, 14)])) == 1  # never saw the drop, lying still 10s+


def test_fall_then_vanish_alarms():
    steps = [(lerp(STAND, LYING, f), 0.1) for f in (0.33, 0.66)]
    ev = run([(STAND, 2)] + steps + [(LYING, 0.3), (None, 8)], frame=FRAME)
    assert len(ev) == 1 and "disappeared" in ev[0]["detail"], ev


def test_sudden_drop_then_vanish_midframe_alarms():
    ev = run([(STAND, 2), (DROP, 0.2), (None, 8)], frame=FRAME)
    assert len(ev) == 1 and "dropped suddenly" in ev[0]["detail"], ev


def test_sudden_drop_at_frame_edge_is_not_a_fall():
    ev = run([(shift(STAND, 770), 2), (shift(DROP, 770), 0.2), (None, 8)], frame=FRAME)
    assert ev == [], ev


def test_walking_out_of_frame_no_alarm():
    walk = [(shift(STAND, d), 0.3) for d in (0, 150, 300, 450, 600, 750, 770)]
    assert run(walk + [(None, 12)], frame=FRAME) == []


def test_standing_then_vanishing_midframe_no_alarm():
    assert run([(STAND, 3), (None, 12)], frame=FRAME) == []  # e.g. stepped behind a door


def test_sitting_not_fall():
    assert run([(STAND, 2), (SIT, 10)]) == []


def test_bend_and_recover_not_fall():
    assert run([(STAND, 2), (LYING, 1.0), (STAND, 6)]) == []


def test_id_switch_after_fall_still_detected():
    """Tracker loses the person during the blur of the fall and issues a NEW id."""
    det, events, t = FallDetector(FALL), [], 0.0
    script = [(STAND, 2, 1), (lerp(STAND, LYING, 0.5), 0.1, 1), (LYING, 7, 2)]  # (pose, secs, track id)
    for pose, secs, tid in script:
        for _ in range(int(secs * 10)):
            box, k = person(*pose)
            events += det.update([tid], [box], k[None], t)[0]
            t += 0.1
    assert len(events) == 1, events


def test_found_lying_alarms():
    cfg = SimpleNamespace(**{**vars(FALL), "long_lie_s": 10})
    det, events, t = FallDetector(cfg), [], 0.0
    for _ in range(150):  # lying from the very first frame, never seen standing
        box, k = person(*LYING)
        events += det.update([1], [box], k[None], t)[0]
        t += 0.1
    assert len(events) == 1 and "lying motionless" in events[0]["detail"], events


def test_walking_not_fall():
    assert run([(STAND, 10)]) == []


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
