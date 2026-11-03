"""Drowning-risk heuristics on tracked swimmers (baseline: tune on YOUR pool footage).

Signals (each converted to a 0..1+ risk, final risk = max):
  1. vertical + not progressing  (instinctive drowning response)  -> vertical_alarm_s
  2. not progressing + no pixel motion in bbox (floating/sinking)  -> motionless_alarm_s
  3. swimmer track vanishes in mid-pool (submerged)                -> submerge_alarm_s
Levels: risk >= warn_at -> warning, risk >= 1 -> alarm. A lost track that re-appears nearby
(new ID after being underwater) inherits its old state so the timers keep running.
"""
import math
from collections import deque

import cv2
import numpy as np


class _Swim:
    def __init__(self, t):
        self.hist = deque()
        self.motion = 0.2
        self.vert_acc = 0.0
        self.still_acc = 0.0
        self.level = 0
        self.last_t = t
        self.lost_at = None
        self.last_box = None
        self.last_cen = (0.0, 0.0)
        self.last_dist = 1e9


class DrowningDetector:
    def __init__(self, cfg, polygon=None):
        self.cfg = cfg
        self.poly = np.array(polygon, np.int32).reshape(-1, 1, 2) if polygon else None
        self.tracks: dict[int, _Swim] = {}
        self.lost: dict[int, _Swim] = {}
        self.prev_gray = None

    def _dist_inside(self, p):
        if self.poly is None:
            return 1e9
        return cv2.pointPolygonTest(self.poly, (float(p[0]), float(p[1])), True)

    def _adopt(self, cen, size, t):
        for tid, st in list(self.lost.items()):
            if t - st.lost_at < 15 and math.dist(cen, st.last_cen) < 1.5 * size:
                del self.lost[tid]
                st.lost_at = None
                return st
        return None

    def update(self, frame, boxes, ids, t):
        c = self.cfg
        events, ann = [], []
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        diff = None if self.prev_gray is None else (cv2.absdiff(gray, self.prev_gray) > 15)
        self.prev_gray = gray
        seen = set()

        for box, tid in zip(boxes, ids):
            x1, y1, x2, y2 = (float(v) for v in box)
            w, h = x2 - x1, y2 - y1
            cen = ((x1 + x2) / 2, (y1 + y2) / 2)
            d = self._dist_inside(cen)
            if d < 0:  # outside the water polygon -> ignore (people on the deck)
                continue
            seen.add(tid)
            st = self.tracks.get(tid)
            if st is None:
                st = self.lost.pop(tid, None) or self._adopt(cen, max(w, h), t) or _Swim(t)
                st.lost_at = None
                self.tracks[tid] = st
            dt = max(t - st.last_t, 1e-3)
            st.last_t, st.last_box, st.last_cen, st.last_dist = t, (x1, y1, x2, y2), cen, d

            if diff is not None:
                reg = diff[max(int(y1), 0):int(y2), max(int(x1), 0):int(x2)]
                st.motion = 0.7 * st.motion + 0.3 * (float(reg.mean()) if reg.size else 0.2)
            st.hist.append((t, cen[0], cen[1], max(w, h)))
            while st.hist and t - st.hist[0][0] > c.window_s:
                st.hist.popleft()

            travel = 9.9  # unknown until the window is mostly filled
            if t - st.hist[0][0] >= 0.8 * c.window_s:
                size = float(np.mean([s[3] for s in st.hist]))
                travel = math.dist(cen, (st.hist[0][1], st.hist[0][2])) / max(size, 1.0)
            stationary = travel < c.low_travel
            vertical = (h / max(w, 1.0)) >= c.vertical_ratio

            st.vert_acc = st.vert_acc + dt if (stationary and vertical) else max(0.0, st.vert_acc - 2 * dt)
            st.still_acc = st.still_acc + dt if (stationary and st.motion < c.low_motion) else max(0.0, st.still_acc - 2 * dt)
            r1, r2 = st.vert_acc / c.vertical_alarm_s, st.still_acc / c.motionless_alarm_s
            risk = max(r1, r2)
            why = "upright and not progressing" if r1 >= r2 else "motionless in water"
            ev = self._level(st, tid, risk, why, c)
            if ev:
                events.append(ev)
            ann.append(self._ann(box, tid, risk, c))

        for tid in [k for k in self.tracks if k not in seen]:  # vanished this frame
            st = self.tracks.pop(tid)
            if st.last_dist > c.edge_margin_px:  # not near the edge -> may be submerged
                st.lost_at = t
                self.lost[tid] = st

        for tid, st in list(self.lost.items()):
            gap = t - st.lost_at
            if gap > 60:
                del self.lost[tid]
                continue
            risk = gap / c.submerge_alarm_s
            ev = self._level(st, tid, risk, f"swimmer disappeared mid-pool for {gap:.0f}s", c)
            if ev:
                events.append(ev)
            ann.append((st.last_box, f"id{tid} SUBMERGED? {gap:.0f}s", (0, 0, 220) if risk >= 1 else (0, 165, 255)))
        return events, ann

    @staticmethod
    def _ann(box, tid, risk, c):
        if risk >= 1:
            return box, f"id{tid} DROWNING RISK", (0, 0, 220)
        if risk >= c.warn_at:
            return box, f"id{tid} watch {risk:.0%}", (0, 165, 255)
        return box, f"id{tid}", (200, 170, 40)

    @staticmethod
    def _level(st, tid, risk, why, c):
        level = 2 if risk >= 1 else 1 if risk >= c.warn_at else 0
        if level == 0 and risk < c.warn_at * 0.5:
            st.level = 0
        if level > st.level:
            st.level = level
            return {"type": "drowning", "level": "alarm" if level == 2 else "warning", "track_id": int(tid),
                    "confidence": float(min(0.95, 0.5 + 0.45 * min(risk, 1.0))), "detail": why}
        return None
