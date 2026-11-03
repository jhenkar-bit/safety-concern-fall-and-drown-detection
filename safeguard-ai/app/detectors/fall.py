"""Fall detection = person boxes/pose + temporal state machine (+ optional LSTM verifier).

Per tracked person we compute torso angle, bbox aspect, hip drop speed, and STILLNESS.
Stillness uses bbox motion + pixel change inside the box (NOT keypoint jitter, which is noisy on lying people).
State machine:  OK -> FALLING (upright -> down) -> FALLEN (down and mostly still for confirm_s).
Robustness fixes for real footage:
  * If the tracker loses the person during the fall (motion blur) and gives a NEW id, the new id adopts
    the old state, so the "was standing a moment ago" memory is not lost.
  * Stillness is a fraction of time (tolerates small twitches), not "never moved".
  * Found lying still for long_lie_s (without seeing the drop) also alarms.
"""
import math
from collections import deque

import cv2
import numpy as np

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12
CONF = 0.3
WINDOW = 30  # frames fed to the LSTM (3 s at 10 fps)


def _mid(k, a, b):
    pts = [k[i, :2] for i in (a, b) if k[i, 2] >= CONF]
    return np.mean(pts, axis=0) if pts else None


class _State:
    def __init__(self):
        self.last_t = None
        self.last_seen = 0.0
        self.prev_box = None  # (cx, cy, w, h)
        self.center = (0.0, 0.0)
        self.ref = 0.0
        self.motion = 1.0
        self.pix = 0.2
        self.hip_hist = deque(maxlen=40)
        self.vys = deque(maxlen=40)
        self.feats = deque(maxlen=WINDOW)
        self.still_hist = deque(maxlen=600)
        self.state = "OK"
        self.cand_t = 0.0
        self.sudden = False
        self.down_since = None
        self.not_down_since = None
        self.up_since = None
        self.last_upright = -1e9
        self.drop_t = -1e9        # last time the hips dropped fast
        self.reported = False     # an alarm for this person is already out
        self.last_xyxy = None


def build_lstm():
    import torch.nn as nn

    class FallLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(5, 64, 2, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(64, 1)

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.fc(o[:, -1]).squeeze(-1)

    return FallLSTM()


class FallClassifier:
    def __init__(self, path: str, device: str = "cpu"):
        import torch
        self.torch = torch
        self.m = build_lstm()
        self.m.load_state_dict(torch.load(path, map_location="cpu"))
        self.m.eval()

    def predict(self, feats) -> float:
        x = self.torch.tensor(np.array(feats), dtype=self.torch.float32)[None]
        with self.torch.no_grad():
            return float(self.torch.sigmoid(self.m(x)))


class FallDetector:
    def __init__(self, cfg, classifier: FallClassifier | None = None):
        self.cfg = cfg
        self.clf = classifier
        self.states: dict[int, _State] = {}
        self._prev_gray = None
        self._shape = None          # (height, width) of the video
        self._extent = (0.0, 0.0)   # fallback: furthest box corner seen so far

    # ---- config with safe defaults (so an older config.yaml still works)
    def _c(self, name, default):
        return getattr(self.cfg, name, default)

    def _diff(self, frame):
        if frame is None:
            return None
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        prev, self._prev_gray = self._prev_gray, g
        if prev is None or prev.shape != g.shape:
            return None
        return cv2.absdiff(g, prev) > 15

    def _adopt(self, box, current, t):
        """New track id appears near a recently lost one (e.g. blur during a fall) -> inherit its state."""
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        best, bd = None, None
        for tid, s in self.states.items():
            if tid in current or s.prev_box is None or t - s.last_seen > 6.0:
                continue
            d = math.hypot(cx - s.center[0], cy - s.center[1])
            if d < 1.5 * max(s.ref, 1.0) and (bd is None or d < bd):
                best, bd = tid, d
        return self.states.pop(best) if best is not None else None

    def update(self, ids, boxes, kpts, t, frame=None):
        """Returns (events, annotations). annotations = [(box, label, bgr_color)]."""
        events, ann = [], []
        diff = self._diff(frame)
        if frame is not None:
            self._shape = frame.shape[:2]
        for b in boxes:
            self._extent = (max(self._extent[0], float(b[2])), max(self._extent[1], float(b[3])))
        current = set(ids)
        for i, tid in enumerate(ids):
            st = self.states.get(tid)
            if st is None:
                st = self._adopt(boxes[i], current, t) or _State()
                self.states[tid] = st
            ev = self._step(st, tid, boxes[i], None if kpts is None else kpts[i], t, diff)
            st.last_seen = t
            if ev:
                events.append(ev)
                st.reported = True
            elif st.state == "OK":
                st.reported = False
            label, color = {"OK": (f"id{tid}", (80, 170, 60)),
                            "FALLING": (f"id{tid} checking", (0, 165, 255)),
                            "FALLEN": (f"id{tid} FALL", (0, 0, 220))}[st.state]
            ann.append((boxes[i], label, color))
        events += self._check_vanished(current, t, self._need())
        keep = max(8.0, self._need() + 3.0)
        for tid in [k for k, s in self.states.items() if t - s.last_seen > keep]:
            del self.states[tid]
        return events, ann

    def finish(self, t):
        """Call when a video ends: a person who went down and is already gone is reported without the full wait."""
        return self._check_vanished(set(), t, min(self._need(), 1.5))

    def _need(self):
        return float(self._c("vanish_confirm_s", 5.0))

    def _at_edge(self, box, m=0.04):
        H, W = self._shape if self._shape else (self._extent[1], self._extent[0])
        x1, y1, x2, y2 = box
        return x1 < m * W or x2 > (1 - m) * W or y1 < m * H or y2 > (1 - m) * H

    def _check_vanished(self, current, t, need):
        """Someone who went down (or dropped suddenly in mid-frame) and then vanished is treated as a fall.
        Walking out of the frame is not: that person was upright and left through an edge."""
        events = []
        if not self._c("vanish_alarm", True):
            return events
        for tid, s in self.states.items():
            if tid in current or s.reported or s.last_xyxy is None or t - s.last_seen < need:
                continue
            fell = s.state == "FALLING"
            sudden = (s.last_seen - s.drop_t <= 2.0) and not self._at_edge(s.last_xyxy)
            if fell or sudden:
                s.reported, s.state = True, "FALLEN"
                what = "went down" if fell else "dropped suddenly"
                events.append(self._event(tid, 0.7 if fell else 0.6,
                                          f"person {what} and then disappeared from view for {need:.0f}s"))
        return events

    @staticmethod
    def _still_frac(st, t, w):
        vals = [s for ts, s in st.still_hist if t - ts <= w]
        if len(vals) < 3 or st.still_hist[0][0] > t - 0.7 * w:
            return 0.0  # not enough history yet
        return sum(vals) / len(vals)

    def _step(self, st, tid, box, k, t, diff):
        c = self.cfg
        x1, y1, x2, y2 = (float(v) for v in box)
        w, h = x2 - x1, y2 - y1
        aspect = w / max(h, 1.0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dt = (t - st.last_t) if st.last_t is not None else None

        hip_y, angle = cy, None
        if k is not None:
            sh, hip = _mid(k, L_SH, R_SH), _mid(k, L_HIP, R_HIP)
            if sh is not None and hip is not None:
                d = sh - hip
                angle = math.degrees(math.atan2(abs(d[0]), abs(d[1]) + 1e-6))
            if hip is not None:
                hip_y = float(hip[1])
        if angle is None:  # no usable pose -> fall back to box shape
            angle = 75.0 if aspect > 1.1 else 15.0

        st.ref = max(max(w, h), st.ref * 0.998)  # body-size reference (posture independent)
        ref = max(st.ref, 1.0)

        if st.prev_box is not None and dt:  # box motion in body-sizes per second
            pcx, pcy, pw, ph = st.prev_box
            bm = (math.hypot(cx - pcx, cy - pcy) + abs(w - pw) + abs(h - ph)) / ref / dt
            st.motion = 0.6 * st.motion + 0.4 * bm
        if diff is not None:  # share of pixels that changed inside the box
            reg = diff[max(int(y1), 0):max(int(y2), 0), max(int(x1), 0):max(int(x2), 0)]
            if reg.size:
                st.pix = 0.6 * st.pix + 0.4 * float(reg.mean())
        st.prev_box, st.center = (cx, cy, w, h), (cx, cy)

        st.hip_hist.append((t, hip_y))
        vy = 0.0
        for ts, y in st.hip_hist:  # oldest sample within the last 0.6 s
            if 0.2 < t - ts <= 0.6:
                vy = (hip_y - y) / (t - ts) / ref
                break
        st.vys.append((t, vy))
        peak = max([v for ts, v in st.vys if t - ts <= 1.0] or [0.0])
        st.feats.append([angle / 90, min(aspect, 3) / 3, max(-3, min(3, vy)) / 3,
                         min(st.motion, 2) / 2, min(max(w, h) / ref, 1.5) / 1.5])
        st.last_t = t
        st.last_xyxy = (x1, y1, x2, y2)
        if peak > c.drop_vy:
            st.drop_t = t

        upright = angle < c.up_angle and aspect < 0.9
        down = angle > c.down_angle or aspect > 1.1
        still = st.motion < c.still_motion and (diff is None or st.pix < self._c("still_pix", 0.08))
        st.still_hist.append((t, still))
        if upright:
            st.last_upright = t
        if down:  # continuous "down" time, tolerating 0.7 s of flicker
            st.not_down_since = None
            st.down_since = st.down_since or t
        else:
            st.not_down_since = st.not_down_since or t
            if t - st.not_down_since > 0.7:
                st.down_since = None

        confirm, ratio = c.confirm_s, self._c("still_ratio", 0.6)
        if st.state == "OK":
            if down and t - st.last_upright <= self._c("upright_window_s", 4.0):
                st.state, st.cand_t, st.sudden = "FALLING", t, peak > c.drop_vy
            elif (down and st.down_since and t - st.down_since >= c.long_lie_s
                  and self._still_frac(st, t, 5.0) >= 0.7):
                st.state = "FALLEN"
                return self._event(tid, 0.7, f"person lying motionless for {c.long_lie_s:.0f}s")
        elif st.state == "FALLING":
            if upright:
                st.up_since = st.up_since or t
                if t - st.up_since > 1.0:  # recovered -> not a fall
                    st.state, st.up_since = "OK", None
            else:
                st.up_since = None
            if st.down_since and t - st.down_since >= confirm + 0.7 and self._still_frac(st, t, confirm) >= ratio:
                p = self.clf.predict(st.feats) if (self.clf and len(st.feats) == WINDOW) else None
                if p is None or p >= 0.5:
                    st.state = "FALLEN"
                    conf = p if p is not None else (0.9 if st.sudden else 0.75)
                    why = "sudden drop" if st.sudden else "gradual drop"
                    return self._event(tid, conf, f"{why}, then lying still for {confirm:.0f}s")
            if t - st.cand_t > c.candidate_timeout_s:
                st.state = "OK"
        elif st.state == "FALLEN":
            if upright:
                st.up_since = st.up_since or t
                if t - st.up_since > 1.0:
                    st.state, st.up_since = "OK", None
            else:
                st.up_since = None
        return None

    @staticmethod
    def _event(tid, conf, detail):
        return {"type": "fall", "level": "alarm", "track_id": int(tid), "confidence": float(conf), "detail": detail}
