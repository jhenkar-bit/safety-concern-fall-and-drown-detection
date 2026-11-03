"""SafeGuard AI - Streamlit front end.   Run:  streamlit run streamlit_app.py
Tab 1: upload a short video -> annotated result + alerts (for demos).
Tab 2: paste a CCTV link (RTSP/HTTP) -> live monitoring with alerts."""
import json
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from types import SimpleNamespace
from urllib.parse import urlsplit

import cv2
import numpy as np
import streamlit as st

from app.alerts import AlertManager
from app.config import load_config
from app.detectors.drowning import DrowningDetector
from app.detectors.fall import FallClassifier, FallDetector
from app.pipeline import FrameReader

HERE = os.path.dirname(os.path.abspath(__file__))
st.set_page_config(page_title="SafeGuard AI", page_icon="🛟", layout="wide")


# ---------------------------------------------------------------- helpers
@st.cache_data
def siren_wav(secs: int) -> bytes:
    sr = 16000
    t = np.linspace(0, secs, int(sr * secs), False)
    freq = np.where((t * 3) % 1 < 0.5, 960, 700)  # two-tone siren
    phase = 2 * np.pi * np.cumsum(freq) / sr  # continuous phase, no clicks
    pcm = (0.45 * np.sin(phase) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def play_siren(ph, secs):
    """Play the siren for `secs` seconds. A sound problem must never stop the analysis."""
    try:
        data = siren_wav(int(secs))
        key = f"siren_{time.time_ns()}"  # unique key: Streamlit rejects two identical audio elements
        ph.empty()
        for kw in ({"autoplay": True, "key": key}, {"key": key}, {}):  # newest to oldest Streamlit
            try:
                ph.audio(data, format="audio/wav", **kw)
                return
            except TypeError:
                continue
    except Exception:
        pass


@st.cache_resource
def _player():
    return {"proc": None}


def stop_local():
    """Silence the alarm playing on this computer."""
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif _player()["proc"] is not None:
            _player()["proc"].terminate()
    except Exception:
        pass


def play_local(secs) -> bool:
    """Play the siren through THIS computer's speakers. Returns False if no player is available."""
    try:
        path = os.path.join(tempfile.gettempdir(), f"safeguard_siren_{int(secs)}.wav")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                f.write(siren_wav(int(secs)))
        stop_local()
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        for cmd in (["afplay", path], ["paplay", path], ["aplay", "-q", path]):  # Mac, Linux
            if shutil.which(cmd[0]):
                _player()["proc"] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
    except Exception:
        pass
    return False


def ring(ph, secs):
    """Computer speakers first (never blocked); browser sound only if that is off or unavailable."""
    if use_speakers and play_local(secs):
        return
    play_siren(ph, secs)


def banner(ph, ev):
    colour = "#c8312b" if ev["level"] == "alarm" else "#b87a00"
    ph.markdown(
        f"<div style='background:{colour};color:#fff;padding:14px 18px;border-radius:6px;font-size:1.05rem'>"
        f"<b>{ev['type'].title()} {ev['level']}</b>: {ev['detail']} ({ev['where']})</div>",
        unsafe_allow_html=True)


def fit_image(target, image):
    """Stretch an image to its column. Newer Streamlit renamed the option, so try the new name first."""
    try:
        target.image(image, width="stretch")
    except Exception:
        target.image(image, use_container_width=True)


def render_events(slot, events, limit=12, cols=4):
    """Show events as a grid, newest first. slot must be st.empty(): each call replaces its content."""
    with slot.container():
        if not events:
            st.caption("No events yet.")
            return
        shown = list(reversed(events[-limit:]))
        for i in range(0, len(shown), cols):
            row = st.columns(cols)
            for col, e in zip(row, shown[i:i + cols]):
                with col:
                    fit_image(col, e["image"])
                    st.markdown(f"**{e['type'].title()} {e['level']}**  \n{e['detail']}  \n"
                                f"{e['where']}, {e['confidence']:.0%}")
        if len(events) > limit:
            st.caption(f"Showing the latest {limit} of {len(events)} events.")


def parse_source(s: str):
    s = s.strip()
    if s.isdigit():
        return int(s)
    return s if urlsplit(s).scheme.lower() in ("rtsp", "rtmp", "http", "https") else None


def parse_polygon(txt: str):
    if not txt.strip():
        return None
    try:
        pts = json.loads(txt)
        assert len(pts) >= 3 and all(len(p) == 2 for p in pts)
        return pts
    except Exception:
        st.sidebar.error("Water area must look like [[x,y],[x,y],[x,y],...]")
        return None


# ---------------------------------------------------------------- sidebar / config
cfg = load_config(os.path.join(HERE, "config.yaml"))
d = cfg.detection
with st.sidebar:
    st.header("Detection")
    mode_label = st.radio("What should it watch for?", ["Fall on land", "Drowning in pool"])
    mode = "land" if mode_label.startswith("Fall") else "pool"
    d.conf = st.slider("Detection confidence", 0.2, 0.9, float(d.conf), 0.05)
    d.target_fps = st.slider("Analysed frames per second", 2, 15, int(d.target_fps))
    d.device = st.text_input("Device", d.device, help='"cpu", or "0" for the first GPU')
    d.imgsz = st.select_slider("Detection resolution (higher = more accurate, slower)", [640, 800, 960, 1280], value=int(d.imgsz) if int(d.imgsz) in (640, 800, 960, 1280) else 800)
    preview_w = st.slider("Video preview width (px)", 320, 1000, 520, 40)
    preview_h = st.slider("Video preview max height (px)", 200, 700, 360, 20)
    polygon = None
    if mode == "land":
        models = {"Fast (nano)": "yolo11n-pose.pt", "Balanced (small)": "yolo11s-pose.pt",
                  "Accurate (medium)": "yolo11m-pose.pt", "Most accurate (large, slow)": "yolo11l-pose.pt"}
        names = list(models)
        cur = [k for k, v in models.items() if v == d.pose_weights]
        d.pose_weights = models[st.selectbox("Model size", names, index=names.index(cur[0]) if cur else 2,
                                             help="Bigger is more accurate but slower. The file downloads by itself the first time.")]
        cfg.fall.confirm_s = st.slider("Seconds lying still before alarm", 1.0, 10.0, float(cfg.fall.confirm_s), 0.5)
        cfg.fall.long_lie_s = st.slider("Alarm if found lying still for (s)", 5, 60, int(cfg.fall.long_lie_s))
        cfg.fall.vanish_alarm = st.checkbox("Alarm if a person falls and disappears from view", True)
        cfg.fall.vanish_confirm_s = st.slider("Disappeared for (seconds) before alarm", 2, 30, int(getattr(cfg.fall, "vanish_confirm_s", 5)))
        cfg.fall.down_angle = st.slider("Lying angle threshold (degrees)", 40, 80, int(cfg.fall.down_angle))
        if os.path.exists(os.path.join(HERE, cfg.fall.lstm_path)):
            cfg.fall.use_lstm = st.checkbox("Confirm with trained LSTM", True)
        else:
            cfg.fall.use_lstm = False
    else:
        d.pool_weights = st.text_input("Swimmer model", d.pool_weights)
        if st.checkbox("Custom swimmer model (no class filter)", False):
            d.pool_classes = None
        profile = st.radio("Timing", ["Short demo clip", "Real pool"], help="Real pools use longer, more conservative timers.")
        dr = cfg.drowning
        if profile == "Short demo clip":
            dr.window_s, dr.vertical_alarm_s, dr.motionless_alarm_s, dr.submerge_alarm_s = 4, 8, 6, 4
        dr.vertical_alarm_s = st.slider("Upright and stuck: alarm after (s)", 3, 60, int(dr.vertical_alarm_s), key=f"va{profile}")
        dr.motionless_alarm_s = st.slider("Motionless: alarm after (s)", 3, 60, int(dr.motionless_alarm_s), key=f"ma{profile}")
        dr.submerge_alarm_s = st.slider("Disappeared: alarm after (s)", 2, 30, int(dr.submerge_alarm_s), key=f"sa{profile}")
        dr.vertical_ratio = st.slider("Upright body: height/width above", 0.8, 2.5, float(dr.vertical_ratio), 0.05)
        polygon = parse_polygon(st.text_area("Water area (optional)", placeholder="[[100,200],[1100,200],[1200,650],[50,650]]",
                                             help="Pixel coordinates of the water. Empty = whole frame."))
    st.header("Alerts")
    st.caption("Alarms always show on screen with a siren sound.")
    alarm_secs = st.slider("Alarm sound length (seconds)", 3, 100, 10)
    cfg.alerts.siren_seconds = alarm_secs  # hardware siren rings for the same time
    use_speakers = st.checkbox("Play alarm through this computer's speakers", True,
                               help="Browsers often block automatic sound. This plays it from the computer running the app, so it always works.")
    st.button("Silence alarm", help="Any button press also silences it. This restarts the current live session.")
    send_real = st.checkbox("Also send Telegram and hardware siren", False)
    if send_real:
        os.environ["TELEGRAM_BOT_TOKEN"] = st.text_input("Telegram bot token", os.getenv("TELEGRAM_BOT_TOKEN", ""), type="password")
        os.environ["TELEGRAM_CHAT_ID"] = st.text_input("Telegram chat ID", os.getenv("TELEGRAM_CHAT_ID", ""))
        os.environ["SIREN_URL"] = st.text_input("Siren URL", os.getenv("SIREN_URL", ""))
# Cap the height of videos and frames (Streamlit has no height option, so we add CSS)
st.markdown(
    f"""<style>
    video {{ max-height: {preview_h}px; width: 100%; object-fit: contain; }}
    [data-testid="stImage"] img {{ max-height: {preview_h}px; width: auto; max-width: 100%; }}
    </style>""",
    unsafe_allow_html=True,
)
stop_local()  # every page action (including the Silence button) stops a ringing alarm
alerts = AlertManager(cfg)


# ---------------------------------------------------------------- engine
def build_engine():
    from ultralytics import YOLO
    if mode == "land":
        clf = FallClassifier(os.path.join(HERE, cfg.fall.lstm_path)) if cfg.fall.use_lstm else None
        det, weights, classes = FallDetector(cfg.fall, clf), d.pose_weights, [0]
    else:
        det, weights, classes = DrowningDetector(cfg.drowning, polygon), d.pool_weights, getattr(d, "pool_classes", None)
    model = YOLO(weights)
    model.predictor = None  # fresh tracker for every run
    return SimpleNamespace(model=model, det=det, classes=classes, alarm_until=-1.0)


def process_frame(eng, frame, t):
    res = eng.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=d.conf, imgsz=d.imgsz,
                          classes=eng.classes, device=d.device, verbose=False)[0]
    b = res.boxes
    if b is not None and len(b) and b.id is not None:
        boxes, ids = b.xyxy.cpu().numpy(), b.id.int().cpu().tolist()
        kpts = res.keypoints.data.cpu().numpy() if (mode == "land" and res.keypoints is not None) else None
    else:
        boxes, ids, kpts = np.zeros((0, 4)), [], None
    events, ann = eng.det.update(ids, boxes, kpts, t, frame) if mode == "land" else eng.det.update(frame, boxes, ids, t)

    vis = frame.copy()
    if polygon:
        cv2.polylines(vis, [np.array(polygon, np.int32)], True, (200, 170, 40), 2)
    for box, label, colour in ann:
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(vis, label, (x1, max(y1 - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
    if any(e["level"] == "alarm" for e in events):
        eng.alarm_until = t + 3
    if t < eng.alarm_until:
        h, w = vis.shape[:2]
        cv2.rectangle(vis, (0, 0), (w - 1, h - 1), (0, 0, 220), 10)
        cv2.putText(vis, "ALARM", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 220), 4)
    return vis, events


def handle_events(events, vis, where, camera):
    """Fill in metadata, optionally dispatch real alerts, return display records."""
    out = []
    for ev in events:
        ev.update(camera_id=camera, camera=camera, mode=mode, ts=time.time())
        if send_real:
            path = os.path.join(tempfile.gettempdir(), f"sg_{int(ev['ts'])}_{ev['type']}_{ev['level']}.jpg")
            cv2.imwrite(path, vis)
            alerts.dispatch(ev, path)
        out.append({**ev, "where": where, "image": vis[:, :, ::-1].copy()})
    return out


# ---------------------------------------------------------------- tabs
st.title("SafeGuard AI")
st.caption("Spots falls on land and swimmers in trouble in a pool, then raises an alert. Assistive tool: it does not replace supervision.")
tab_demo, tab_live = st.tabs(["Upload a video", "Live CCTV link"])

# ------------------------------- demo
def run_demo(upload):
    suffix = os.path.splitext(upload.name)[1] or ".mp4"
    src = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    src.write(upload.getbuffer())
    src.close()
    cap = cv2.VideoCapture(src.name)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, round(fps / d.target_fps))
    raw_out = os.path.join(tempfile.gettempdir(), f"sg_out_{int(time.time())}.mp4")
    writer = cv2.VideoWriter(raw_out, cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (W, H))

    with st.spinner("Loading model"):
        eng = build_engine()
    prog, banner_ph, sound_ph = st.progress(0.0), st.empty(), st.empty()
    frame_ph, log_ph = st.empty(), st.empty()
    stored, i, shown, t_start = [], 0, 0, time.time()
    last_ring = 0.0
    last_vis, last_t = None, 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % step:
                continue
            t = i / fps  # video time, so timers behave the same however fast we process
            vis, events = process_frame(eng, frame, t)
            last_vis, last_t = vis, t
            writer.write(vis)
            new = handle_events(events, vis, f"at {t:.1f}s in the video", upload.name)
            if new:
                stored += new
                banner(banner_ph, new[-1])
                if any(e["level"] == "alarm" for e in new) and time.time() - last_ring > max(alarm_secs, 5):
                    ring(sound_ph, alarm_secs)
                    last_ring = time.time()
                render_events(log_ph, stored)
            shown += 1
            if shown % 2 == 0 or new:
                frame_ph.image(vis[:, :, ::-1], width=preview_w)
                prog.progress(min(i / total, 1.0), text=f"Analysing: {t:.1f}s")
        if mode == "land" and last_vis is not None:  # video ended: report anyone who went down and is already gone
            late = handle_events(eng.det.finish(last_t), last_vis, f"at the end of the video ({last_t:.1f}s)", upload.name)
            if late:
                stored += late
                banner(banner_ph, late[-1])
                ring(sound_ph, alarm_secs)
    finally:
        cap.release()
        writer.release()
        os.unlink(src.name)
    for ph in (prog, banner_ph, frame_ph):
        ph.empty()
    log_ph.empty()

    playable = raw_out
    if shutil.which("ffmpeg"):  # browsers need H.264; OpenCV writes mp4v
        h264 = raw_out.replace(".mp4", "_h264.mp4")
        r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", raw_out, "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", h264], capture_output=True)
        if r.returncode == 0:
            playable = h264
    with open(playable, "rb") as f:
        video_bytes = f.read()
    st.session_state.demo = {"events": stored, "video": video_bytes, "playable": playable != raw_out,
                             "seconds": time.time() - t_start, "frames": shown, "name": upload.name}


def show_demo_result(r):
    st.subheader("Result")
    alarms = [e for e in r["events"] if e["level"] == "alarm"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Frames analysed", r["frames"])
    c2.metric("Alarms", len(alarms))
    c3.metric("Processing time", f"{r['seconds']:.0f}s")
    if alarms:
        banner(st.empty(), alarms[0])
    else:
        st.info("No fall or drowning event detected in this video with the current settings.")
    left, _ = st.columns([2, 3])
    with left:
        if r["playable"]:
            st.video(r["video"])
        else:
            st.caption("Install ffmpeg to preview the annotated video in the browser. You can still download it.")
        st.download_button("Download annotated video", r["video"], file_name=f"annotated_{r['name']}.mp4", mime="video/mp4")
    st.markdown("**Events**")
    render_events(st.empty(), r["events"], limit=16, cols=4)


with tab_demo:
    st.write("Upload a short clip, for example a staged fall or a swimmer in difficulty. The annotated result plays back with each alert marked.")
    up = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
    if st.button("Analyse video", type="primary", disabled=up is None):
        run_demo(up)
    if st.session_state.get("demo"):
        show_demo_result(st.session_state.demo)

# ------------------------------- live
def run_live(src, name):
    eng_ph = st.empty()
    with eng_ph, st.spinner("Loading model and connecting"):
        eng = build_engine()
    eng_ph.empty()
    reader = FrameReader(src)
    reader.start()
    banner_ph, sound_ph, status_ph = st.empty(), st.empty(), st.empty()
    frame_ph, log_ph = st.empty(), st.empty()
    last_seq, next_t, started, fps, last_siren = -1, 0.0, time.time(), 0.0, 0.0
    try:
        while True:
            now = time.time()
            if now < next_t:
                time.sleep(0.005)
                continue
            frame, seq = reader.get()
            if frame is None or seq == last_seq:
                if frame is None and now - started > 20:
                    status_ph.error("Could not read from this link. Check the address, login and that this computer can reach the camera.")
                    st.session_state.live = None
                    return
                time.sleep(0.05)
                continue
            last_seq, next_t = seq, now + 1.0 / d.target_fps
            vis, events = process_frame(eng, frame, now)
            fps = 0.8 * fps + 0.2 / max(time.time() - now, 1e-3) if fps else 1 / max(time.time() - now, 1e-3)
            frame_ph.image(vis[:, :, ::-1], width=preview_w)
            status_ph.caption(f"Monitoring {name}: {fps:.0f} analysed frames per second")
            new = handle_events(events, vis, time.strftime("%H:%M:%S"), name)
            if new:
                st.session_state.live_events = (st.session_state.live_events + new)[-20:]
                banner(banner_ph, new[-1])
                render_events(log_ph, st.session_state.live_events)
                if any(e["level"] == "alarm" for e in new) and now - last_siren > max(alarm_secs, 5):
                    ring(sound_ph, alarm_secs)
                    last_siren = now
                st.toast(f"{new[-1]['type'].title()} {new[-1]['level']}: {new[-1]['detail']}")
    finally:
        reader.stop()


with tab_live:
    st.write("Paste the camera's stream address. The computer running this app must be able to reach the camera (same network or VPN).")
    url = st.text_input("CCTV stream link", placeholder="rtsp://user:password@192.168.1.50:554/stream1",
                        help="RTSP, RTMP or HTTP(S) stream. Enter 0 to use this computer's webcam.")
    name = st.text_input("Camera name", "CCTV 1")
    c1, c2, _ = st.columns([1, 1, 4])
    start, stop = c1.button("Start", type="primary"), c2.button("Stop")
    st.session_state.setdefault("live", None)
    st.session_state.setdefault("live_events", [])
    if stop:
        st.session_state.live = None
    if start:
        src = parse_source(url)
        if src is None:
            st.error("Enter a link starting with rtsp://, rtmp://, http:// or https://, or 0 for a webcam.")
        else:
            st.session_state.live, st.session_state.live_events = {"src": src, "name": name}, []
    if st.session_state.live:
        shown = re.sub(r"//[^@/]+@", "//***@", str(st.session_state.live["src"]))
        st.caption(f"Source: {shown}")
        run_live(st.session_state.live["src"], st.session_state.live["name"])
    elif st.session_state.live_events:
        st.markdown("**Events from the last session**")
        render_events(st.empty(), st.session_state.live_events)
