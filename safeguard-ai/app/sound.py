"""Play the alarm through THIS computer's speakers (used by the always-on home service)."""
import io
import os
import shutil
import subprocess
import sys
import tempfile
import wave

import numpy as np

_proc = None


def _wav_path(secs: int) -> str:
    path = os.path.join(tempfile.gettempdir(), f"safeguard_siren_{secs}.wav")
    if not os.path.exists(path):
        sr = 16000
        t = np.linspace(0, secs, int(sr * secs), False)
        freq = np.where((t * 3) % 1 < 0.5, 960, 700)  # two-tone siren
        pcm = (0.45 * np.sin(2 * np.pi * np.cumsum(freq) / sr) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        with open(path, "wb") as f:
            f.write(buf.getvalue())
    return path


def stop_local():
    global _proc
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(None, winsound.SND_PURGE)
        elif _proc is not None:
            _proc.terminate()
    except Exception:
        pass


def play_local(secs) -> bool:
    """Start the siren for `secs` seconds. Returns False if this machine has no way to play sound."""
    global _proc
    try:
        path = _wav_path(max(1, int(secs)))
        stop_local()
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            return True
        for cmd in (["afplay", path], ["paplay", path], ["aplay", "-q", path]):  # Mac, Linux
            if shutil.which(cmd[0]):
                _proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
    except Exception:
        pass
    return False
