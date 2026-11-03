import os
from types import SimpleNamespace
import yaml

try:  # load secrets from a .env file next to config.yaml, if present
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _ns(x):
    if isinstance(x, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_ns(i) for i in x]
    return x


def camera_ready(cam) -> bool:
    """False when the camera link is empty or still an unfilled ${VARIABLE} (e.g. HOME_CAM_URL not set yet)."""
    src = str(getattr(cam, "source", "")).strip()
    return src != "" and "${" not in src


def load_config(path: str | None = None):
    path = path or os.getenv("CONFIG_PATH", "config.yaml")
    with open(path) as f:
        text = os.path.expandvars(f.read())  # allows ${ENV_VAR} (e.g. RTSP credentials)
    return _ns(yaml.safe_load(text))
