"""Downloads the YOLO models into this folder (run once, needs internet).
    python download_models.py                 -> the usual three
    python download_models.py yolo11l-pose.pt -> only the one you name
"""
import os
import sys

from ultralytics import YOLO

MODELS = sys.argv[1:] or ["yolo11m-pose.pt", "yolo11s-pose.pt", "yolo11n.pt"]
for name in MODELS:
    print("Downloading", name, "...")
    YOLO(name)
print("Done. The files are in", os.getcwd())
