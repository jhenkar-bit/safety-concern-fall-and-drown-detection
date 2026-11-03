"""Train the optional LSTM fall verifier on your own labelled videos.

Layout:   videos/adl/*.mp4          (daily activities: walk, sit, bend, crouch, lie on sofa...)
          videos/fall/*.mp4         (falls)
          videos/labels.csv         header: video,fall_start_s,fall_end_s   (one row per fall video)
Run:      python -m train.train_fall_lstm --videos videos --out weights/fall_lstm.pt
Features are produced by the SAME code used at inference (FallDetector), so train/serve match.
Datasets to start from: Le2i, UR Fall (URFD), UP-Fall, CAUCAFall, Multiple Cameras Fall.
"""
import argparse
import csv
import glob
import os
import random
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from app.detectors.fall import WINDOW, FallDetector, build_lstm

CFG = SimpleNamespace(down_angle=55, up_angle=35, still_motion=0.25, confirm_s=3.0,
                      candidate_timeout_s=12, long_lie_s=30, drop_vy=1.0)


def windows_from_video(model, path, label_range, target_fps=10):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps / target_fps))
    det, X, Y, i = FallDetector(CFG), [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        t = i / fps
        r = model.track(fr, persist=True, classes=[0], conf=0.4, verbose=False)[0]
        b = r.boxes
        if b is None or b.id is None or not len(b):
            det.update([], np.zeros((0, 4)), None, t, fr)
            continue
        ids, kp = b.id.int().tolist(), r.keypoints.data.cpu().numpy()
        det.update(ids, b.xyxy.cpu().numpy(), kp, t, fr)
        for tid in ids:
            f = det.states[tid].feats
            if len(f) == WINDOW and i % (step * 5) == 0:  # stride 5 frames
                X.append(np.array(f, np.float32))
                y = 0
                if label_range and label_range[0] <= t <= label_range[1] + 3.0:
                    y = 1
                Y.append(y)
    cap.release()
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="videos")
    ap.add_argument("--out", default="weights/fall_lstm.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pose", default="yolo11n-pose.pt")
    a = ap.parse_args()
    labels = {}
    with open(os.path.join(a.videos, "labels.csv")) as f:
        for r in csv.DictReader(f):
            labels[r["video"]] = (float(r["fall_start_s"]), float(r["fall_end_s"]))
    model = YOLO(a.pose)
    vids = sorted(glob.glob(f"{a.videos}/adl/*") + glob.glob(f"{a.videos}/fall/*"))
    random.seed(0)
    random.shuffle(vids)
    split = int(0.8 * len(vids))  # split BY VIDEO to avoid leakage between train and val
    data = {"train": ([], []), "val": ([], [])}
    for n, v in enumerate(vids):
        X, Y = windows_from_video(model, v, labels.get(os.path.basename(v)))
        k = "train" if n < split else "val"
        data[k][0].extend(X)
        data[k][1].extend(Y)
        print(f"[{n+1}/{len(vids)}] {os.path.basename(v)} -> {len(X)} windows ({k})")
    Xt, Yt = map(np.array, data["train"])
    Xv, Yv = map(np.array, data["val"])
    net, opt = build_lstm(), None
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    pw = torch.tensor((Yt == 0).sum() / max((Yt == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt_t, Yt_t = torch.tensor(Xt), torch.tensor(Yt, dtype=torch.float32)
    best = -1.0
    for ep in range(a.epochs):
        net.train()
        perm = torch.randperm(len(Xt_t))
        for j in range(0, len(perm), 64):
            idx = perm[j:j + 64]
            opt.zero_grad()
            loss = loss_fn(net(Xt_t[idx]), Yt_t[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(net(torch.tensor(Xv))) > 0.5).numpy().astype(int)
        tp = int(((pred == 1) & (Yv == 1)).sum()); fp = int(((pred == 1) & (Yv == 0)).sum()); fn = int(((pred == 0) & (Yv == 1)).sum())
        prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        print(f"epoch {ep+1:02d} loss {loss.item():.3f} val precision {prec:.2f} recall {rec:.2f} f1 {f1:.2f}")
        if f1 > best:
            best = f1
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            torch.save(net.state_dict(), a.out)
    print(f"saved best model (val f1 {best:.2f}) -> {a.out}. Set fall.use_lstm: true in config.yaml")


if __name__ == "__main__":
    main()
