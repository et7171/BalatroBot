# -*- coding: utf-8 -*-
# flake8: noqa
import os, sys, json
from pathlib import Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
from screenCapture.screenCapturer import ScreenCapturer

def px_roi(frame, roi):
    h, w = frame.shape[:2]
    if "x_px" in roi:
        x, y, ww, hh = int(roi["x_px"]), int(roi["y_px"]), int(roi["w_px"]), int(roi["h_px"])
    else:
        x = int(roi["x"] * w); y = int(roi["y"] * h)
        ww = int(roi["w"] * w); hh = int(roi["h"] * h)
    x = max(0, min(x, w-1)); y = max(0, min(y, h-1))
    ww = max(1, min(ww, w - x)); hh = max(1, min(hh, h - y))
    return x, y, ww, hh

def focus(crop, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
    h, w = crop.shape[:2]
    X0, X1 = int(x0*w), int(x1*w)
    Y0, Y1 = int(y0*h), int(y1*h)
    X0 = max(0, min(X0, w-1)); Y0 = max(0, min(Y0, h-1))
    X1 = max(X0+1, min(X1, w)); Y1 = max(Y0+1, min(Y1, h))
    return crop[Y0:Y1, X0:X1].copy()

# same focuses as the parser uses
FOCUS = {
    "roundScore":      [(0.65, 0.98, 0.15, 0.85), (0.72, 0.98, 0.25, 0.80)],
    "chips":           [(0.05, 0.40, 0.10, 0.90)],
    "multiplier":      [(0.60, 0.95, 0.10, 0.90)],
    "hands":           [(0.25, 0.80, 0.10, 0.90)],
    "discardsRemaining":[(0.25, 0.80, 0.10, 0.90)],
    "money":           [(0.20, 0.85, 0.15, 0.90)],
    "ante":            [(0.45, 0.98, 0.10, 0.90)],
    "round":           [(0.40, 0.90, 0.10, 0.90)],
    "cardsRemaining":  [(0.10, 0.95, 0.10, 0.90)],
    # "blindTarget" uses full crop; no focus needed
}

def main(settings_path="defaultSettings.json", out_dir="debugOut"):
    cfg = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    os.makedirs(out_dir, exist_ok=True)

    cap = ScreenCapturer()
    frame = cap.capture()

    roi_cfg = cfg.get("roi", {})
    for key, roi in roi_cfg.items():
        x, y, w, h = px_roi(frame, roi)
        crop = frame[y:y+h, x:x+w].copy()
        cv2.imwrite(os.path.join(out_dir, f"{key}_raw.png"), crop)

        if key in FOCUS:
            for i, box in enumerate(FOCUS[key]):
                fc = focus(crop, *box)
                cv2.imwrite(os.path.join(out_dir, f"{key}_focus{i+1}.png"), fc)

    print(f"Wrote crops to {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="defaultSettings.json")
    ap.add_argument("--out", default="debugOut")
    args = ap.parse_args()
    main(args.settings, args.out)
