# dumps one frame + ROI overlay + raw/prep crops (no GUI)
# flake8: noqa

# testScripts/dumpHudDebug.py
import json
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

from screenCapture.screenCapturer import ScreenCapturer  

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

SETTINGS = "defaultSettings.json"
OUTDIR = Path("debugOut"); OUTDIR.mkdir(exist_ok=True)

# Draw these if present in settings["roi"]
ROI_KEYS = [
    "blindTarget", "roundScore", "score",          # supports either
    "chips", "multiplier",
    "hands", "handsLeft",                          # supports either
    "discardsRemaining", "discardsLeft",           # supports either
    "money", "ante", "round",
    "cardsRemaining",
    "playHand", "discardHand",
]

def draw_box(img, roi, color=(0,255,0), label=""):
    h, w = img.shape[:2]
    if "x_px" in roi:
        x, y, ww, hh = int(roi["x_px"]), int(roi["y_px"]), int(roi["w_px"]), int(roi["h_px"])
    else:
        x = int(roi["x"] * w); y = int(roi["y"] * h)
        ww = int(roi["w"] * w); hh = int(roi["h"] * h)
    cv2.rectangle(img, (x,y), (x+ww,y+hh), color, 2)
    if label:
        cv2.putText(img, label, (x, max(0, y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    cfg = json.loads(Path(SETTINGS).read_text(encoding="utf-8"))

    cap = ScreenCapturer()               # whole screen (or pass your monitor/region)
    frame = cap.capture()                # BGR

    h, w = frame.shape[:2]
    print(f"Frame size: {w} x {h}")

    # Save raw frame
    raw_path = OUTDIR / "frame.png"
    cv2.imwrite(str(raw_path), frame)

    # Draw boxes
    vis = frame.copy()
    roi_cfg = cfg.get("roi", {})

    for key in ROI_KEYS:
        if key in roi_cfg:
            draw_box(vis, roi_cfg[key], (0,255,0), key)

    # Draw handBand if present
    if "handBand" in cfg:
        draw_box(vis, cfg["handBand"], (255,0,0), "handBand")

    out_path = OUTDIR / "frame_with_boxes.png"
    cv2.imwrite(str(out_path), vis)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
