# flake8: noqa

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mss
import cv2
import numpy as np 

OUT = Path("debugOut"); OUT.mkdir(exist_ok=True)

with mss.mss() as sct:
    for i, mon in enumerate(sct.monitors):
        shot = sct.grab(mon)
        bgr = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
        name = "monitor_0_all.png" if i == 0 else f"monitor_{i}.png"
        cv2.imwrite(str(OUT / name), bgr)
        print(f"Saved {name}  size={mon['width']}x{mon['height']}  top={mon['top']} left={mon['left']}")

print("Wrote snapshots to:", OUT.resolve())