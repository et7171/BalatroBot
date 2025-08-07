"""
Captures frames from the Balatro game window.
Acts as an adapter for MSS, OpenCV, or RAM reading tools.
"""

# flake8: noqa


import mss
import numpy as np
import cv2


class ScreenCapturer:
    def __init__(self, region=None):
        # Region format: {"top": Y, "left": X, "width": W, "height": H}
        self.region = region

    def capture(self):
        with mss.mss() as sct:
            screenshot = (
                sct.grab(self.region) if self.region else sct.grab(sct.monitors[1])
            )
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
