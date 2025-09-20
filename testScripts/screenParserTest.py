# flake8: noqa
"""
Live HUD parser smoke test (matches your new labels).
- Default: headless (no popups); prints parsed state every tick.
- Optional: --show to see a live window with ROI overlays.
Controls: in the window, press 'q' to quit. In headless, Ctrl+C to stop.
"""

import os, sys, json, time, argparse
from pathlib import Path

# allow running directly: python .\testScripts\screenParserTest.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2  # noqa: E402
from screenCapture.screenCapturer import ScreenCapturer  # noqa: E402
from gameExtractor.gameStateParser import GameStateParser  # noqa: E402


ROI_KEYS = [
    "blindTarget",
    "roundScore", "score",               # supports either
    "chips", "multiplier",
    "hands", "handsLeft",                # supports either
    "discardsRemaining", "discardsLeft", # supports either
    "money", "ante", "round",
    "cardsRemaining",
    "playHand", "discardHand",
]

def draw_box(img, roi, color=(0, 255, 0), label=""):
    h, w = img.shape[:2]
    if "x_px" in roi:
        x, y, ww, hh = int(roi["x_px"]), int(roi["y_px"]), int(roi["w_px"]), int(roi["h_px"])
    else:
        x = int(roi["x"] * w); y = int(roi["y"] * h)
        ww = int(roi["w"] * w); hh = int(roi["h"] * h)
    cv2.rectangle(img, (x, y), (x + ww, y + hh), color, 2)
    if label:
        cv2.putText(img, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def overlay(frame, cfg):
    vis = frame.copy()
    roi = cfg.get("roi", {})
    for key in ROI_KEYS:
        if key in roi:
            draw_box(vis, roi[key], (0, 255, 0), key)
    if "handBand" in cfg:
        draw_box(vis, cfg["handBand"], (255, 0, 0), "handBand")
    return vis

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--settings", default="defaultSettings.json", help="Path to settings JSON")
    ap.add_argument("--fps", type=float, default=4.0, help="Poll rate")
    ap.add_argument("--show", action="store_true", help="Show live window with boxes")
    args = ap.parse_args()

    settings_path = Path(args.settings)
    cfg = json.loads(settings_path.read_text(encoding="utf-8"))
    print(f"[INFO] Using settings: {settings_path}")

    cap = ScreenCapturer()                       # whole screen capture
    parser = GameStateParser(settingsPath=str(settings_path))  # loads same config

    delay = 1.0 / max(0.5, args.fps)
    win_name = "HUD Live" if args.show else None

    try:
        while True:
            frame = cap.capture()                # BGR
            state = parser.parseFrame(frame, debugBoxes=False)

            # pretty print a compact summary
            summary = {
                "blindTarget": state.get("blindTarget"),
                "roundScore": state.get("roundScore"),
                "chips": state.get("chips"),
                "multiplier": state.get("multiplier"),
                "hands": state.get("hands"),
                "discardsRemaining": state.get("discardsRemaining"),
                "money": state.get("money"),
                "ante": state.get("ante"),
                "round": state.get("round"),
                "cardsRemaining": state.get("cardsRemaining"),
                "buttons": state.get("buttons"),
            }
            print(summary)

            if args.show:
                vis = overlay(frame, cfg)
                cv2.imshow("HUD Live", vis)
                k = cv2.waitKey(int(delay * 1000)) & 0xFF
                if k == ord('q'):
                    break
            else:
                time.sleep(delay)

    except KeyboardInterrupt:
        pass
    finally:
        if args.show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
