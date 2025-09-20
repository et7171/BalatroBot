# gameExtractor/gameStateParser.py
"""
Parse the Balatro HUD into a structured state.
ROIs can be percent (x,y,w,h) or pixels (x_px,y_px,w_px,h_px).
"""
# flake8: noqa

import re
import json
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import cv2
import numpy as np
import pytesseract
from dataclasses import dataclass


# ---------------- datatypes ----------------

@dataclass
class DetectedCard:
    rank: Optional[str]
    suit: Optional[str]
    rank_conf: float
    suit_conf: float


# ---------------- parser ----------------

class GameStateParser:

    def __init__(
        self,
        settingsPath: str = "defaultSettings.json",
        tesseractPath: str = r"H:\TesseractOCR\tesseract.exe",
        templateDir: str = "assets/templates",  # ranks/ and suits/ inside here
    ):
        self.settingsPath = settingsPath
        self.settings = json.loads(Path(settingsPath).read_text(encoding="utf-8"))

        pytesseract.pytesseract.tesseract_cmd = tesseractPath

        self.templateThreshold = float(self.settings.get("templateThreshold", 0.74))
        self.rankTemplates = self._loadTemplates(Path(templateDir) / "ranks")
        self.suitTemplates  = self._loadTemplates(Path(templateDir) / "suits")

        # smoothing cache for noisy OCR
        self._last = {"roundScore": None, "chips": None, "multiplier": None}

    # ---------- template helpers ----------

    def _loadTemplates(self, folder: Path) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {}
        if not folder.exists():
            return d
        for p in folder.glob("*.*"):
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            d[p.stem] = img
        return d

    def _bestTemplate(self, grayCrop: np.ndarray, bank: Dict[str, np.ndarray]) -> Tuple[Optional[str], float]:
        best_label, best_score = None, -1.0
        h, w = grayCrop.shape[:2]
        if max(h, w) < 64:
            grayCrop = cv2.resize(grayCrop, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        for label, tmpl in bank.items():
            th, tw = tmpl.shape[:2]
            if h < th or w < tw:
                scale = max(th / max(h, 1), tw / max(w, 1)) * 1.05
                if 1.0 < scale <= 3.0:
                    scaled = cv2.resize(grayCrop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    res = cv2.matchTemplate(scaled, tmpl, cv2.TM_CCOEFF_NORMED)
                else:
                    continue
            else:
                res = cv2.matchTemplate(grayCrop, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max()) if res.size else -1.0
            if score > best_score:
                best_score, best_label = score, label
        if best_score < self.templateThreshold:
            return None, best_score
        return best_label, best_score

    # ---------- ROI / crops ----------

    def _pxRoi(self, frame: np.ndarray, roiCfg: Dict) -> Tuple[int, int, int, int]:
        """Supports px or percent ROI entries."""
        h, w = frame.shape[:2]
        if "x_px" in roiCfg:
            x, y, ww, hh = int(roiCfg["x_px"]), int(roiCfg["y_px"]), int(roiCfg["w_px"]), int(roiCfg["h_px"])
        else:
            x = int(roiCfg["x"] * w)
            y = int(roiCfg["y"] * h)
            ww = int(roiCfg["w"] * w)
            hh = int(roiCfg["h"] * h)
        x = max(0, min(x, w - 1)); y = max(0, min(y, h - 1))
        ww = max(1, min(ww, w - x)); hh = max(1, min(hh, h - y))
        return x, y, ww, hh

    def _cropPercentRoi(self, frame: np.ndarray, roiCfg: Dict) -> np.ndarray:
        x, y, ww, hh = self._pxRoi(frame, roiCfg)
        return frame[y:y + hh, x:x + ww].copy()

    # ---------- OCR prep ----------

    def _prepDigits(self, imgBgr: np.ndarray) -> np.ndarray:
        """General purpose digit prep (good for multi-digit)."""
        gray = cv2.cvtColor(imgBgr, cv2.COLOR_BGR2GRAY)
        scale = 2.0 if max(gray.shape[:2]) < 200 else 1.5
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if np.mean(gray) < 110:
            gray = 255 - gray
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        if gray.var() < 400:
            th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10)
        else:
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
        return th

    def _prepDigits_keepThin(self, imgBgr: np.ndarray) -> np.ndarray:
        """Preprocess but keep thin strokes (so '/' doesn't vanish)."""
        gray = cv2.cvtColor(imgBgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 5)
        return th

    def _postFixDigits(self, s: str) -> str:
        return (s.replace("O", "0").replace("o", "0")
                 .replace("l", "1").replace("I", "1")
                 .replace("S", "5").replace("B", "8"))

    # ---------- OCR engines ----------

    def _ocrDigits(self, imgBgr: np.ndarray, allowX: bool = False) -> Tuple[Optional[int], float]:
        """
        Robust OCR for numbers (with optional 'x').
        Tries multiple PSMs on normal + inverted to catch lonely '0' etc.
        """
        def run_data(img, psm, wl):
            cfg = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={wl} -c preserve_interword_spaces=1'
            data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
            texts, confs = [], []
            for txt, conf in zip(data.get("text", []), data.get("conf", [])):
                if txt and txt.strip():
                    texts.append(txt)
                    try: confs.append(float(conf))
                    except: pass
            return "".join(texts), (float(np.mean(confs)) if confs else 0.0)

        wl = "0123456789xX" if allowX else "0123456789"
        prep = self._prepDigits(imgBgr)

        for psm in (7, 8, 6, 10, 13):
            text, conf = run_data(prep, psm, wl)
            text = self._postFixDigits(text)
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                return int(digits), conf

        inv = 255 - prep
        for psm in (10, 7, 8, 6, 13):
            text, conf = run_data(inv, psm, wl)
            text = self._postFixDigits(text)
            digits = "".join(ch for ch in text if ch.isdigit())
            if digits:
                return int(digits), conf

        return None, 0.0

    def _ocrTwoNumbers(self, imgBgr: np.ndarray) -> Tuple[Tuple[Optional[int], Optional[int]], float]:
        """
        Extract up to two numbers left->right even if '/' is missing.
        """
        def run(img, psm):
            cfg = r'--oem 3 --psm %d -c tessedit_char_whitelist=0123456789' % psm
            data = pytesseract.image_to_data(img, config=cfg, output_type=pytesseract.Output.DICT)
            items = []
            for i in range(len(data["text"])):
                t = (data["text"][i] or "").strip()
                if t.isdigit():
                    try: conf = float(data["conf"][i])
                    except: conf = 0.0
                    x = int(data["left"][i])
                    items.append((x, int(t), conf))
            items.sort(key=lambda z: z[0])
            nums  = [n for _, n, _ in items]
            confs = [c for _, _, c in items]
            if not nums:
                return (None, None), 0.0
            if len(nums) == 1:
                return (nums[0], None), (confs[0] if confs else 0.0)
            return (nums[0], nums[1]), (sum(confs[:2]) / max(1, len(confs[:2])))

        prep = self._prepDigits_keepThin(imgBgr)
        for psm in (7, 6, 8, 13, 3, 10):
            vals, conf = run(prep, psm)
            if vals != (None, None):
                return vals, conf
        inv = 255 - prep
        for psm in (7, 6, 8, 13, 3, 10):
            vals, conf = run(inv, psm)
            if vals != (None, None):
                return vals, conf
        return (None, None), 0.0

    def _ocrNumSlash(self, imgBgr: np.ndarray) -> Tuple[Tuple[Optional[int], Optional[int]], float]:
        """Prefer 'a/b', but if slash disappears, fall back to two-number extraction."""
        img = self._prepDigits_keepThin(imgBgr)
        cfg = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789/'
        s = pytesseract.image_to_string(img, config=cfg).strip()

        m = re.search(r'(\d+)\s*/\s*(\d+)', s)
        if m:
            return (int(m.group(1)), int(m.group(2))), 90.0

        nums = re.findall(r'\d+', s)
        if len(nums) >= 2:
            return (int(nums[0]), int(nums[1])), 70.0
        if len(nums) == 1:
            return (int(nums[0]), None), 50.0

        return self._ocrTwoNumbers(imgBgr)

    # ---------- public field readers ----------

    def readBlindTarget(self, frame, debug=False):
        roi = self.settings["roi"].get("blindTarget")
        if not roi: return (None, 0.0)
        crop = self._cropPercentRoi(frame, roi)
        if debug: self._showDebug("blindTarget", crop)
        return self._ocrDigits(crop, allowX=False)

    def readRoundScore(self, frame, debug=False):
        roi = self.settings["roi"].get("roundScore", self.settings["roi"].get("score"))
        if not roi: return (None, 0.0)
        crop = self._cropPercentRoi(frame, roi)
        if debug: self._showDebug("roundScore", crop)
        val, conf = self._ocrDigits(crop, allowX=False)
        if val is not None:
            return val, conf
        # try right-side of the bar where the number sits
        h, w = crop.shape[:2]
        sub = crop[:, int(0.57 * w):]
        return self._ocrDigits(sub, allowX=False)

    def readChips(self, frame, debug=False):
        roi = self.settings["roi"].get("chips")
        if not roi: return (None, 0.0)
        crop = self._cropPercentRoi(frame, roi)
        if debug: self._showDebug("chips", crop)
        return self._ocrDigits(crop, allowX=False)

    def readMultiplier(self, frame, debug=False):
        roi = self.settings["roi"].get("multiplier")
        if not roi: return (None, 0.0)
        crop = self._cropPercentRoi(frame, roi)
        if debug: self._showDebug("multiplier", crop)
        return self._ocrDigits(crop, allowX=True)

    def readHands(self, frame, debug=False):
        roi = self.settings["roi"].get("hands", self.settings["roi"].get("handsLeft"))
        if not roi: return (None, 0.0)
        crop = self._cropPercentRoi(frame, roi)
        if debug: self._showDebug("hands", crop)
        return self._ocrDigits(crop, allowX=False)

    def readDiscardsRemaining(self, frame, debug=False):
        roi = self.settings["roi"].get("discardsRemaining", self.settings["roi"].get("discardsLeft"))
        if not roi: return (None, 0.0)
        crop = self._cropPercentRoi(frame, roi)
        if debug: self._showDebug("discardsRemaining", crop)
        return self._ocrDigits(crop, allowX=False)

    def readMoney(self, frame, debug=False):
        roi = self.settings["roi"].get("money")
        if not roi: return (None, 0.0)
        crop = self._cropPercentRoi(frame, roi)
        if debug: self._showDebug("money", crop)
        return self._ocrDigits(crop, allowX=False)

    def readAnte(self, frame, debug=False):
        roi = self.settings["roi"].get("ante")
        if not roi: return ((None, None), 0.0)
        crop = self._cropPercentRoi(frame, roi)
        if debug: self._showDebug("ante", crop)
        return self._ocrNumSlash(crop)

    def readRound(self, frame, debug=False):
        roi = self.settings["roi"].get("round")
        if not roi: return (None, 0.0)
        crop = self._cropPercentRoi(frame, roi)
        if debug: self._showDebug("round", crop)
        return self._ocrDigits(crop, allowX=False)

    def readCardsRemaining(self, frame, debug=False):
        roi = self.settings["roi"].get("cardsRemaining")
        if not roi: return ((None, None), 0.0)
        crop = self._cropPercentRoi(frame, roi)
        if debug: self._showDebug("cardsRemaining", crop)
        return self._ocrNumSlash(crop)

    # ---------- button state ----------

    def _buttonActive(self, cropBgr) -> bool:
        hsv = cv2.cvtColor(cropBgr, cv2.COLOR_BGR2HSV)
        sat = float(np.mean(hsv[:, :, 1])); val = float(np.mean(hsv[:, :, 2]))
        return (sat > 40) and (val > 80)

    # ---------- dynamic hand detection ----------

    def _nms(self, boxes, iou_thresh=0.3):
        if not boxes: return []
        areas = [w*h for (_,_,w,h) in boxes]
        idxs  = list(range(len(boxes)))
        idxs.sort(key=lambda i: areas[i], reverse=True)
        keep=[]
        def iou(a,b):
            ax,ay,aw,ah=a; bx,by,bw,bh=b
            x1=max(ax,bx); y1=max(ay,by)
            x2=min(ax+aw, bx+bw); y2=min(ay+ah, by+bh)
            iw=max(0,x2-x1); ih=max(0,y2-y1)
            inter=iw*ih
            if inter<=0: return 0.0
            ua=aw*ah + bw*bh - inter
            return inter/ua
        while idxs:
            i=idxs.pop(0); keep.append(i)
            idxs=[j for j in idxs if iou(boxes[i], boxes[j]) < iou_thresh]
        return [boxes[i] for i in keep]

    def _detectHandBoxes(self, frame: np.ndarray):
        if "handBand" not in self.settings:
            return []
        cd = self.settings.get("cardDetect", {})
        min_h_rel=float(cd.get("min_h_rel",0.12))
        max_h_rel=float(cd.get("max_h_rel",0.30))
        aspect_min=float(cd.get("aspect_min",0.60))
        aspect_max=float(cd.get("aspect_max",0.80))
        nms_iou=float(cd.get("nms_iou",0.30))

        H, W = frame.shape[:2]
        fx, fy, fw, fh = self._pxRoi(frame, self.settings["handBand"])
        band = frame[fy:fy+fh, fx:fx+fw].copy()

        hsv  = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0,0,180), (180,70,255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8), iterations=2)

        edges = cv2.Canny(cv2.cvtColor(band, cv2.COLOR_BGR2GRAY), 80, 160)
        edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)

        combo = cv2.bitwise_or(mask, edges)
        contours, _ = cv2.findContours(combo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes=[]
        min_h_px=int(min_h_rel*H); max_h_px=int(max_h_rel*H)
        for c in contours:
            x,y,ww,hh = cv2.boundingRect(c)
            X, Y = fx + x, fy + y
            if hh < min_h_px or hh > max_h_px: continue
            ar = ww/float(hh+1e-6)
            if ar < aspect_min or ar > aspect_max: continue
            pad = max(2, int(0.01*H))
            X=max(0,X-pad); Y=max(0,Y-pad)
            ww=min(W-X, ww+2*pad); hh=min(H-Y, hh+2*pad)
            boxes.append((X,Y,ww,hh))

        boxes = self._nms(boxes, nms_iou)
        boxes.sort(key=lambda b: b[0])
        return boxes

    def _prepSymbol(self, imgBgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(imgBgr, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) < 110: gray = 255 - gray
        gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return gray

    def _detectCard_from_crop(self, cardBgr: np.ndarray) -> DetectedCard:
        if "cardRoi" not in self.settings:
            return DetectedCard(None, None, 0.0, 0.0)
        cardCfg = self.settings["cardRoi"]
        ch, cw = cardBgr.shape[:2]
        rx=int(cardCfg["rank"]["x"]*cw); ry=int(cardCfg["rank"]["y"]*ch)
        rw=int(cardCfg["rank"]["w"]*cw); rh=int(cardCfg["rank"]["h"]*ch)
        sx=int(cardCfg["suit"]["x"]*cw); sy=int(cardCfg["suit"]["y"]*ch)
        sw=int(cardCfg["suit"]["w"]*cw); sh=int(cardCfg["suit"]["h"]*ch)

        rankCrop = cardBgr[ry:ry+rh, rx:rx+rw].copy()
        suitCrop = cardBgr[sy:sy+sh, sx:sx+sw].copy()

        rankGray = self._prepSymbol(rankCrop)
        suitGray = self._prepSymbol(suitCrop)

        rankLabel, rankScore = self._bestTemplate(rankGray, self.rankTemplates)
        suitLabel, suitScore = self._bestTemplate(suitGray, self.suitTemplates)
        if rankLabel == "10":
            rankLabel = "T"

        return DetectedCard(
            rank=rankLabel,
            suit=suitLabel,
            rank_conf=float(rankScore or 0.0),
            suit_conf=float(suitScore or 0.0),
        )

    # ---------- main entry ----------

    def parseFrame(self, frame: np.ndarray, debugBoxes: bool = False) -> Dict:
        blindTarget, blindConf   = self.readBlindTarget(frame)
        roundScore, scoreConf    = self.readRoundScore(frame)
        chips, chipsConf         = self.readChips(frame)
        mult, multConf           = self.readMultiplier(frame)

        def smooth(key, val, conf, thresh=60.0):
            if val is None or conf < thresh:
                return self._last.get(key, None)
            self._last[key] = val
            return val

        state = {
            "blindTarget": blindTarget,
            "roundScore":  smooth("roundScore", roundScore, scoreConf),
            "chips":       smooth("chips", chips, chipsConf),
            "multiplier":  smooth("multiplier", mult, multConf),
            "confidence": {
                "blindTarget": blindConf,
                "roundScore": scoreConf,
                "chips": chipsConf,
                "multiplier": multConf,
            },
        }

        roi = self.settings.get("roi", {})

        if "hands" in roi or "handsLeft" in roi:
            v, c = self.readHands(frame); state["hands"] = v; state["confidence"]["hands"] = c
        if "discardsRemaining" in roi or "discardsLeft" in roi:
            v, c = self.readDiscardsRemaining(frame); state["discardsRemaining"] = v; state["confidence"]["discardsRemaining"] = c
        if "money" in roi:
            v, c = self.readMoney(frame); state["money"] = v; state["confidence"]["money"] = c
        if "ante" in roi:
            v, c = self.readAnte(frame); state["ante"] = v; state["confidence"]["ante"] = c
        if "round" in roi:
            v, c = self.readRound(frame); state["round"] = v; state["confidence"]["round"] = c
        if "cardsRemaining" in roi:
            v, c = self.readCardsRemaining(frame); state["cardsRemaining"] = v; state["confidence"]["cardsRemaining"] = c

        if "playHand" in roi and "discardHand" in roi:
            playCrop    = self._cropPercentRoi(frame, roi["playHand"])
            discardCrop = self._cropPercentRoi(frame, roi["discardHand"])
            state["buttons"] = {
                "playHandActive":    self._buttonActive(playCrop),
                "discardHandActive": self._buttonActive(discardCrop),
            }

        # dynamic hand parsing if configured
        hand_list = []
        if "handBand" in self.settings and "cardRoi" in self.settings:
            for (x, y, w, h) in self._detectHandBoxes(frame):
                cardCrop = frame[y:y+h, x:x+w]
                card = self._detectCard_from_crop(cardCrop)
                hand_list.append(card.__dict__)
        if hand_list:
            state["hand"] = hand_list

        if debugBoxes:
            self._drawRoiBoxes(frame.copy())

        return state

    # ---------- debug helpers ----------

    def _drawRoiBoxes(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        for key, roi in self.settings.get("roi", {}).items():
            x, y, ww, hh = self._pxRoi(frame, roi)
            cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
            cv2.putText(frame, key, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        if "handBand" in self.settings:
            x, y, ww, hh = self._pxRoi(frame, self.settings["handBand"])
            cv2.rectangle(frame, (x, y), (x + ww, y + hh), (255, 0, 0), 2)
            cv2.putText(frame, "handBand", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        cv2.imshow("roiBoxes", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def _showDebug(self, name: str, cropBgr: np.ndarray) -> None:
        prepDigits = self._prepDigits(cropBgr)
        cv2.imshow(f"{name}-raw", cropBgr)
        cv2.imshow(f"{name}-prepDigits", prepDigits)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
