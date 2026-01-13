#!/usr/bin/env python3
# printing_detection.py
#
# Drop-in detector module for ui_recorder_v.py
# Provides: PrintingDetector.update(frame_bgr) -> (vis_bgr, state, debug)

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np
from ultralytics import YOLO


# -----------------------------
# Small helpers
# -----------------------------
def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(int(v), int(hi)))


def crop_square(cx: int, cy: int, size: int, W: int, H: int) -> Tuple[int, int, int, int]:
    half = int(size) // 2
    x1 = clamp(cx - half, 0, W - 1)
    y1 = clamp(cy - half, 0, H - 1)
    x2 = clamp(cx + half, 0, W - 1)
    y2 = clamp(cy + half, 0, H - 1)
    return int(x1), int(y1), int(x2), int(y2)


def count_mask_pixels(mask_u8: np.ndarray, box_xyxy: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box_xyxy
    if x2 <= x1 or y2 <= y1:
        return 0
    # mask is 0/1
    return int(mask_u8[y1:y2, x1:x2].sum())


def safe_put_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.7, thickness: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def class_color(name: str) -> Tuple[int, int, int]:
    # deterministic color from class name
    h = abs(hash(name)) % (256 * 256 * 256)
    b = (h) & 255
    g = (h >> 8) & 255
    r = (h >> 16) & 255
    b = max(int(b), 60)
    g = max(int(g), 60)
    r = max(int(r), 60)
    return (b, g, r)  # BGR


def overlay_masks(frame_bgr: np.ndarray, masks_dict: Dict[str, np.ndarray], classes, alpha: float = 0.6) -> np.ndarray:
    overlay = frame_bgr.copy()
    for cname in classes:
        m = masks_dict.get(cname, None)
        if m is None:
            continue
        overlay[m == 1] = class_color(cname)
    return cv2.addWeighted(overlay, float(alpha), frame_bgr, 1.0 - float(alpha), 0.0)


def get_best_box_of_class(boxes, names: Dict[int, str], class_name: str):
    if boxes is None or len(boxes) == 0:
        return None
    cls = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()

    best = None
    for i in range(len(cls)):
        cname = names[int(cls[i])]
        if cname != class_name:
            continue
        c = float(confs[i])
        x1, y1, x2, y2 = xyxy[i]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cand = (c, cx, cy, (int(x1), int(y1), int(x2), int(y2)))
        if best is None or cand[0] > best[0]:
            best = cand
    return best


def build_class_masks(res, names: Dict[int, str], H: int, W: int) -> Dict[str, np.ndarray]:
    masks_out: Dict[str, np.ndarray] = {}
    if res.masks is None or res.masks.data is None:
        return masks_out
    if res.boxes is None or len(res.boxes) != len(res.masks.data):
        return masks_out

    mdata = res.masks.data.cpu().numpy()  # (n,h,w) float
    mcls = res.boxes.cls.cpu().numpy().astype(int)

    for i in range(len(mdata)):
        cname = names[int(mcls[i])]
        m = (mdata[i] > 0.5).astype(np.uint8)

        if m.shape[0] != H or m.shape[1] != W:
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        if cname not in masks_out:
            masks_out[cname] = m
        else:
            masks_out[cname] = np.maximum(masks_out[cname], m)

    return masks_out


# -----------------------------
# Public API
# -----------------------------
@dataclass
class PrintingState:
    printing: bool
    nozzle_conf: float
    new_area_roi_pct: float
    overlap_tip: float
    on_count: int
    off_count: int


class PrintingDetector:
    """
    UI friendly detector wrapper.

    Call update(frame_bgr) where frame_bgr is a numpy uint8 BGR image.
    Returns:
      vis_bgr: BGR image with mask overlay + ROI boxes + labels
      state: PrintingState
      debug: dict
    """

    def __init__(
        self,
        model_path: str,
        conf: float = 0.35,
        class_nozzle: str = "nozzle",
        class_new: str = "new_deposit",
        overlay_classes = ("nozzle", "new_deposit", "old_deposit"),
        mask_alpha: float = 0.60,
        roi_size: int = 220,
        tip_zone_w: int = 90,
        tip_zone_h: int = 70,
        tip_y_offset: int = 0,
        a_min_px: int = 900,
        o_min: float = 0.08,
        k_on: int = 3,
        k_off: int = 10,
        nozzle_hold_sec: float = 0.5,
    ):
        self.model = YOLO(str(model_path))
        self.names = self.model.names

        self.conf = float(conf)

        self.CLASS_NOZZLE = str(class_nozzle)
        self.CLASS_NEW = str(class_new)
        self.OVERLAY_CLASSES = list(overlay_classes)

        self.MASK_ALPHA = float(mask_alpha)

        self.ROI_SIZE = int(roi_size)
        self.TIP_ZONE_W = int(tip_zone_w)
        self.TIP_ZONE_H = int(tip_zone_h)
        self.TIP_Y_OFFSET = int(tip_y_offset)

        self.A_MIN = int(a_min_px)
        self.O_MIN = float(o_min)
        self.K_ON = int(k_on)
        self.K_OFF = int(k_off)
        self.NOZZLE_HOLD_SEC = float(nozzle_hold_sec)

        # hysteresis
        self.printing = False
        self.on_count = 0
        self.off_count = 0

        # nozzle hold
        self.last_nozzle_time = 0.0
        self.last_nozzle = None

        self._last_vis = None
        self._last_state = PrintingState(False, 0.0, 0.0, 0.0, 0, 0)

    def update(self, frame_bgr: np.ndarray):
        t0 = time.time()

        if frame_bgr is None:
            return None, self._last_state, {"python": "frame=None"}

        H, W = frame_bgr.shape[:2]
        if H < 2 or W < 2:
            return frame_bgr, self._last_state, {"python": "frame too small"}

        res = self.model.predict(frame_bgr, conf=self.conf, verbose=False)[0]

        nozzle = get_best_box_of_class(res.boxes, self.names, self.CLASS_NOZZLE)
        masks = build_class_masks(res, self.names, H, W)
        new_mask = masks.get(self.CLASS_NEW, np.zeros((H, W), dtype=np.uint8))

        vis = overlay_masks(frame_bgr, masks, self.OVERLAY_CLASSES, alpha=self.MASK_ALPHA)

        now = time.time()
        if nozzle is not None:
            self.last_nozzle = nozzle
            self.last_nozzle_time = now

        use_nozzle = None
        if self.last_nozzle is not None and (now - self.last_nozzle_time) <= self.NOZZLE_HOLD_SEC:
            use_nozzle = self.last_nozzle

        nozzle_conf = 0.0
        nx = ny = -1
        new_area_roi = 0
        overlap_tip = 0.0
        new_area_roi_pct = 0.0

        if use_nozzle is not None:
            nozzle_conf, nx, ny, nbbox = use_nozzle

            roi_box = crop_square(nx, ny, self.ROI_SIZE, W, H)

            tx1 = clamp(nx - self.TIP_ZONE_W // 2, 0, W - 1)
            tx2 = clamp(nx + self.TIP_ZONE_W // 2, 0, W - 1)
            ty1 = clamp(ny + self.TIP_Y_OFFSET, 0, H - 1)
            ty2 = clamp(ny + self.TIP_Y_OFFSET + self.TIP_ZONE_H, 0, H - 1)
            tip_box = (int(tx1), int(ty1), int(tx2), int(ty2))

            new_area_roi = count_mask_pixels(new_mask, roi_box)
            roi_area = max(1, (roi_box[2] - roi_box[0]) * (roi_box[3] - roi_box[1]))
            new_area_roi_pct = float(new_area_roi) / float(roi_area)

            tip_area = max(1, (tip_box[2] - tip_box[0]) * (tip_box[3] - tip_box[1]))
            tip_new = count_mask_pixels(new_mask, tip_box)
            overlap_tip = float(tip_new) / float(tip_area)

            is_printing_now = (new_area_roi >= self.A_MIN) and (overlap_tip >= self.O_MIN)

            if is_printing_now:
                self.on_count += 1
                self.off_count = 0
            else:
                self.off_count += 1
                self.on_count = 0

            if (not self.printing) and self.on_count >= self.K_ON:
                self.printing = True
            if self.printing and self.off_count >= self.K_OFF:
                self.printing = False

            # draw helpers
            cv2.rectangle(vis, (roi_box[0], roi_box[1]), (roi_box[2], roi_box[3]), (0, 255, 255), 2)
            cv2.rectangle(vis, (tip_box[0], tip_box[1]), (tip_box[2], tip_box[3]), (255, 255, 0), 2)

            x1, y1, x2, y2 = nbbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.circle(vis, (nx, ny), 5, (0, 255, 255), -1)

        else:
            # no nozzle, decay to NOT PRINTING with hysteresis
            self.off_count += 1
            self.on_count = 0
            if self.printing and self.off_count >= self.K_OFF:
                self.printing = False

        # label
        state_text = "PRINTING" if self.printing else "NOT PRINTING"
        safe_put_text(vis, state_text, (15, 32), scale=0.95, thickness=2)
        safe_put_text(vis, f"roi_new={new_area_roi_pct*100.0:.2f}%  tip_ov={overlap_tip:.3f}", (15, 60), scale=0.65, thickness=2)
        safe_put_text(vis, f"nozzle={nozzle_conf:.2f}  on={self.on_count} off={self.off_count}", (15, 84), scale=0.65, thickness=2)

        state = PrintingState(
            printing=bool(self.printing),
            nozzle_conf=float(nozzle_conf),
            new_area_roi_pct=float(new_area_roi_pct),
            overlap_tip=float(overlap_tip),
            on_count=int(self.on_count),
            off_count=int(self.off_count),
        )

        dbg = {"python": f"dt={(time.time()-t0)*1000.0:.1f}ms"}

        self._last_vis = vis
        self._last_state = state
        return vis, state, dbg
