#!/usr/bin/env python3
# printing_detection.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import time
import sys

import numpy as np
import cv2

YOLO_IMPORT_ERROR = None
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    YOLO_IMPORT_ERROR = e


COLORS_BGR = {
    "nozzle": (202, 250, 2),        # #02FACA
    "old_deposit": (89, 5, 253),    # #FD0559
    "new_deposit": (32, 168, 156),  # #9CA820
}

NEW_DEPOSIT_HL_BGR = (0, 252, 199)  # #C7FC00


@dataclass
class DetectionState:
    printing: bool
    nozzle_conf: float
    new_area_roi_px: int
    new_area_roi_pct: float
    overlap_tip: float
    on_count: int
    off_count: int


def overlay_masks_fixed(frame_bgr: np.ndarray, masks_dict: Dict[str, np.ndarray], classes: List[str], alpha: float = 0.6) -> np.ndarray:
    overlay = frame_bgr.copy()
    for cname in classes:
        m = masks_dict.get(cname, None)
        if m is None:
            continue
        color = COLORS_BGR.get(cname, (180, 180, 180))
        overlay[m > 0] = color
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0)


def draw_legend(img_bgr: np.ndarray, items: List[str], x: int = 20, y: int = 150) -> None:
    pad = 10
    sw = 22
    sh = 14
    line_h = 22

    w = 220
    h = pad * 2 + line_h * len(items)

    panel = img_bgr.copy()
    cv2.rectangle(panel, (x, y), (x + w, y + h), (0, 0, 0), -1)
    img_bgr[:] = cv2.addWeighted(panel, 0.35, img_bgr, 0.65, 0)

    for i, name in enumerate(items):
        yy = y + pad + i * line_h
        color = COLORS_BGR.get(name, (180, 180, 180))
        cv2.rectangle(img_bgr, (x + pad, yy), (x + pad + sw, yy + sh), color, -1)
        cv2.rectangle(img_bgr, (x + pad, yy), (x + pad + sw, yy + sh), (255, 255, 255), 1)
        cv2.putText(
            img_bgr,
            name,
            (x + pad + sw + 10, yy + sh - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


def draw_mask_edges(img_bgr: np.ndarray, mask_u8: np.ndarray, color: Tuple[int, int, int] = (255, 255, 255), thickness: int = 2) -> None:
    mu8 = (mask_u8 > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mu8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        cv2.drawContours(img_bgr, cnts, -1, color, thickness)


def _largest_contour_bbox(mask_u8: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    return x, y, w, h


def _clamp_roi(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


class PrintingDetector:
    def __init__(
        self,
        model_path: str,
        nozzle_class_ids: Optional[List[int]] = None,
        new_deposit_class_ids: Optional[List[int]] = None,
        old_deposit_class_ids: Optional[List[int]] = None,
        alpha: float = 0.6,
        roi_half_size_px: int = 110,
        tip_radius_px: int = 45,
        overlap_on_thresh: float = 0.05,
        overlap_off_thresh: float = 0.02,
        roi_area_on_pct: float = 0.010,
        roi_area_off_pct: float = 0.004,
        k_on: int = 3,
        k_off: int = 6,
        min_on_seconds: float = 0.4,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.6,
        img_size: int = 640,
        verbose: bool = False,
    ):
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics import failed.\n"
                f"python: {sys.executable}\n"
                f"error: {YOLO_IMPORT_ERROR}\n"
                "Fix: install in this env: pip install ultralytics"
            )

        self.model = YOLO(model_path)
        self.model_path = model_path

        self.nozzle_class_ids = nozzle_class_ids
        self.new_deposit_class_ids = new_deposit_class_ids
        self.old_deposit_class_ids = old_deposit_class_ids

        self.alpha = float(alpha)
        self.roi_half = int(roi_half_size_px)
        self.tip_radius = int(tip_radius_px)

        self.overlap_on_thresh = float(overlap_on_thresh)
        self.overlap_off_thresh = float(overlap_off_thresh)

        self.roi_area_on_pct = float(roi_area_on_pct)
        self.roi_area_off_pct = float(roi_area_off_pct)

        self.k_on = int(k_on)
        self.k_off = int(k_off)
        self.min_on_seconds = float(min_on_seconds)

        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.img_size = int(img_size)
        self.verbose = bool(verbose)

        self._on_count = 0
        self._off_count = 0
        self._printing = False
        self._on_since: Optional[float] = None

        self._names = getattr(self.model, "names", None)

    def _resolve_ids_by_name(self) -> Tuple[List[int], List[int], List[int]]:
        nozzle_ids: List[int] = []
        new_ids: List[int] = []
        old_ids: List[int] = []

        if isinstance(self._names, dict):
            for cid, nm in self._names.items():
                n = str(nm).lower()
                if "nozzle" in n or "extruder" in n or "tip" in n:
                    nozzle_ids.append(int(cid))
                if "old" in n:
                    old_ids.append(int(cid))
                if "new" in n or "deposit" in n or "layer" in n or "material" in n or "print" in n:
                    new_ids.append(int(cid))

        if not nozzle_ids:
            nozzle_ids = [1]
        if not new_ids:
            new_ids = [0]
        return nozzle_ids, new_ids, old_ids

    def _get_class_ids(self) -> Tuple[List[int], List[int], List[int]]:
        if self.nozzle_class_ids is not None or self.new_deposit_class_ids is not None or self.old_deposit_class_ids is not None:
            nozzle_ids = self.nozzle_class_ids if self.nozzle_class_ids is not None else []
            new_ids = self.new_deposit_class_ids if self.new_deposit_class_ids is not None else []
            old_ids = self.old_deposit_class_ids if self.old_deposit_class_ids is not None else []
            n2, new2, old2 = self._resolve_ids_by_name()
            if not nozzle_ids:
                nozzle_ids = n2
            if not new_ids:
                new_ids = new2
            if not old_ids:
                old_ids = old2
            return nozzle_ids, new_ids, old_ids

        return self._resolve_ids_by_name()

    def _closest_component_area_in_roi(
        self,
        bin_mask_u8: np.ndarray,
        tip_xy: Tuple[int, int],
        roi: Tuple[int, int, int, int]
    ) -> Tuple[int, float]:
        x1, y1, x2, y2 = roi
        m_roi = bin_mask_u8[y1:y2, x1:x2]
        if m_roi.size == 0:
            return 0, 0.0

        num, _, stats, centroids = cv2.connectedComponentsWithStats((m_roi > 0).astype(np.uint8), connectivity=8)
        if num <= 1:
            return 0, 0.0

        tip_cx, tip_cy = tip_xy
        tip_rx = tip_cx - x1
        tip_ry = tip_cy - y1

        best_d2 = 1e18
        best_area = 0
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            cx, cy = centroids[i]
            d2 = (cx - tip_rx) ** 2 + (cy - tip_ry) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_area = area

        roi_area = max(1, int((x2 - x1) * (y2 - y1)))
        return best_area, float(best_area) / float(roi_area)

    def update(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, DetectionState, Dict[str, Any]]:
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        h, w = frame_bgr.shape[:2]
        nozzle_ids, new_ids, old_ids = self._get_class_ids()

        masks_dict = {
            "nozzle": np.zeros((h, w), dtype=np.uint8),
            "new_deposit": np.zeros((h, w), dtype=np.uint8),
            "old_deposit": np.zeros((h, w), dtype=np.uint8),
        }
        nozzle_conf = 0.0

        res = self.model.predict(
            source=frame_bgr,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.img_size,
            verbose=self.verbose
        )

        if res and len(res) > 0:
            r0 = res[0]
            boxes = getattr(r0, "boxes", None)
            masks = getattr(r0, "masks", None)

            if boxes is not None and masks is not None and getattr(masks, "data", None) is not None:
                cls = boxes.cls.detach().cpu().numpy().astype(int) if boxes.cls is not None else None
                conf = boxes.conf.detach().cpu().numpy().astype(float) if boxes.conf is not None else None
                mdata = masks.data.detach().cpu().numpy()

                for i in range(mdata.shape[0]):
                    cid = int(cls[i]) if cls is not None else -1
                    mi = (mdata[i] > 0.5).astype(np.uint8) * 255

                    if cid in nozzle_ids:
                        masks_dict["nozzle"] = cv2.bitwise_or(masks_dict["nozzle"], mi)
                        if conf is not None:
                            nozzle_conf = max(nozzle_conf, float(conf[i]))

                    if cid in new_ids:
                        masks_dict["new_deposit"] = cv2.bitwise_or(masks_dict["new_deposit"], mi)

                    if cid in old_ids:
                        masks_dict["old_deposit"] = cv2.bitwise_or(masks_dict["old_deposit"], mi)

        nozzle_bbox = _largest_contour_bbox(masks_dict["nozzle"])

        tip_cx = w // 2
        tip_cy = h // 2
        if nozzle_bbox is not None:
            x, y, bw, bh = nozzle_bbox
            tip_cx = int(x + bw / 2)
            tip_cy = int(y + bh)

        x1, y1, x2, y2 = _clamp_roi(tip_cx - self.roi_half, tip_cy - self.roi_half, tip_cx + self.roi_half, tip_cy + self.roi_half, w, h)
        roi = (x1, y1, x2, y2)

        new_area_px, new_area_pct = self._closest_component_area_in_roi(masks_dict["new_deposit"], (tip_cx, tip_cy), roi)

        tip_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(tip_mask, (tip_cx, tip_cy), self.tip_radius, 255, -1)

        overlap = cv2.bitwise_and(masks_dict["new_deposit"], tip_mask)
        tip_area = max(1, int(cv2.countNonZero(tip_mask)))
        overlap_tip = float(cv2.countNonZero(overlap)) / float(tip_area)

        on_signal = (new_area_pct >= self.roi_area_on_pct) and (overlap_tip >= self.overlap_on_thresh)
        off_signal = (new_area_pct <= self.roi_area_off_pct) or (overlap_tip <= self.overlap_off_thresh)

        if on_signal:
            self._on_count += 1
            self._off_count = max(0, self._off_count - 1)
        elif off_signal:
            self._off_count += 1
            self._on_count = max(0, self._on_count - 1)

        now = time.monotonic()

        if (not self._printing) and (self._on_count >= self.k_on):
            self._printing = True
            self._off_count = 0
            self._on_since = now

        if self._printing:
            on_age = 0.0 if self._on_since is None else (now - self._on_since)
            if on_age >= self.min_on_seconds and self._off_count >= self.k_off:
                self._printing = False
                self._on_count = 0
                self._on_since = None

        vis = frame_bgr.copy()
        overlay_classes = ["new_deposit", "old_deposit", "nozzle"]
        vis = overlay_masks_fixed(vis, masks_dict, overlay_classes, alpha=self.alpha)

        for cname in overlay_classes:
            m = masks_dict.get(cname, None)
            if m is not None:
                draw_mask_edges(vis, m, color=(255, 255, 255), thickness=2)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.circle(vis, (tip_cx, tip_cy), self.tip_radius, (255, 255, 255), 2)

        draw_legend(vis, ["nozzle", "new_deposit", "old_deposit"], x=20, y=150)

        status = "PRINTING" if self._printing else "NOT PRINTING"
        label_bg = NEW_DEPOSIT_HL_BGR if self._printing else (0, 0, 0)

        x0, y0 = 12, 10
        w_box, h_box = 420, 64
        panel = vis.copy()
        cv2.rectangle(panel, (x0, y0), (x0 + w_box, y0 + h_box), label_bg, -1)
        vis[:] = cv2.addWeighted(panel, 0.55, vis, 0.45, 0)

        txt_color = (0, 0, 0) if self._printing else (255, 255, 255)
        cv2.putText(vis, status, (x0 + 10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.95, txt_color, 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"roi={new_area_pct*100:.2f}% ov={overlap_tip:.3f} noz={nozzle_conf:.2f} on={self._on_count} off={self._off_count}",
            (x0 + 10, y0 + 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            txt_color,
            2,
            cv2.LINE_AA
        )

        st = DetectionState(
            printing=bool(self._printing),
            nozzle_conf=float(nozzle_conf),
            new_area_roi_px=int(new_area_px),
            new_area_roi_pct=float(new_area_pct),
            overlap_tip=float(overlap_tip),
            on_count=int(self._on_count),
            off_count=int(self._off_count),
        )

        dbg: Dict[str, Any] = {
            "tip": (tip_cx, tip_cy),
            "roi": (x1, y1, x2, y2),
            "python": sys.executable,
            "nozzle_ids": nozzle_ids,
            "new_ids": new_ids,
            "old_ids": old_ids,
        }
        return vis, st, dbg
