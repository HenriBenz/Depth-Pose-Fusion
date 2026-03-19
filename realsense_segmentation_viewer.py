#!/usr/bin/env python3
"""
Minimal live segmentation viewer for Intel RealSense.

What it does
- detects connected RealSense camera(s)
- opens the first available device
- runs YOLO segmentation on the live RGB stream
- shows segmentation masks + bounding boxes + class labels + confidence
- optional crop and rotation
- optional nozzle-only stable tracking overlay

Keys
- q / ESC : quit
- s       : save current frame with overlay
- r       : toggle raw / overlay view
- n       : toggle nozzle-only highlight
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# important on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import torch  # noqa: F401
    from ultralytics import YOLO
except Exception as e:
    print(f"[ERROR] Failed to import torch/ultralytics: {e}")
    sys.exit(1)

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
except Exception as e:
    rs = None
    RS_IMPORT_ERROR = e
else:
    RS_IMPORT_ERROR = None


CLASS_COLORS = {
    "new_deposit": (0, 0, 255),
    "old_deposit": (0, 255, 255),
    "nozzle": (255, 0, 0),
}


def safe_text(img: np.ndarray, text: str, org: Tuple[int, int], scale: float = 0.65,
              fg: Tuple[int, int, int] = (255, 255, 255),
              bg: Tuple[int, int, int] = (0, 0, 0), thickness: int = 2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, bg, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)


def class_color(name: str) -> Tuple[int, int, int]:
    return CLASS_COLORS.get(name, (180, 180, 180))


def rotate_img(img: np.ndarray, deg: int) -> np.ndarray:
    if deg == 0:
        return img
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 270 or deg == -90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)


def clamp_crop(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


class StableNozzlePicker:
    def __init__(self, hold_sec: float = 1.2, ema_alpha: float = 0.2, dist_weight: float = 0.0025):
        self.hold_sec = float(hold_sec)
        self.ema_alpha = float(ema_alpha)
        self.dist_weight = float(dist_weight)
        self.last_pick: Optional[Dict] = None
        self.last_time = 0.0
        self.sx: Optional[float] = None
        self.sy: Optional[float] = None

    def pick(self, boxes, names: Dict[int, str], target_name: str = "nozzle") -> Optional[Dict]:
        now = time.time()
        best = None

        if boxes is not None and len(boxes) > 0:
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            for i in range(len(cls)):
                cname = names[int(cls[i])]
                if cname != target_name:
                    continue
                x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                conf = float(confs[i])
                score = conf
                if self.sx is not None and self.sy is not None:
                    dist = float(np.hypot(cx - self.sx, cy - self.sy))
                    score = conf - self.dist_weight * dist
                cand = {
                    "score": score,
                    "conf": conf,
                    "cx": float(cx),
                    "cy": float(cy),
                    "box": (x1, y1, x2, y2),
                }
                if best is None or cand["score"] > best["score"]:
                    best = cand

        if best is not None:
            if self.sx is None:
                self.sx = best["cx"]
                self.sy = best["cy"]
            else:
                self.sx = (1.0 - self.ema_alpha) * self.sx + self.ema_alpha * best["cx"]
                self.sy = (1.0 - self.ema_alpha) * self.sy + self.ema_alpha * best["cy"]
            best["cx_smooth"] = int(round(self.sx))
            best["cy_smooth"] = int(round(self.sy))
            self.last_pick = best
            self.last_time = now
            return best

        if self.last_pick is not None and (now - self.last_time) <= self.hold_sec:
            held = dict(self.last_pick)
            held["held"] = True
            return held

        return None


def build_instance_masks(result, H: int, W: int) -> List[Dict]:
    out: List[Dict] = []
    if result.masks is None or result.boxes is None or result.masks.data is None:
        return out

    masks = result.masks.data.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy()
    names = result.names

    n = min(len(masks), len(cls), len(confs), len(xyxy))
    for i in range(n):
        mask = (masks[i] > 0.5).astype(np.uint8)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        cname = names[int(cls[i])]
        x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
        out.append({
            "class_name": cname,
            "conf": float(confs[i]),
            "box": (x1, y1, x2, y2),
            "mask": mask,
        })
    return out


def overlay_instances(frame_bgr: np.ndarray, instances: List[Dict], alpha: float = 0.55,
                      nozzle_pick: Optional[Dict] = None, nozzle_only: bool = False) -> np.ndarray:
    vis = frame_bgr.copy()
    paint = frame_bgr.copy()

    nozzle_box = None if nozzle_pick is None else nozzle_pick.get("box")

    for inst in instances:
        cname = inst["class_name"]
        box = inst["box"]
        if nozzle_only and cname == "nozzle":
            keep = nozzle_box is not None and tuple(box) == tuple(nozzle_box)
            if not keep:
                continue

        color = class_color(cname)
        mask = inst["mask"]
        paint[mask == 1] = color

    vis = cv2.addWeighted(paint, alpha, vis, 1.0 - alpha, 0.0)

    for inst in instances:
        cname = inst["class_name"]
        box = inst["box"]
        conf = inst["conf"]

        if nozzle_only and cname == "nozzle":
            keep = nozzle_box is not None and tuple(box) == tuple(nozzle_box)
            if not keep:
                continue

        x1, y1, x2, y2 = box
        color = class_color(cname)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        safe_text(vis, f"{cname} {conf:.2f}", (x1, max(22, y1 - 8)), scale=0.55, fg=color)

    if nozzle_pick is not None:
        x1, y1, x2, y2 = nozzle_pick["box"]
        cx = int(nozzle_pick.get("cx_smooth", (x1 + x2) // 2))
        cy = int(nozzle_pick.get("cy_smooth", (y1 + y2) // 2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.circle(vis, (cx, cy), 5, (255, 255, 255), -1)
        extra = "HELD" if nozzle_pick.get("held", False) else "TRACKED"
        safe_text(vis, f"nozzle_best {nozzle_pick['conf']:.2f} {extra}", (15, 55), scale=0.70)

    return vis


class RealSenseRGB:
    def __init__(self, width: int, height: int, fps: int):
        if rs is None:
            raise RuntimeError(f"pyrealsense2 import failed: {RS_IMPORT_ERROR}")
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.pipeline: Optional[rs.pipeline] = None
        self.profile = None
        self.device_name = ""
        self.serial = ""

    def start(self) -> None:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("No RealSense camera detected.")

        dev = devices[0]
        self.device_name = dev.get_info(rs.camera_info.name)
        self.serial = dev.get_info(rs.camera_info.serial_number)

        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

        self.pipeline = rs.pipeline()
        self.profile = self.pipeline.start(config)

    def read(self) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("RealSense pipeline not started.")
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("No color frame received.")
        img = np.asanyarray(color_frame.get_data())
        return img

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live RealSense segmentation viewer")
    p.add_argument("--model", type=str, required=True, help="Path to YOLO segmentation model")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--rotate", type=int, default=0, help="0, 90, 180, 270")
    p.add_argument("--crop-x", type=int, default=0)
    p.add_argument("--crop-y", type=int, default=0)
    p.add_argument("--crop-w", type=int, default=0, help="0 means full width")
    p.add_argument("--crop-h", type=int, default=0, help="0 means full height")
    p.add_argument("--alpha", type=float, default=0.55, help="Mask overlay opacity")
    p.add_argument("--save-dir", type=str, default="captures_live")
    p.add_argument("--device", type=str, default="cpu", help="cpu, cuda, 0, ...")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    print("[INFO] Loading model...")
    model = YOLO(str(model_path))
    print(f"[INFO] Model loaded: {model_path}")

    cam = RealSenseRGB(args.width, args.height, args.fps)
    try:
        cam.start()
    except Exception as e:
        print(f"[ERROR] Camera start failed: {e}")
        return 2

    print(f"[INFO] RealSense connected: {cam.device_name} | serial={cam.serial}")
    print("[INFO] Keys: q/ESC quit | s save frame | r toggle raw/overlay | n toggle nozzle-only highlight")

    nozzle_picker = StableNozzlePicker()
    show_overlay = True
    nozzle_only = False
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    t_prev = time.time()
    fps_smooth = 0.0

    try:
        while True:
            frame = cam.read()
            if frame is None:
                continue

            frame = rotate_img(frame, int(args.rotate))
            H, W = frame.shape[:2]
            crop_w = W if int(args.crop_w) <= 0 else int(args.crop_w)
            crop_h = H if int(args.crop_h) <= 0 else int(args.crop_h)
            x, y, w, h = clamp_crop(args.crop_x, args.crop_y, crop_w, crop_h, W, H)
            crop = frame[y:y + h, x:x + w].copy()

            result = model.predict(crop, conf=float(args.conf), imgsz=int(args.imgsz), verbose=False, device=args.device)[0]
            instances = build_instance_masks(result, crop.shape[0], crop.shape[1])
            nozzle_pick = nozzle_picker.pick(result.boxes, result.names, target_name="nozzle")

            vis_crop = overlay_instances(crop, instances, alpha=float(args.alpha), nozzle_pick=nozzle_pick, nozzle_only=nozzle_only)

            if not show_overlay:
                vis_crop = crop.copy()

            # header
            now = time.time()
            dt = max(1e-6, now - t_prev)
            t_prev = now
            fps_now = 1.0 / dt
            fps_smooth = fps_now if fps_smooth <= 0 else 0.9 * fps_smooth + 0.1 * fps_now

            safe_text(vis_crop, f"FPS {fps_smooth:.1f}", (15, 25), scale=0.7)
            safe_text(vis_crop, f"objects {len(instances)}", (15, 85), scale=0.7)
            safe_text(vis_crop, f"view {'overlay' if show_overlay else 'raw'} | nozzle_only {int(nozzle_only)}", (15, 115), scale=0.65)

            cv2.imshow("Live Segmentation", vis_crop)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            elif key == ord('r'):
                show_overlay = not show_overlay
            elif key == ord('n'):
                nozzle_only = not nozzle_only
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                out = save_dir / f"seg_{ts}.png"
                cv2.imwrite(str(out), vis_crop)
                print(f"[INFO] Saved: {out}")

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        return 3
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
