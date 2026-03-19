#!/usr/bin/env python3
# ui_recorder_v.py

import os
import sys
import time
import re
import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Deque, Tuple
from collections import deque

# ------------------------------------------------------------
# Windows stability: import torch/ultralytics BEFORE numpy/cv2/Qt
# This avoids OpenMP DLL conflicts that cause WinError 1114.
# ------------------------------------------------------------
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # helps some OpenMP collisions
os.environ.setdefault("OMP_NUM_THREADS", "1")

TORCH_OK = False
TORCH_ERR = None
try:
    import torch  # noqa
    from ultralytics import YOLO  # noqa
    TORCH_OK = True
except Exception as e:
    TORCH_OK = False
    TORCH_ERR = e

# Now import the rest
import numpy as np
import cv2
import pyrealsense2 as rs
from py_openshowvar import openshowvar

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# Import detector AFTER torch preload
from printing_detection import PrintingDetector


def safe_float(s: str) -> Optional[float]:
    try:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))
        if not m:
            return None
        return float(m.group(0))
    except Exception:
        return None


def parse_axis_act(raw: str) -> Optional[np.ndarray]:
    try:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(raw))
        if len(nums) < 6:
            return None
        vals = [float(x) for x in nums[:6]]
        return np.array(vals, dtype=np.float32)
    except Exception:
        return None


def parse_pwm(raw: str) -> Optional[float]:
    return safe_float(raw)



def _to_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    return str(v).strip()


def _parse_num_from_struct(s: str, key: str) -> Optional[float]:
    """Parse 'A1 12.3' style tokens from KUKA $AXIS_ACT / structs."""
    try:
        m = re.search(rf"\b{re.escape(key)}\s+(-?\d+(?:\.\d+)?)", str(s))
        if not m:
            return None
        return float(m.group(1))
    except Exception:
        return None

def frame_id_str(i: int) -> str:
    return f"{i:06d}"


def ensure_dirs(base: Path) -> None:
    (base / "camera" / "rgb").mkdir(parents=True, exist_ok=True)
    (base / "camera" / "depth").mkdir(parents=True, exist_ok=True)
    (base / "robot").mkdir(parents=True, exist_ok=True)
    (base / "calibration").mkdir(parents=True, exist_ok=True)


def next_session_name(base: Path) -> str:
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    yy = time.strftime("%y%m%d")
    prefix = f"{yy}_Recording_"
    existing = [p.name for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    max_id = 0
    for name in existing:
        m = re.match(rf"{re.escape(prefix)}(\d+)$", name)
        if m:
            max_id = max(max_id, int(m.group(1)))
    return f"{prefix}{max_id + 1:03d}"


def save_realsense_intrinsics(out_dir: Path, profile: rs.pipeline_profile, no_rgb: bool) -> None:
    intr = {}
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intr = depth_stream.get_intrinsics()
    intr["depth"] = {
        "width": depth_intr.width,
        "height": depth_intr.height,
        "ppx": depth_intr.ppx,
        "ppy": depth_intr.ppy,
        "fx": depth_intr.fx,
        "fy": depth_intr.fy,
        "model": str(depth_intr.model),
        "coeffs": list(depth_intr.coeffs),
    }

    if not no_rgb:
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        color_intr = color_stream.get_intrinsics()
        intr["color"] = {
            "width": color_intr.width,
            "height": color_intr.height,
            "ppx": color_intr.ppx,
            "ppy": color_intr.ppy,
            "fx": color_intr.fx,
            "fy": color_intr.fy,
            "model": str(color_intr.model),
            "coeffs": list(color_intr.coeffs),
        }

    with open(out_dir / "calibration" / "realsense_intrinsics.json", "w", encoding="utf-8") as f:
        json.dump(intr, f, indent=2)


def rotate_img(img: np.ndarray, deg: int) -> np.ndarray:
    if img is None:
        return img
    deg = int(deg)
    if deg == 0:
        return img
    if deg == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if deg == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if deg == -90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    interp = cv2.INTER_NEAREST if img.dtype != np.uint8 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=interp)


def depth_to_preview(depth_u16: np.ndarray) -> np.ndarray:
    d = depth_u16.astype(np.float32)
    nz = d[d > 0]
    if nz.size < 50:
        lo, hi = 0.0, max(1.0, float(d.max()))
    else:
        lo, hi = float(np.percentile(nz, 5)), float(np.percentile(nz, 95))
        if hi <= lo:
            hi = lo + 1.0
    d = np.clip((d - lo) / (hi - lo), 0, 1)
    gray = (d * 255.0).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def clamp_crop(x: int, y: int, w: int, h: int, W: int, H: int) -> Tuple[int, int, int, int]:
    x = max(0, min(int(x), W - 1))
    y = max(0, min(int(y), H - 1))
    w = max(1, min(int(w), W - x))
    h = max(1, min(int(h), H - y))
    return x, y, w, h


@dataclass
class RobotSample:
    t_wall: float
    axis: Optional[np.ndarray]
    pwm: Optional[float]
    raw: Dict[str, Any]


class RobotClient:
    """KUKA reader (OpenShowVar). Tries to read E1_ON if available.
    If E1_ON is not available, derives a binary ON/OFF signal from E1 velocity
    (same logic style as ui_recorder.py).
    """

    def __init__(self, ip: str, port: int = 7000, e1_vel_threshold: float = 5.0):
        self.ip = ip
        self.port = port
        self.e1_vel_threshold = float(e1_vel_threshold)

        self._osv = openshowvar(self.ip, self.port)

        self._last_e1: Optional[float] = None
        self._last_t: Optional[float] = None

    def _e1_on_from_velocity(self, t: float, e1: Optional[float]) -> Optional[float]:
        if e1 is None:
            return None
        if self._last_e1 is None or self._last_t is None:
            self._last_e1 = float(e1)
            self._last_t = float(t)
            return None
        dt = float(t) - float(self._last_t)
        if dt <= 1e-6:
            return None
        vel = abs((float(e1) - float(self._last_e1)) / dt)
        self._last_e1 = float(e1)
        self._last_t = float(t)
        return 1.0 if vel >= self.e1_vel_threshold else 0.0

    def poll(self) -> Optional[RobotSample]:
        t = time.time()

        try:
            raw_axis = _to_text(self._osv.read("$AXIS_ACT"))
        except Exception:
            raw_axis = ""
        if not raw_axis:
            return None

        # axis A1..A6 + E1 if present
        a1 = _parse_num_from_struct(raw_axis, "A1")
        a2 = _parse_num_from_struct(raw_axis, "A2")
        a3 = _parse_num_from_struct(raw_axis, "A3")
        a4 = _parse_num_from_struct(raw_axis, "A4")
        a5 = _parse_num_from_struct(raw_axis, "A5")
        a6 = _parse_num_from_struct(raw_axis, "A6")
        e1 = _parse_num_from_struct(raw_axis, "E1")

        axis = None
        if all(v is not None for v in (a1, a2, a3, a4, a5, a6)):
            axis = np.array([a1, a2, a3, a4, a5, a6], dtype=np.float32)
        else:
            # fallback: old parser (first 6 numeric tokens)
            axis = parse_axis_act(raw_axis)

        # Prefer direct E1_ON variable if it exists
        raw_e1_on = ""
        pwm = None
        try:
            raw_e1_on = _to_text(self._osv.read("E1_ON"))
            pwm = parse_pwm(raw_e1_on)
        except Exception:
            raw_e1_on = ""
            pwm = None

        # If not available, derive from E1 velocity
        if pwm is None:
            pwm = self._e1_on_from_velocity(t, e1)

        # Optional extra vars for debugging/logging (no hard dependency)
        raw_e_rpm = ""
        raw_extr_mod = ""
        raw_ov = ""
        try:
            raw_e_rpm = _to_text(self._osv.read("E_RPM"))
        except Exception:
            pass
        try:
            raw_extr_mod = _to_text(self._osv.read("EXTR_MOD"))
        except Exception:
            pass
        try:
            raw_ov = _to_text(self._osv.read("$OV_PRO"))
        except Exception:
            pass

        return RobotSample(
            t_wall=t,
            axis=axis,
            pwm=pwm,
            raw={
                "$AXIS_ACT": raw_axis,
                "E1_ON": raw_e1_on,
                "E_RPM": raw_e_rpm,
                "EXTR_MOD": raw_extr_mod,
                "$OV_PRO": raw_ov,
            },
        )


class CaptureWorker(QtCore.QThread):
    sig_frame = QtCore.pyqtSignal(dict)
    sig_robot = QtCore.pyqtSignal(object)
    sig_status = QtCore.pyqtSignal(str)

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        no_rgb: bool = False,
        rotate_deg: int = 180,
        robot_ip: str = "10.1.0.121",
        port: int = 7000,
    ):
        super().__init__()
        self.width = width
        self.height = height
        self.fps = fps
        self.no_rgb = no_rgb
        self.rotate_deg = rotate_deg

        self.depth_min_mm = 10
        self.depth_max_mm = 500
        self.depth_unit_mm = 1.0

        self.crop_x = 0
        self.crop_y = 0
        self.crop_w = width
        self.crop_h = height

        self.det_enabled = False
        self.det_every_n = 1
        self._det_frame_i = 0
        self._detector: Optional[PrintingDetector] = None

        self._stop = False
        self.recording = False
        self.record_end_mono: Optional[float] = None

        self.session_dir: Optional[Path] = None
        self.frame_idx = 0

        self.pipeline: Optional[rs.pipeline] = None
        self.align: Optional[rs.align] = None
        self._rs_profile: Optional[rs.pipeline_profile] = None

        self.robot_ip = robot_ip
        self.port = port
        self.robot: Optional[RobotClient] = None

        self.robot_f = None
        self.robot_writer = None
        self.cam_ts_f = None
        self.cam_ts_writer = None

    def stop(self):
        self._stop = True

    def set_rotate(self, deg: int):
        self.rotate_deg = int(deg)

    def set_depth_range_mm(self, dmin_mm: int, dmax_mm: int):
        self.depth_min_mm = int(dmin_mm)
        self.depth_max_mm = int(dmax_mm)

    def set_crop(self, x: int, y: int, w: int, h: int):
        self.crop_x = int(x)
        self.crop_y = int(y)
        self.crop_w = int(w)
        self.crop_h = int(h)

    def set_detection(self, enabled: bool, detector: Optional[PrintingDetector], every_n: int):
        self.det_enabled = bool(enabled)
        self._detector = detector if self.det_enabled else None
        self.det_every_n = max(1, int(every_n))

    def start_recording(self, out_base: Path, session_name: str, duration_sec: float):
        out_dir = Path(out_base) / session_name
        ensure_dirs(out_dir)
        self.session_dir = out_dir
        self.frame_idx = 0

        self.robot_f = open(out_dir / "robot" / "robot.csv", "w", newline="", encoding="utf-8")
        self.robot_writer = csv.writer(self.robot_f)
        self.robot_writer.writerow(["t_wall", "a1", "a2", "a3", "a4", "a5", "a6", "pwm", "raw_axis", "raw_pwm"])

        self.cam_ts_f = open(out_dir / "camera" / "camera_timestamps.csv", "w", newline="", encoding="utf-8")
        self.cam_ts_writer = csv.writer(self.cam_ts_f)
        self.cam_ts_writer.writerow(["frame_id", "t_wall", "t_mono", "t_rs_depth_ms", "rotate_deg"])

        meta = {
            "session_name": session_name,
            "out_dir": str(out_dir),
            "start_time_wall": time.time(),
            "duration_sec": float(duration_sec),
        }
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if duration_sec and duration_sec > 0:
            self.record_end_mono = time.monotonic() + float(duration_sec)
        else:
            self.record_end_mono = None

        self.recording = True
        self.sig_status.emit(f"Recording started: {out_dir}")

        if self._rs_profile is not None:
            try:
                save_realsense_intrinsics(out_dir, self._rs_profile, self.no_rgb)
            except Exception:
                pass

    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False
        self.record_end_mono = None

        try:
            if self.robot_f:
                self.robot_f.flush()
                self.robot_f.close()
        except Exception:
            pass
        try:
            if self.cam_ts_f:
                self.cam_ts_f.flush()
                self.cam_ts_f.close()
        except Exception:
            pass

        self.sig_status.emit("Recording stopped")

    def _save_images(self, out_dir: Path, frame_id: str, rgb: Optional[np.ndarray], depth: np.ndarray):
        if rgb is not None:
            cv2.imwrite(str(out_dir / "camera" / "rgb" / f"{frame_id}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "camera" / "depth" / f"{frame_id}.png"), depth)

    def run(self):
        self.sig_status.emit("Starting RealSense...")
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        if not self.no_rgb:
            cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)

        self._rs_profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color) if not self.no_rgb else None

        try:
            depth_sensor = self._rs_profile.get_device().first_depth_sensor()
            depth_scale = float(depth_sensor.get_depth_scale())
            self.depth_unit_mm = depth_scale * 1000.0
        except Exception:
            self.depth_unit_mm = 1.0

        self.sig_status.emit(f"RealSense started (depth_unit_mm={self.depth_unit_mm:.4f})")

        try:
            self.robot = RobotClient(self.robot_ip, self.port)
            self.sig_status.emit(f"Robot connected: {self.robot_ip}:{self.port}")
        except Exception as e:
            self.robot = None
            self.sig_status.emit(f"Robot connect failed: {e}")

        while not self._stop:
            if self.recording and self.record_end_mono is not None and time.monotonic() >= self.record_end_mono:
                self.stop_recording()

            st = None
            if self.robot is not None:
                try:
                    st = self.robot.poll()
                except Exception:
                    st = None

            if st is not None:
                self.sig_robot.emit(st)
                if self.recording and self.robot_writer is not None:
                    a = st.axis if st.axis is not None else [None] * 6
                    self.robot_writer.writerow([st.t_wall, a[0], a[1], a[2], a[3], a[4], a[5], st.pwm, st.raw.get("$AXIS_ACT", ""), st.raw.get("E1_ON", "")])
                    self.robot_f.flush()

            try:
                frames = self.pipeline.wait_for_frames(15000)
                t_wall = time.time()

                if self.align is not None:
                    frames = self.align.process(frames)

                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue
                color_frame = frames.get_color_frame() if not self.no_rgb else None

                depth = np.asanyarray(depth_frame.get_data())
                rgb = np.asanyarray(color_frame.get_data()) if color_frame else None

                depth = rotate_img(depth, self.rotate_deg)
                if rgb is not None:
                    rgb = rotate_img(rgb, self.rotate_deg)

                depth_prev = depth_to_preview(depth)

                crop_bgr = None
                det_vis_bgr = None
                det_state = None

                if rgb is not None:
                    H, W = rgb.shape[:2]
                    cx, cy, cw, ch = clamp_crop(self.crop_x, self.crop_y, self.crop_w, self.crop_h, W, H)

                    rgb_crop = rgb[cy:cy + ch, cx:cx + cw]
                    depth_crop = depth[cy:cy + ch, cx:cx + cw]

                    crop_bgr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)

                    if self.det_enabled and self._detector is not None:
                        self._det_frame_i += 1
                        if (self._det_frame_i % self.det_every_n) == 0:
                            try:
                                d_mm = depth_crop.astype(np.float32) * float(self.depth_unit_mm)
                                m = (depth_crop > 0) & (d_mm >= float(self.depth_min_mm)) & (d_mm <= float(self.depth_max_mm))
                                crop_bgr_in = crop_bgr.copy()
                                crop_bgr_in[~m] = 0
                            except Exception:
                                crop_bgr_in = crop_bgr

                            try:
                                det_vis_bgr, st_det, dbg = self._detector.update(crop_bgr_in)
                                det_state = {
                                    "printing": bool(st_det.printing),
                                    "nozzle_conf": float(st_det.nozzle_conf),
                                    "new_area_roi_pct": float(st_det.new_area_roi_pct),
                                    "overlap_tip": float(st_det.overlap_tip),
                                    "on_count": int(st_det.on_count),
                                    "off_count": int(st_det.off_count),
                                    "python": dbg.get("python", ""),
                                }
                            except Exception as e:
                                self.sig_status.emit(f"Detection runtime error: {e}")
                                det_vis_bgr = None
                                det_state = None

                payload = {
                    "t_wall": t_wall,
                    "t_rs_depth": depth_frame.get_timestamp(),
                    "rgb_full": rgb,
                    "depth_preview": depth_prev,
                    "rgb_crop_bgr": crop_bgr,
                    "det_vis_bgr": det_vis_bgr,
                    "det_state": det_state,
                }
                self.sig_frame.emit(payload)

                if self.recording and self.session_dir is not None and self.cam_ts_writer is not None:
                    fid = frame_id_str(self.frame_idx)
                    self._save_images(self.session_dir, fid, rgb, depth)
                    self.cam_ts_writer.writerow([fid, t_wall, time.monotonic(), depth_frame.get_timestamp(), int(self.rotate_deg)])
                    self.cam_ts_f.flush()
                    self.frame_idx += 1

            except Exception as e:
                self.sig_status.emit(f"Capture ERROR: {e}")
                time.sleep(0.2)

        try:
            if self.pipeline:
                self.pipeline.stop()
        except Exception:
            pass


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth-Pose Logger UI")
        self.resize(1700, 930)

        self.worker = CaptureWorker()
        self.worker.sig_frame.connect(self.on_frame)
        self.worker.sig_robot.connect(self.on_robot)
        self.worker.sig_status.connect(self.on_status)

        self.worker.start()

        self._t0 = time.time()
        self._last_printing: Optional[bool] = None
        self._no_print_since: Optional[float] = None  # wall time when NOT PRINTING started
        self._x: Deque[float] = deque(maxlen=600)
        self._y_axes: Deque[np.ndarray] = deque(maxlen=600)
        self._y_pwm: Deque[float] = deque(maxlen=600)

        layout = QtWidgets.QGridLayout(self)

        # ------------------------------------------------------------
        # Camera views layout (matches: 1 big left, 2/3/4 stacked right)
        # 1 = segmentation/decision, 2 = RGB, 3 = RGBD (depth), 4 = crop RGB (ROI)
        # ------------------------------------------------------------
        self.rgb_full_label = QtWidgets.QLabel()   # (2) RGB
        self.depth_label = QtWidgets.QLabel()      # (3) Depth preview
        self.rgb_crop_label = QtWidgets.QLabel()   # (4) ROI crop
        self.det_label = QtWidgets.QLabel()        # (1) Segmentation + decision

        for lab in [self.rgb_full_label, self.depth_label, self.rgb_crop_label, self.det_label]:
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setStyleSheet("background:#0b0b0b; border:1px solid #2a2a2a;")

        # make right column compact, left panel dominant
        self.rgb_full_label.setMinimumSize(360, 210)
        self.depth_label.setMinimumSize(360, 210)
        self.rgb_crop_label.setMinimumSize(360, 210)
        self.det_label.setMinimumSize(980, 650)

        def _view_box(title: str, subtitle: str, inner: QtWidgets.QWidget) -> QtWidgets.QGroupBox:
            box = QtWidgets.QGroupBox(title)
            v = QtWidgets.QVBoxLayout(box)
            sublab = QtWidgets.QLabel(subtitle)
            sublab.setStyleSheet("color:#bdbdbd;")
            sublab.setWordWrap(True)
            v.addWidget(sublab)
            v.addWidget(inner, 1)
            v.setContentsMargins(10, 12, 10, 10)
            v.setSpacing(6)
            return box

        self.box_det = _view_box("1  Segmentation (decision)", "Masks (60% opacity) + ROI + tip-zone. This is the view that decides PRINTING vs NOT PRINTING.", self.det_label)
        self.box_rgb = _view_box("2  RGB (raw)", "Clean RGB feed without overlays.", self.rgb_full_label)
        self.box_depth = _view_box("3  RGBD (depth preview)", "Aligned depth preview for debugging (not used for the decision).", self.depth_label)
        self.box_crop = _view_box("4  Crop RGB (ROI)", "ROI crop used as input for detection.", self.rgb_crop_label)

        cam_grid = QtWidgets.QGridLayout()
        cam_grid.addWidget(self.box_det, 0, 0, 3, 1)
        cam_grid.addWidget(self.box_rgb, 0, 1)
        cam_grid.addWidget(self.box_depth, 1, 1)
        cam_grid.addWidget(self.box_crop, 2, 1)
        cam_grid.setColumnStretch(0, 4)
        cam_grid.setColumnStretch(1, 1)
        cam_grid.setRowStretch(0, 1)
        cam_grid.setRowStretch(1, 1)
        cam_grid.setRowStretch(2, 1)

        layout.addLayout(cam_grid, 0, 0, 1, 2)


        # plot
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.18)
        self.plot.setBackground((14, 14, 14))
        self.plot.getPlotItem().setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)

        self.legend = self.plot.addLegend(offset=(15, 15))
        self.curves = []
        curve_names = ["Axis 01", "Axis 02", "Axis 03", "Axis 04", "Axis 05", "Axis 06"]
        for i in range(6):
            pen = pg.mkPen(pg.intColor(i, hues=6), width=2)
            c = self.plot.plot([], [], pen=pen, name=curve_names[i])
            self.curves.append(c)
        self.pwm_curve = self.plot.plot([], [], pen=pg.mkPen((240, 240, 240), width=2), name="E1_ON (PWM)")

        # controls
        ctrl = QtWidgets.QGroupBox("Controls")
        form = QtWidgets.QFormLayout(ctrl)

        self.out_base = QtWidgets.QLineEdit(str(Path("dataset").resolve()))
        self.out_base.textChanged.connect(lambda: self.update_next_name())

        self.next_name = QtWidgets.QLineEdit("")
        self.next_name.setReadOnly(True)

        self.duration = QtWidgets.QDoubleSpinBox()
        self.duration.setRange(0.0, 3600.0)
        self.duration.setValue(0.0)
        self.duration.setSingleStep(1.0)
        self.duration.setSuffix(" s (0=manual stop)")
        self.duration.setDecimals(1)

        self.rotate_combo = QtWidgets.QComboBox()
        self.rotate_combo.addItems(["180", "0", "90", "-90"])
        self.rotate_combo.setCurrentText("180")
        self.rotate_combo.currentTextChanged.connect(self.on_rotate_changed)

        self.det_enable = QtWidgets.QCheckBox("Enable printing detection")
        self.det_enable.setChecked(False)
        self.det_enable.stateChanged.connect(self.on_detection_changed)

        self.det_model = QtWidgets.QLineEdit("Print_detection_v1.pt")
        self.det_model.textChanged.connect(self.on_detection_changed)

        self.det_every = QtWidgets.QSpinBox()
        self.det_every.setRange(1, 30)
        self.det_every.setValue(1)
        self.det_every.valueChanged.connect(self.on_detection_changed)

        # detection tuning
        self.det_conf = QtWidgets.QDoubleSpinBox()
        self.det_conf.setRange(0.01, 0.99)
        self.det_conf.setSingleStep(0.01)
        self.det_conf.setValue(0.35)
        self.det_conf.setToolTip("Minimum confidence for detections. Higher = fewer false positives, but may miss weak masks.")
        self.det_conf.valueChanged.connect(self.on_detection_changed)

        self.det_alpha = QtWidgets.QDoubleSpinBox()
        self.det_alpha.setRange(0.0, 1.0)
        self.det_alpha.setSingleStep(0.05)
        self.det_alpha.setValue(0.60)
        self.det_alpha.setToolTip("Mask overlay opacity in view 1 (segmentation). 0 = invisible, 1 = solid.")
        self.det_alpha.valueChanged.connect(self.on_detection_changed)

        self.depth_min = QtWidgets.QSpinBox()
        self.depth_min.setRange(0, 10000)
        self.depth_min.setValue(10)
        self.depth_min.setSuffix(" mm")
        self.depth_min.valueChanged.connect(self.on_depth_range_changed)

        self.depth_max = QtWidgets.QSpinBox()
        self.depth_max.setRange(0, 10000)
        self.depth_max.setValue(500)
        self.depth_max.setSuffix(" mm")
        self.depth_max.valueChanged.connect(self.on_depth_range_changed)

        self.crop_x = QtWidgets.QSpinBox()
        self.crop_x.setRange(0, 10000)
        self.crop_x.setValue(0)
        self.crop_x.valueChanged.connect(self.on_crop_changed)

        self.crop_y = QtWidgets.QSpinBox()
        self.crop_y.setRange(0, 10000)
        self.crop_y.setValue(0)
        self.crop_y.valueChanged.connect(self.on_crop_changed)

        self.crop_w = QtWidgets.QSpinBox()
        self.crop_w.setRange(1, 10000)
        self.crop_w.setValue(1280)
        self.crop_w.valueChanged.connect(self.on_crop_changed)

        self.crop_h = QtWidgets.QSpinBox()
        self.crop_h.setRange(1, 10000)
        self.crop_h.setValue(720)
        self.crop_h.valueChanged.connect(self.on_crop_changed)

        self.btn_record = QtWidgets.QPushButton("● Record")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_record)


        # status (operator friendly)
        self.print_pill = QtWidgets.QLabel("PRINTING: -")
        self.print_pill.setAlignment(QtCore.Qt.AlignCenter)
        self.print_pill.setStyleSheet("padding:10px; border-radius:12px; background:#1a1a1a; color:#eaeaea; font-size:18px; font-weight:600;")

        self.det_reason = QtWidgets.QLabel("Reason: -")
        self.det_reason.setStyleSheet("color:#cfcfcf;")
        self.det_reason.setWordWrap(True)

        self.det_conf_lbl = QtWidgets.QLabel("Confidence: -")
        self.det_conf_lbl.setStyleSheet("color:#cfcfcf;")

        self.det_timer_lbl = QtWidgets.QLabel("No-print timer: -")
        self.det_timer_lbl.setStyleSheet("color:#cfcfcf;")

        # legend (colors are deterministic by class name in printing_detection.py)
        try:
            from printing_detection import class_color
            def _sw(name: str) -> str:
                b, g, r = class_color(name)
                return f"#{r:02x}{g:02x}{b:02x}"
            nozzle_c = _sw("nozzle")
            new_c = _sw("new_deposit")
            old_c = _sw("old_deposit")
        except Exception:
            nozzle_c, new_c, old_c = "#55ddee", "#dd55ee", "#dddd55"

        self.legend_lbl = QtWidgets.QLabel(            f"<span style=\"display:inline-block;width:12px;height:12px;background:{nozzle_c};\"></span> nozzle &nbsp;&nbsp;"            f"<span style=\"display:inline-block;width:12px;height:12px;background:{new_c};\"></span> new_deposit &nbsp;&nbsp;"            f"<span style=\"display:inline-block;width:12px;height:12px;background:{old_c};\"></span> old_deposit"        )
        self.legend_lbl.setTextFormat(QtCore.Qt.RichText)
        self.legend_lbl.setStyleSheet("color:#dcdcdc;")
        small = QtGui.QFont("Segoe UI", 8)
        self.lbl_ts = QtWidgets.QLabel("t_wall: - | rs_depth_ms: - | det: -")
        self.lbl_ts.setFont(small)
        self.lbl_robot = QtWidgets.QLabel("Robot: -")
        self.lbl_robot.setFont(small)
        self.lbl_status = QtWidgets.QLabel("Status: starting...")
        self.lbl_status.setFont(small)

        form.addRow("Output base folder:", self.out_base)
        form.addRow("Next session folder:", self.next_name)
        form.addRow("Duration:", self.duration)
        form.addRow("Rotate (deg):", self.rotate_combo)
        form.addRow(self.print_pill)
        form.addRow(self.det_reason)
        form.addRow(self.det_conf_lbl)
        form.addRow(self.det_timer_lbl)
        form.addRow(self.legend_lbl)

        form.addRow(self.det_enable)
        form.addRow("Model:", self.det_model)
        form.addRow("Infer every N frames:", self.det_every)
        form.addRow("Conf threshold:", self.det_conf)
        form.addRow("Mask opacity:", self.det_alpha)
        form.addRow("Depth min:", self.depth_min)
        form.addRow("Depth max:", self.depth_max)
        form.addRow("Crop x:", self.crop_x)
        form.addRow("Crop y:", self.crop_y)
        form.addRow("Crop w:", self.crop_w)
        form.addRow("Crop h:", self.crop_h)
        form.addRow(self.btn_record)
        form.addRow(self.lbl_ts)
        form.addRow(self.lbl_robot)
        form.addRow(self.lbl_status)

        layout.addWidget(self.plot, 1, 0, 1, 2)
        layout.addWidget(ctrl, 2, 0, 1, 2)
        layout.setRowStretch(0, 5)
        layout.setRowStretch(1, 2)
        layout.setRowStretch(2, 0)


        self._detector_ui: Optional[PrintingDetector] = None

        self.update_next_name()
        self.on_depth_range_changed()
        self.on_crop_changed()

        # Now it's safe to show torch status
        if TORCH_OK:
            self.on_status("Torch/Ultralytics preload OK (imported before Qt/OpenCV)")
        else:
            self.on_status(f"Torch preload FAILED: {TORCH_ERR}")

        self.on_detection_changed()

    def update_next_name(self):
        base = Path(self.out_base.text()).expanduser()
        try:
            self.next_name.setText(next_session_name(base))
        except Exception:
            self.next_name.setText("YYMMDD_Recording_###")

    def on_rotate_changed(self, txt: str):
        try:
            deg = int(txt)
        except Exception:
            deg = 180
        self.worker.set_rotate(deg)

    def on_depth_range_changed(self):
        dmin = int(self.depth_min.value())
        dmax = int(self.depth_max.value())
        if dmax != 0 and dmin > dmax:
            dmin, dmax = dmax, dmin
            self.depth_min.setValue(dmin)
            self.depth_max.setValue(dmax)
        self.worker.set_depth_range_mm(dmin, dmax)

    def on_crop_changed(self):
        self.worker.set_crop(int(self.crop_x.value()), int(self.crop_y.value()), int(self.crop_w.value()), int(self.crop_h.value()))

    def on_detection_changed(self):
        enabled = bool(self.det_enable.isChecked())
        every_n = int(self.det_every.value())
        model = str(self.det_model.text()).strip()

        detector = None
        if enabled:
            if not TORCH_OK:
                self.on_status(f"Detection blocked: torch preload failed: {TORCH_ERR}")
                self.det_enable.setChecked(False)
                enabled = False
            else:
                try:
                    detector = PrintingDetector(model, conf=float(self.det_conf.value()), mask_alpha=float(self.det_alpha.value()))
                    self._detector_ui = detector
                    self.on_status(f"Detection ready: {model}")
                except Exception as e:
                    self._detector_ui = None
                    self.on_status(f"Detection init failed: {e}")
                    enabled = False
                    self.det_enable.setChecked(False)

        self.worker.set_detection(enabled, detector, every_n)

    def toggle_record(self):
        if self.btn_record.isChecked():
            base = Path(self.out_base.text()).expanduser()
            name = self.next_name.text().strip() or next_session_name(base)
            dur = float(self.duration.value())
            self.worker.start_recording(base, name, dur)
            self.btn_record.setText("■ Stop")
        else:
            self.worker.stop_recording()
            self.btn_record.setText("● Record")
            self.update_next_name()

    def _show_bgr_on_label(self, bgr: np.ndarray, label: QtWidgets.QLabel):
        if bgr is None:
            return
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        label.setPixmap(pix.scaled(label.width(), label.height(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    @QtCore.pyqtSlot(dict)
    def on_frame(self, payload: dict):
        rgb_full = payload.get("rgb_full", None)
        depth_prev = payload.get("depth_preview", None)
        rgb_crop_bgr = payload.get("rgb_crop_bgr", None)
        det_vis_bgr = payload.get("det_vis_bgr", None)
        det_state = payload.get("det_state", None)
        # update operator-friendly status blocks (PRINTING / NOT PRINTING)
        if det_state is not None:
            printing = bool(det_state.get("printing", False))
            nozzle_conf = float(det_state.get("nozzle_conf", 0.0))
            new_area = float(det_state.get("new_area_roi_pct", 0.0))
            overlap_tip = float(det_state.get("overlap_tip", 0.0))
            on_count = int(det_state.get("on_count", 0))
            off_count = int(det_state.get("off_count", 0))

            # state timer
            now = time.time()
            if printing:
                self._no_print_since = None
            else:
                if self._no_print_since is None:
                    self._no_print_since = now

            # pill styling
            if printing:
                self.print_pill.setText("PRINTING")
                self.print_pill.setStyleSheet("padding:10px; border-radius:12px; background:#123a24; color:#eaeaea; font-size:18px; font-weight:700;")
            else:
                self.print_pill.setText("NOT PRINTING")
                self.print_pill.setStyleSheet("padding:10px; border-radius:12px; background:#3a1414; color:#eaeaea; font-size:18px; font-weight:700;")

            reason = f"Reason: overlap_tip={overlap_tip:.3f}, new_area_roi={new_area*100:.2f}%, on={on_count}, off={off_count}"
            self.det_reason.setText(reason)
            self.det_conf_lbl.setText(f"Confidence: nozzle_conf={nozzle_conf:.2f}")

            if self._no_print_since is None:
                self.det_timer_lbl.setText("No-print timer: 0.0 s")
            else:
                self.det_timer_lbl.setText(f"No-print timer: {now - self._no_print_since:.1f} s")


        if rgb_full is not None:
            self._show_bgr_on_label(cv2.cvtColor(rgb_full, cv2.COLOR_RGB2BGR), self.rgb_full_label)
        if depth_prev is not None:
            self._show_bgr_on_label(depth_prev, self.depth_label)
        if rgb_crop_bgr is not None:
            self._show_bgr_on_label(rgb_crop_bgr, self.rgb_crop_label)
        if det_vis_bgr is not None:
            self._show_bgr_on_label(det_vis_bgr, self.det_label)

        t_wall = payload.get("t_wall", None)
        t_rs = payload.get("t_rs_depth", None)

        det_txt = "-"
        if det_state is not None:
            det_txt = (
                ("PRINTING" if det_state.get("printing", False) else "NOT PRINTING")
                + f" | roi={det_state.get('new_area_roi_pct', 0.0) * 100:.2f}%"
                + f" | ov={det_state.get('overlap_tip', 0.0):.3f}"
            )

        if t_wall is not None and t_rs is not None:
            self.lbl_ts.setText(f"t_wall: {t_wall:.3f} | rs_depth_ms: {t_rs:.1f} | det: {det_txt}")

    @QtCore.pyqtSlot(object)
    def on_robot(self, st: RobotSample):
        if st is None:
            return

        if st.axis is not None:
            t = st.t_wall - self._t0
            self._x.append(float(t))
            self._y_axes.append(st.axis.astype(np.float32))
            self._y_pwm.append(float(st.pwm) if st.pwm is not None else 0.0)

            xs = np.array(self._x, dtype=np.float32)
            ys = np.stack(self._y_axes, axis=0) if len(self._y_axes) > 0 else None
            yp = np.array(self._y_pwm, dtype=np.float32)

            if ys is not None and ys.shape[0] == xs.shape[0]:
                for i in range(6):
                    self.curves[i].setData(xs, ys[:, i])
                self.pwm_curve.setData(xs, yp)

        a = st.axis.tolist() if st.axis is not None else None
        self.lbl_robot.setText(f"Robot: axis={a} pwm={st.pwm}")

    @QtCore.pyqtSlot(str)
    def on_status(self, msg: str):
        self.lbl_status.setText(f"Status: {msg}")


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()