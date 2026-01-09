#!/usr/bin/env python3
"""
ui_recorder.py (FULL REPLACEMENT)

Clean UI:
- RGB + Depth preview (rotated)
- ONE minimalist plot: Axis 1-6 + E1_ON (PWM)
  - no title
  - no axis labels
  - small fonts
  - legend on the right
  - subtle grid
- Record button + duration + auto session naming YYMMDD_Recording_###

Install:
  python -m pip install pyqt5 pyqtgraph numpy opencv-python pyrealsense2 py-openshowvar
"""

import sys
import time
import re
import json
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Deque
from collections import deque

import numpy as np
import cv2
import pyrealsense2 as rs
from py_openshowvar import openshowvar

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg


# -------- Plot style: clean + light --------
pg.setConfigOptions(antialias=True)
pg.setConfigOption("background", (250, 250, 250))
pg.setConfigOption("foreground", (30, 30, 30))


# -----------------------------
# Helpers
# -----------------------------
def _to_text(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bytes, bytearray)):
        return v.decode("utf-8", errors="ignore").replace("\x00", "").strip()
    return str(v).strip()


def _parse_num_from_struct(s: str, key: str) -> Optional[float]:
    m = re.search(rf"\b{re.escape(key)}\s+(-?\d+(?:\.\d+)?)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def frame_id_str(i: int) -> str:
    return f"{i:06d}"


def ensure_dirs(base: Path):
    (base / "camera" / "rgb").mkdir(parents=True, exist_ok=True)
    (base / "camera" / "depth").mkdir(parents=True, exist_ok=True)
    (base / "robot").mkdir(parents=True, exist_ok=True)
    (base / "calibration").mkdir(parents=True, exist_ok=True)


def save_realsense_intrinsics(base: Path, profile: rs.pipeline_profile, no_rgb: bool):
    depth_stream = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    depth_intr = depth_stream.get_intrinsics()

    intr = {
        "depth": {
            "width": depth_intr.width,
            "height": depth_intr.height,
            "ppx": depth_intr.ppx,
            "ppy": depth_intr.ppy,
            "fx": depth_intr.fx,
            "fy": depth_intr.fy,
            "model": str(depth_intr.model),
            "coeffs": list(depth_intr.coeffs),
        }
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

    with open(base / "calibration" / "camera_intrinsics.json", "w", encoding="utf-8") as f:
        json.dump(intr, f, indent=2)


def rotate_img(img: Optional[np.ndarray], deg: int) -> Optional[np.ndarray]:
    if img is None:
        return None
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
        lo = float(np.percentile(nz, 5))
        hi = float(np.percentile(nz, 95))
        hi = max(hi, lo + 1.0)

    d = np.clip((d - lo) / (hi - lo), 0.0, 1.0)
    return (d * 255.0).astype(np.uint8)


def np_bgr_to_qimage(bgr: np.ndarray) -> QtGui.QImage:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
    return qimg.copy()


def np_gray_to_qimage(gray: np.ndarray) -> QtGui.QImage:
    h, w = gray.shape[:2]
    qimg = QtGui.QImage(gray.data, w, h, w, QtGui.QImage.Format_Grayscale8)
    return qimg.copy()


def next_session_name(out_base: Path) -> str:
    out_base.mkdir(parents=True, exist_ok=True)
    date = time.strftime("%y%m%d")
    prefix = f"{date}_Recording_"
    rx = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    max_n = 0
    for p in out_base.iterdir():
        if p.is_dir():
            m = rx.match(p.name)
            if m:
                try:
                    max_n = max(max_n, int(m.group(1)))
                except ValueError:
                    pass
    return f"{prefix}{max_n + 1:03d}"


def create_next_session_dir(out_base: Path) -> Path:
    name = next_session_name(out_base)
    p = out_base / name
    p.mkdir(parents=True, exist_ok=False)
    return p


def build_step_wave(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make a square-wave polyline from samples."""
    if x.size == 0:
        return x, y
    xs = [x[0]]
    ys = [y[0]]
    for i in range(1, len(x)):
        xs.extend([x[i], x[i]])
        ys.extend([ys[-1], y[i]])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# -----------------------------
# Robot sampling
# -----------------------------
@dataclass
class RobotLite:
    t_wall: float
    t_mono: float
    a: Tuple[float, float, float, float, float, float]
    e1: float
    e_rpm: float
    extr_mod: float
    e_rpm_cmd: float
    ov_pro: float


class KukaClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.c = None

    def connect(self):
        self.c = openshowvar(self.host, self.port)

    def close(self):
        if self.c is not None:
            try:
                self.c.close()
            except Exception:
                pass
        self.c = None

    def read_text(self, var: str) -> str:
        if self.c is None:
            return ""
        try:
            return _to_text(self.c.read(var))
        except Exception:
            return ""

    def poll(self) -> Optional[RobotLite]:
        axis = self.read_text("$AXIS_ACT")
        if not axis:
            return None

        e_rpm_s = self.read_text("E_RPM")
        extr_mod_s = self.read_text("EXTR_MOD")
        ov_s = self.read_text("$OV_PRO")
        if e_rpm_s == "" or extr_mod_s == "":
            return None

        a1 = _parse_num_from_struct(axis, "A1")
        a2 = _parse_num_from_struct(axis, "A2")
        a3 = _parse_num_from_struct(axis, "A3")
        a4 = _parse_num_from_struct(axis, "A4")
        a5 = _parse_num_from_struct(axis, "A5")
        a6 = _parse_num_from_struct(axis, "A6")
        e1 = _parse_num_from_struct(axis, "E1")

        if any(v is None for v in (a1, a2, a3, a4, a5, a6)):
            return None

        try:
            e_rpm = float(e_rpm_s)
            extr_mod = float(extr_mod_s)
        except ValueError:
            return None

        e_rpm_cmd = e_rpm * extr_mod
        try:
            ov = float(ov_s) if ov_s != "" else float("nan")
        except ValueError:
            ov = float("nan")

        return RobotLite(
            t_wall=time.time(),
            t_mono=time.monotonic(),
            a=(float(a1), float(a2), float(a3), float(a4), float(a5), float(a6)),
            e1=float(e1) if e1 is not None else float("nan"),
            e_rpm=e_rpm,
            extr_mod=extr_mod,
            e_rpm_cmd=e_rpm_cmd,
            ov_pro=ov
        )


# -----------------------------
# Worker thread
# -----------------------------
class CaptureWorker(QtCore.QThread):
    sig_frame = QtCore.pyqtSignal(object)
    sig_robot = QtCore.pyqtSignal(object)
    sig_status = QtCore.pyqtSignal(str)
    sig_record_stopped = QtCore.pyqtSignal()

    def __init__(self, host: str, port: int, fps: int, width: int, height: int, no_rgb: bool, rotate_deg: int):
        super().__init__()
        self.host = host
        self.port = port
        self.fps = fps
        self.width = width
        self.height = height
        self.no_rgb = no_rgb
        self.rotate_deg = rotate_deg

        self._stop = False
        self.pipeline = None
        self.align = None
        self._rs_profile = None

        self.robot = KukaClient(host, port)

        self.recording = False
        self.record_end_mono = None
        self.session_dir: Optional[Path] = None
        self.frame_idx = 1

        self.cam_ts_f = None
        self.cam_ts_writer = None
        self.robot_f = None
        self.robot_writer = None

        self._rec_start_mono = None
        self._rec_frame_count = 0
        self._meta_path: Optional[Path] = None

    def set_rotation(self, deg: int):
        self.rotate_deg = deg

    def stop(self):
        self._stop = True

    def start_recording(self, out_dir: Path, duration_s: float):
        self.session_dir = out_dir
        self.recording = True
        self.record_end_mono = time.monotonic() + float(duration_s) if duration_s > 0 else None
        self.frame_idx = 1

        self._rec_start_mono = time.monotonic()
        self._rec_frame_count = 0

        ensure_dirs(out_dir)

        self.cam_ts_f = open(out_dir / "camera" / "timestamps.csv", "w", newline="", encoding="utf-8")
        self.cam_ts_writer = csv.writer(self.cam_ts_f)
        self.cam_ts_writer.writerow([
            "frame_id", "t_wall", "t_mono",
            "t_rs_depth_ms", "t_rs_color_ms",
            "rgb_file", "depth_file",
            "rotate_deg"
        ])

        self.robot_f = open(out_dir / "robot" / "state.csv", "w", newline="", encoding="utf-8")
        self.robot_writer = csv.writer(self.robot_f)
        self.robot_writer.writerow([
            "t_wall","t_mono",
            "a1","a2","a3","a4","a5","a6","e1",
            "E_RPM","EXTR_MOD","E_RPM_CMD","OV_PRO"
        ])

        meta = {
            "session": out_dir.name,
            "created_unix": time.time(),
            "session_naming": "YYMMDD_Recording_###",
            "camera": {
                "model": "Intel RealSense D405",
                "width": self.width,
                "height": self.height,
                "fps_configured": self.fps,
                "no_rgb": self.no_rgb,
                "rotate_deg": int(self.rotate_deg),
            },
            "robot": {
                "host": self.host,
                "port": self.port,
                "vars": ["$AXIS_ACT", "E_RPM", "EXTR_MOD", "$OV_PRO"],
            },
            "fps_measured": None
        }
        self._meta_path = out_dir / "metadata.json"
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if self._rs_profile is not None:
            try:
                save_realsense_intrinsics(out_dir, self._rs_profile, self.no_rgb)
            except Exception:
                pass

        self.sig_status.emit(f"RECORDING → {out_dir}")

    def stop_recording(self):
        fps_measured = None
        if self._rec_start_mono is not None and self._rec_frame_count > 1:
            dt = max(1e-6, time.monotonic() - self._rec_start_mono)
            fps_measured = float(self._rec_frame_count / dt)

        self.recording = False
        self.record_end_mono = None

        try:
            if self.cam_ts_f:
                self.cam_ts_f.close()
        except Exception:
            pass
        try:
            if self.robot_f:
                self.robot_f.close()
        except Exception:
            pass

        self.cam_ts_f = None
        self.cam_ts_writer = None
        self.robot_f = None
        self.robot_writer = None

        if self._meta_path is not None:
            try:
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["fps_measured"] = fps_measured
                with open(self._meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

        self.sig_status.emit(f"RECORDING STOPPED (fps_measured={fps_measured:.2f} if available)")
        self.sig_record_stopped.emit()

    def run(self):
        # Camera init
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            if not self.no_rgb:
                config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)

            self.sig_status.emit("Starting RealSense (close RealSense Viewer!) ...")
            self._rs_profile = self.pipeline.start(config)
            self.align = rs.align(rs.stream.color) if not self.no_rgb else None
            self.sig_status.emit("RealSense OK.")
        except Exception as e:
            self.sig_status.emit(f"RealSense ERROR: {e}")
            return

        # Robot init
        try:
            self.robot.connect()
            self.sig_status.emit(f"Robot connected: {self.host}:{self.port}")
        except Exception as e:
            self.sig_status.emit(f"Robot connect ERROR: {e} (continuing without robot)")
            self.robot = None

        while not self._stop:
            if self.recording and self.record_end_mono is not None and time.monotonic() >= self.record_end_mono:
                self.stop_recording()

            # Robot sample
            st = None
            if self.robot is not None:
                try:
                    st = self.robot.poll()
                except Exception:
                    st = None

            if st is not None:
                self.sig_robot.emit(st)
                if self.recording and self.robot_writer is not None:
                    self.robot_writer.writerow([
                        st.t_wall, st.t_mono,
                        *st.a, st.e1,
                        st.e_rpm, st.extr_mod, st.e_rpm_cmd, st.ov_pro
                    ])
                    self.robot_f.flush()

            # Camera sample
            try:
                frames = self.pipeline.wait_for_frames(15000)
                t_wall = time.time()
                t_mono = time.monotonic()

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

                payload = {
                    "t_wall": t_wall,
                    "t_mono": t_mono,
                    "t_rs_depth": depth_frame.get_timestamp(),
                    "rgb": rgb,
                    "depth_preview": depth_to_preview(depth),
                    "rotate_deg": int(self.rotate_deg),
                }
                self.sig_frame.emit(payload)

                if self.recording and self.session_dir is not None and self.cam_ts_writer is not None:
                    fid = frame_id_str(self.frame_idx)

                    depth_rel = Path("camera") / "depth" / f"{fid}.png"
                    cv2.imwrite(str(self.session_dir / depth_rel), depth)

                    rgb_rel = ""
                    if rgb is not None:
                        rgb_rel_path = Path("camera") / "rgb" / f"{fid}.png"
                        cv2.imwrite(str(self.session_dir / rgb_rel_path), rgb)
                        rgb_rel = str(rgb_rel_path).replace("\\", "/")

                    self.cam_ts_writer.writerow([
                        fid, t_wall, t_mono,
                        payload["t_rs_depth"], (color_frame.get_timestamp() if color_frame else None),
                        rgb_rel, str(depth_rel).replace("\\", "/"),
                        int(self.rotate_deg)
                    ])
                    self.cam_ts_f.flush()
                    self.frame_idx += 1
                    self._rec_frame_count += 1

            except Exception as e:
                self.sig_status.emit(f"Capture ERROR: {e}")
                time.sleep(0.2)

        # Cleanup
        try:
            if self.pipeline:
                self.pipeline.stop()
        except Exception:
            pass
        try:
            if self.robot:
                self.robot.close()
        except Exception:
            pass
        if self.recording:
            self.stop_recording()

        self.sig_status.emit("Worker stopped.")


# -----------------------------
# Main UI
# -----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth–Pose Logger UI (minimal plot)")
        self.resize(1250, 720)

        # Defaults
        self.robot_host = "10.1.0.121"
        self.robot_port = 7000
        self.fps_cfg = 30
        self.width = 640
        self.height = 480
        self.no_rgb = False
        self.rotate_deg = 180

        # Plot window
        self.plot_window_s = 10.0

        # Data buffers
        self.t_buf: Deque[float] = deque(maxlen=6000)
        self.a_buf = [deque(maxlen=6000) for _ in range(6)]
        self.e1_on_buf: Deque[int] = deque(maxlen=6000)

        # E1 ON detection from E1 velocity
        self._last_e1 = None
        self._last_e1_t = None
        self.e1_vel_threshold = 5.0  # tune if needed

        # UI FPS estimate
        self.frame_t_buf: Deque[float] = deque(maxlen=120)

        self._build_ui()

        self.worker = CaptureWorker(
            self.robot_host, self.robot_port,
            self.fps_cfg, self.width, self.height,
            self.no_rgb, self.rotate_deg
        )
        self.worker.sig_frame.connect(self.on_frame)
        self.worker.sig_robot.connect(self.on_robot)
        self.worker.sig_status.connect(self.on_status)
        self.worker.sig_record_stopped.connect(self.update_next_name)
        self.worker.start()

        self.update_next_name()

    def _small_axis(self, axis: pg.AxisItem):
        font = QtGui.QFont("Segoe UI", 8)
        axis.setTickFont(font)
        axis.setTextPen(pg.mkPen((40, 40, 40)))
        axis.setPen(pg.mkPen((60, 60, 60), width=1))

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        layout = QtWidgets.QGridLayout(central)
        layout.setColumnStretch(0, 2)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 3)

        # Images
        self.rgb_label = QtWidgets.QLabel("RGB")
        self.rgb_label.setAlignment(QtCore.Qt.AlignCenter)
        self.rgb_label.setMinimumSize(320, 240)
        self.rgb_label.setStyleSheet("QLabel { background: #111; color: #ddd; }")

        self.depth_label = QtWidgets.QLabel("Depth")
        self.depth_label.setAlignment(QtCore.Qt.AlignCenter)
        self.depth_label.setMinimumSize(320, 240)
        self.depth_label.setStyleSheet("QLabel { background: #111; color: #ddd; }")

        # Plot
        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.setMenuEnabled(False)

        # Remove labels/titles
        self.plot.setTitle("")
        self.plot.setLabel("left", "")
        self.plot.setLabel("bottom", "")

        # Smaller ticks + cleaner axes
        self._small_axis(self.plot.getAxis("bottom"))
        self._small_axis(self.plot.getAxis("left"))

        # Make plot border subtle
        self.plot.getPlotItem().setContentsMargins(10, 10, 10, 10)

        # Curves A1..A6 (thinner)
        self.curves = []
        curve_names = ["Axis 01", "Axis 02", "Axis 03", "Axis 04", "Axis 05", "Axis 06"]
        for i in range(6):
            pen = pg.mkPen(pg.intColor(i, hues=6), width=2)
            c = self.plot.plot([], [], pen=pen, name=curve_names[i])
            c.setDownsampling(auto=True, method="peak")
            c.setClipToView(True)
            self.curves.append(c)

        # E1_ON PWM (thin dark-green)
        self.e1_curve = self.plot.plot([], [], pen=pg.mkPen((0, 110, 0), width=2), name="E1_ON")

        # Legend on the right, small font
        self.legend = self.plot.addLegend(offset=(15, 15))
        self.legend.setParentItem(self.plot.getPlotItem())
        self.legend.anchor(itemPos=(1, 0), parentPos=(1, 0), offset=(-10, 10))
        self.legend.setLabelTextSize("8pt")

        # Controls
        ctrl = QtWidgets.QGroupBox("Controls")
        form = QtWidgets.QFormLayout(ctrl)

        self.out_base = QtWidgets.QLineEdit(str(Path("dataset").resolve()))
        self.out_base.textChanged.connect(lambda: self.update_next_name())

        self.next_name = QtWidgets.QLineEdit("")
        self.next_name.setReadOnly(True)

        self.duration = QtWidgets.QDoubleSpinBox()
        self.duration.setRange(0.0, 36000.0)
        self.duration.setValue(10.0)
        self.duration.setSuffix(" s (0=manual stop)")
        self.duration.setDecimals(1)

        self.rotate_combo = QtWidgets.QComboBox()
        self.rotate_combo.addItems(["180", "0", "90", "-90"])
        self.rotate_combo.setCurrentText("180")
        self.rotate_combo.currentTextChanged.connect(self.on_rotate_changed)

        self.btn_record = QtWidgets.QPushButton("● Record")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.toggle_record)

        # Small status text
        small = QtGui.QFont("Segoe UI", 8)
        self.lbl_ts = QtWidgets.QLabel("t_wall: - | fps_est: - | rs_depth_ms: -")
        self.lbl_ts.setFont(small)
        self.lbl_robot = QtWidgets.QLabel("Robot: -")
        self.lbl_robot.setFont(small)
        self.lbl_status = QtWidgets.QLabel("Status: starting...")
        self.lbl_status.setFont(small)

        form.addRow("Output base folder:", self.out_base)
        form.addRow("Next session folder:", self.next_name)
        form.addRow("Duration:", self.duration)
        form.addRow("Rotate (deg):", self.rotate_combo)
        form.addRow(self.btn_record)
        form.addRow("Timestamps:", self.lbl_ts)
        form.addRow("Robot:", self.lbl_robot)
        form.addRow("Status:", self.lbl_status)

        layout.addWidget(self.rgb_label, 0, 0)
        layout.addWidget(self.depth_label, 0, 1)
        layout.addWidget(self.plot, 0, 2)
        layout.addWidget(ctrl, 1, 0, 1, 3)

    def update_next_name(self):
        base = Path(self.out_base.text()).expanduser()
        try:
            self.next_name.setText(next_session_name(base))
        except Exception:
            self.next_name.setText("YYMMDD_Recording_###")

    def on_rotate_changed(self, txt: str):
        try:
            deg = int(txt)
        except ValueError:
            deg = 180
        self.rotate_deg = deg
        if self.worker is not None:
            self.worker.set_rotation(deg)

    def _fps_estimate(self) -> Optional[float]:
        if len(self.frame_t_buf) < 10:
            return None
        dt = self.frame_t_buf[-1] - self.frame_t_buf[0]
        if dt <= 1e-6:
            return None
        return (len(self.frame_t_buf) - 1) / dt

    @QtCore.pyqtSlot(object)
    def on_frame(self, payload: Dict[str, Any]):
        self.frame_t_buf.append(payload["t_mono"])
        fps_est = self._fps_estimate()
        fps_txt = f"{fps_est:.2f}" if fps_est is not None else "-"

        self.lbl_ts.setText(
            f"t_wall: {payload['t_wall']:.3f} | fps_est: {fps_txt} | rs_depth_ms: {payload['t_rs_depth']:.1f}"
        )

        rgb = payload["rgb"]
        if rgb is not None:
            qimg = np_bgr_to_qimage(rgb)
            pix = QtGui.QPixmap.fromImage(qimg).scaled(
                self.rgb_label.width(), self.rgb_label.height(),
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
            )
            self.rgb_label.setPixmap(pix)
        else:
            self.rgb_label.setText("RGB disabled")

        dp = payload["depth_preview"]
        qimg_d = np_gray_to_qimage(dp)
        pix_d = QtGui.QPixmap.fromImage(qimg_d).scaled(
            self.depth_label.width(), self.depth_label.height(),
            QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.depth_label.setPixmap(pix_d)

    def _e1_on_from_velocity(self, t_mono: float, e1: float) -> int:
        if np.isnan(e1):
            return 0
        if self._last_e1 is None or self._last_e1_t is None:
            self._last_e1 = e1
            self._last_e1_t = t_mono
            return 0
        dt = t_mono - self._last_e1_t
        if dt <= 1e-6:
            return 0
        vel = abs((e1 - self._last_e1) / dt)
        self._last_e1 = e1
        self._last_e1_t = t_mono
        return 1 if vel >= self.e1_vel_threshold else 0

    @QtCore.pyqtSlot(object)
    def on_robot(self, st: RobotLite):
        self.lbl_robot.setText(
            f"A1..A6: {[round(x,2) for x in st.a]} | E1: {st.e1:.1f} | E_RPM: {st.e_rpm:.2f} | OV: {st.ov_pro:.1f}"
        )

        t_now = st.t_mono
        self.t_buf.append(t_now)
        for i in range(6):
            self.a_buf[i].append(st.a[i])

        e1_on = self._e1_on_from_velocity(t_now, st.e1)
        self.e1_on_buf.append(e1_on)

        # crop window
        while self.t_buf and (t_now - self.t_buf[0]) > self.plot_window_s:
            self.t_buf.popleft()
            self.e1_on_buf.popleft()
            for i in range(6):
                self.a_buf[i].popleft()

        if len(self.t_buf) < 2:
            return

        t0 = self.t_buf[0]
        x = np.array([tt - t0 for tt in self.t_buf], dtype=np.float32)

        # Plot axes
        y_all = []
        for i in range(6):
            y = np.array(self.a_buf[i], dtype=np.float32)
            self.curves[i].setData(x, y)
            if y.size:
                y_all.append(y)

        # Place PWM track at top with small headroom
        if y_all:
            y_stack = np.concatenate(y_all)
            y_min = float(np.nanmin(y_stack))
            y_max = float(np.nanmax(y_stack))
        else:
            y_min, y_max = -180.0, 180.0

        pad = 15.0
        pwm_low = y_max + pad
        pwm_high = y_max + pad + 12.0  # smaller PWM height (cleaner)

        s = np.array(self.e1_on_buf, dtype=np.float32)
        y_pwm = np.where(s > 0.5, pwm_high, pwm_low).astype(np.float32)

        xs, ys = build_step_wave(x, y_pwm)
        self.e1_curve.setData(xs, ys)

        self.plot.setYRange(y_min - pad, pwm_high + pad, padding=0.0)

    @QtCore.pyqtSlot(str)
    def on_status(self, msg: str):
        self.lbl_status.setText(f"Status: {msg}")
        if "RECORDING STOPPED" in msg:
            self.update_next_name()

    def toggle_record(self, checked: bool):
        base = Path(self.out_base.text()).expanduser()
        if checked:
            out_dir = create_next_session_dir(base)
            dur = float(self.duration.value())
            self.worker.start_recording(out_dir, dur)
            self.btn_record.setText("■ Stop")
            self.update_next_name()
        else:
            self.worker.stop_recording()
            self.btn_record.setText("● Record")
            self.update_next_name()

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.worker.stop()
            self.worker.wait(2000)
        except Exception:
            pass
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
