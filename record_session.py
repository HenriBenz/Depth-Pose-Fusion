#!/usr/bin/env python3
"""
record_session.py (FULL REPLACEMENT)

Logs:
- Intel RealSense D405 RGB + Depth frames (rotated BEFORE saving)
- Camera timestamps (laptop + RealSense timestamps)
- KUKA robot state via KUKAVARPROXY / OpenShowVar protocol using py-openshowvar

Session folder naming (AUTO):
- Creates a new session folder inside --out_base (default: dataset)
- Name scheme: YYMMDD_Recording_###  (counter increases per day)

Examples:
  python record_session.py --duration 10
  python record_session.py --out_base dataset --duration 10 --rotate 180
  python record_session.py --out_base D:/logs --duration 0

Install:
  python -m pip install numpy opencv-python pyyaml pyrealsense2 py-openshowvar
"""

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import numpy as np
import yaml
import cv2
import pyrealsense2 as rs

from py_openshowvar import openshowvar


# ----------------------------
# Data model
# ----------------------------
@dataclass
class RobotSample:
    t_wall: float
    t_mono: float
    t_robot: Optional[float]

    a1: float; a2: float; a3: float; a4: float; a5: float; a6: float
    e1: float  # NaN if unavailable

    # pose optional in this setup
    x: float; y: float; z: float
    qx: float; qy: float; qz: float; qw: float

    E_RPM: float
    EXTR_MOD: float
    E_RPM_CMD: float
    EXTR_CMD: int
    OV_PRO: float  # NaN if unavailable


# ----------------------------
# Helpers
# ----------------------------
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


def euler_zyx_deg_to_quat(A: float, B: float, C: float) -> Tuple[float, float, float, float]:
    yaw = math.radians(A)
    pitch = math.radians(B)
    roll = math.radians(C)

    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n > 1e-12:
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    else:
        qx = qy = qz = qw = float("nan")
    return qx, qy, qz, qw


def rotate_img(img: Optional[np.ndarray], deg: int) -> Optional[np.ndarray]:
    """Rotate image by 0/90/180/-90 degrees."""
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

    # fallback for arbitrary angles
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    interp = cv2.INTER_NEAREST if img.dtype != np.uint8 else cv2.INTER_LINEAR
    return cv2.warpAffine(img, M, (w, h), flags=interp)


def ensure_dirs(base: Path):
    (base / "camera" / "rgb").mkdir(parents=True, exist_ok=True)
    (base / "camera" / "depth").mkdir(parents=True, exist_ok=True)
    (base / "robot").mkdir(parents=True, exist_ok=True)
    (base / "sync").mkdir(parents=True, exist_ok=True)
    (base / "calibration").mkdir(parents=True, exist_ok=True)


def next_session_dir(out_base: Path) -> Path:
    """
    Creates/returns next session directory with scheme:
      YYMMDD_Recording_###  (### starts at 001 per day)
    """
    out_base.mkdir(parents=True, exist_ok=True)
    date = time.strftime("%y%m%d")
    prefix = f"{date}_Recording_"
    rx = re.compile(rf"^{re.escape(prefix)}(\d+)$")

    max_n = 0
    for p in out_base.iterdir():
        if not p.is_dir():
            continue
        m = rx.match(p.name)
        if m:
            try:
                max_n = max(max_n, int(m.group(1)))
            except ValueError:
                pass

    new_name = f"{prefix}{max_n + 1:03d}"
    session = out_base / new_name
    session.mkdir(parents=True, exist_ok=False)
    return session


def write_metadata(base: Path, args: argparse.Namespace):
    meta: Dict[str, Any] = {
        "session": base.name,
        "created_unix": time.time(),
        "robot": {
            "host": args.robot_host,
            "port": args.robot_port,
            "vars": ["$AXIS_ACT", "E_RPM", "EXTR_MOD", "$OV_PRO", "$POS_ACT/$POS_ACT_MES/$POS_INT (optional)"],
            "rpm_on_threshold": args.rpm_on_threshold,
            "pose_optional": True
        },
        "camera": {
            "model": "Intel RealSense D405",
            "serial": args.camera_serial or "",
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "no_rgb": bool(args.no_rgb),
            "rotate_deg": int(args.rotate),
        },
        "session_naming": "YYMMDD_Recording_###",
        "notes": "Close RealSenseViewer.exe while logging."
    }
    with open(base / "metadata.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False)


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


def frame_id_str(i: int) -> str:
    return f"{i:06d}"


# ----------------------------
# Robot client
# ----------------------------
class KukaVarProxyClient:
    def __init__(self, host: str, port: int, rpm_on_threshold: float = 2.0, debug: bool = False):
        self.host = host
        self.port = port
        self.rpm_on_threshold = rpm_on_threshold
        self.debug = debug
        self.client = None

    def connect(self):
        print(f"[ROBOT] Connecting to {self.host}:{self.port} ...")
        self.client = openshowvar(self.host, self.port)
        print("[ROBOT] Connected.")

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        self.client = None

    def read_text(self, var: str) -> str:
        if self.client is None:
            return ""
        try:
            val = self.client.read(var)
            txt = _to_text(val)
            if self.debug:
                print(f"[ROBOT] {var} => {txt!r}")
            return txt
        except Exception as e:
            if self.debug:
                print(f"[ROBOT] READ ERROR {var}: {e}")
            return ""

    def poll(self) -> Optional[RobotSample]:
        axis_s = self.read_text("$AXIS_ACT")
        if not axis_s:
            return None

        e_rpm_s = self.read_text("E_RPM")
        extr_mod_s = self.read_text("EXTR_MOD")
        if e_rpm_s == "" or extr_mod_s == "":
            return None

        a1 = _parse_num_from_struct(axis_s, "A1")
        a2 = _parse_num_from_struct(axis_s, "A2")
        a3 = _parse_num_from_struct(axis_s, "A3")
        a4 = _parse_num_from_struct(axis_s, "A4")
        a5 = _parse_num_from_struct(axis_s, "A5")
        a6 = _parse_num_from_struct(axis_s, "A6")
        e1 = _parse_num_from_struct(axis_s, "E1")

        if any(v is None for v in [a1, a2, a3, a4, a5, a6]):
            return None

        try:
            E_RPM = float(e_rpm_s)
            EXTR_MOD = float(extr_mod_s)
        except ValueError:
            return None

        E_RPM_CMD = E_RPM * EXTR_MOD
        EXTR_CMD = 1 if E_RPM_CMD >= self.rpm_on_threshold else 0

        ov_s = self.read_text("$OV_PRO")
        try:
            OV_PRO = float(ov_s) if ov_s != "" else float("nan")
        except ValueError:
            OV_PRO = float("nan")

        # Pose optional
        pos_s = ""
        for name in ("$POS_ACT", "$POS_ACT_MES", "$POS_INT"):
            pos_s = self.read_text(name)
            if pos_s:
                break

        if pos_s:
            x = _parse_num_from_struct(pos_s, "X")
            y = _parse_num_from_struct(pos_s, "Y")
            z = _parse_num_from_struct(pos_s, "Z")
            A = _parse_num_from_struct(pos_s, "A")
            B = _parse_num_from_struct(pos_s, "B")
            C = _parse_num_from_struct(pos_s, "C")
        else:
            x = y = z = A = B = C = None

        if None in (x, y, z, A, B, C):
            x = y = z = float("nan")
            qx = qy = qz = qw = float("nan")
        else:
            qx, qy, qz, qw = euler_zyx_deg_to_quat(float(A), float(B), float(C))

        return RobotSample(
            t_wall=time.time(),
            t_mono=time.monotonic(),
            t_robot=None,
            a1=float(a1), a2=float(a2), a3=float(a3), a4=float(a4), a5=float(a5), a6=float(a6),
            e1=float(e1) if e1 is not None else float("nan"),
            x=float(x), y=float(y), z=float(z),
            qx=qx, qy=qy, qz=qz, qw=qw,
            E_RPM=E_RPM,
            EXTR_MOD=EXTR_MOD,
            E_RPM_CMD=E_RPM_CMD,
            EXTR_CMD=EXTR_CMD,
            OV_PRO=OV_PRO
        )


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="dataset", help="Base folder for sessions (default: dataset)")
    ap.add_argument("--duration", type=float, default=0.0, help="Seconds to record (0 = until Ctrl+C)")

    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--no_rgb", action="store_true", help="Record depth only (no color stream)")
    ap.add_argument("--camera_serial", default="", help="RealSense serial number (optional)")
    ap.add_argument("--wait_timeout_ms", type=int, default=15000, help="wait_for_frames timeout in ms")
    ap.add_argument("--rotate", type=int, default=180, help="Rotate images: 0, 90, -90, 180 (default 180)")

    ap.add_argument("--robot_host", default="10.1.0.121")
    ap.add_argument("--robot_port", type=int, default=7000)
    ap.add_argument("--robot_disable", action="store_true", help="Do not connect/log robot data")
    ap.add_argument("--rpm_on_threshold", type=float, default=2.0)
    ap.add_argument("--robot_debug", action="store_true", help="Verbose robot reads")

    args = ap.parse_args()

    if args.rotate not in (0, 90, -90, 180):
        print("[WARN] --rotate should be 0, 90, -90, or 180. Using 180.")
        args.rotate = 180

    out_base = Path(args.out_base).expanduser()
    base = next_session_dir(out_base)  # <-- AUTO session name

    ensure_dirs(base)
    write_metadata(base, args)

    cam_ts_file = open(base / "camera" / "timestamps.csv", "w", newline="", encoding="utf-8")
    robot_file = open(base / "robot" / "state.csv", "w", newline="", encoding="utf-8")

    cam_ts_writer = csv.writer(cam_ts_file)
    cam_ts_writer.writerow([
        "frame_id", "t_wall", "t_mono",
        "t_rs_depth_ms", "t_rs_color_ms",
        "rgb_file", "depth_file",
        "rotate_deg"
    ])

    robot_writer = csv.writer(robot_file)
    robot_writer.writerow([
        "t_wall","t_mono","t_robot",
        "a1","a2","a3","a4","a5","a6","e1",
        "x","y","z","qx","qy","qz","qw",
        "E_RPM","EXTR_MOD","E_RPM_CMD","EXTR_CMD","OV_PRO"
    ])

    pipeline = rs.pipeline()
    config = rs.config()

    if args.camera_serial:
        config.enable_device(args.camera_serial)

    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    if not args.no_rgb:
        config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    print(f"[INFO] Session folder: {base.resolve()}")
    print("[RS] Starting pipeline ... (close RealSense Viewer!)")
    profile = pipeline.start(config)
    save_realsense_intrinsics(base, profile, args.no_rgb)
    align = rs.align(rs.stream.color) if not args.no_rgb else None

    robot = None
    if not args.robot_disable:
        robot = KukaVarProxyClient(args.robot_host, args.robot_port,
                                   rpm_on_threshold=args.rpm_on_threshold,
                                   debug=args.robot_debug)
        try:
            robot.connect()
        except Exception as e:
            print(f"[WARN] Robot connect failed: {e}")
            robot = None

    print("[INFO] Press Ctrl+C to stop.")
    t0 = time.time()
    frame_idx = 1
    robot_rows = 0

    try:
        while True:
            if args.duration > 0 and (time.time() - t0) >= args.duration:
                break

            # Robot: multiple samples per camera frame
            if robot is not None:
                for _ in range(30):
                    st = robot.poll()
                    if st is None:
                        break
                    robot_writer.writerow([
                        st.t_wall, st.t_mono, st.t_robot if st.t_robot is not None else "",
                        st.a1, st.a2, st.a3, st.a4, st.a5, st.a6, st.e1,
                        st.x, st.y, st.z, st.qx, st.qy, st.qz, st.qw,
                        st.E_RPM, st.EXTR_MOD, st.E_RPM_CMD, st.EXTR_CMD, st.OV_PRO
                    ])
                    robot_rows += 1
                robot_file.flush()

            frames = pipeline.wait_for_frames(args.wait_timeout_ms)
            t_wall = time.time()
            t_mono = time.monotonic()

            if align is not None:
                frames = align.process(frames)

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame() if not args.no_rgb else None
            if not depth_frame:
                continue

            t_rs_depth = depth_frame.get_timestamp()
            t_rs_color = color_frame.get_timestamp() if color_frame else None

            fid = frame_id_str(frame_idx)

            depth = np.asanyarray(depth_frame.get_data())  # uint16
            rgb = np.asanyarray(color_frame.get_data()) if color_frame else None  # BGR uint8

            # Rotate BEFORE saving
            depth = rotate_img(depth, args.rotate)
            if rgb is not None:
                rgb = rotate_img(rgb, args.rotate)

            depth_rel = Path("camera") / "depth" / f"{fid}.png"
            cv2.imwrite(str(base / depth_rel), depth)

            rgb_rel = ""
            if rgb is not None:
                rgb_rel_path = Path("camera") / "rgb" / f"{fid}.png"
                cv2.imwrite(str(base / rgb_rel_path), rgb)
                rgb_rel = str(rgb_rel_path).replace("\\", "/")

            cam_ts_writer.writerow([
                fid, t_wall, t_mono,
                t_rs_depth, t_rs_color,
                rgb_rel, str(depth_rel).replace("\\", "/"),
                int(args.rotate)
            ])
            cam_ts_file.flush()

            frame_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        if robot is not None:
            try:
                robot.close()
            except Exception:
                pass
        cam_ts_file.close()
        robot_file.close()

    print(f"[INFO] Done. Robot rows written: {robot_rows}")


if __name__ == "__main__":
    main()
