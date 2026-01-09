#!/usr/bin/env python3
"""
Create associations.csv linking each camera frame to nearest/interpolated robot pose.

Run:
  python make_associations.py --session dataset/session_001
"""

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np


def read_camera_ts(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["t_mono"] = float(row["t_mono"])
            rows.append(row)
    return rows


def read_robot(path: Path) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            row["t_mono"] = float(row["t_mono"])
            # floats
            for k in ["a1","a2","a3","a4","a5","a6","x","y","z","qx","qy","qz","qw"]:
                row[k] = float(row[k])
            rows.append(row)
    rows.sort(key=lambda d: d["t_mono"])
    return rows


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def interp_pose(r0: Dict, r1: Dict, t_mono: float) -> Dict:
    # Linear interpolation for position and joints.
    # Quaternion interpolation: for minimal logger, we do linear + renormalize (ok as placeholder).
    t0, t1 = r0["t_mono"], r1["t_mono"]
    if t1 <= t0:
        return r0

    alpha = (t_mono - t0) / (t1 - t0)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    out = {"t_mono": t_mono}
    for k in ["a1","a2","a3","a4","a5","a6","x","y","z","qx","qy","qz","qw"]:
        out[k] = lerp(r0[k], r1[k], alpha)

    # renormalize quaternion
    q = np.array([out["qx"], out["qy"], out["qz"], out["qw"]], dtype=np.float64)
    n = np.linalg.norm(q)
    if n > 1e-12:
        q /= n
    out["qx"], out["qy"], out["qz"], out["qw"] = q.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--max_dt", type=float, default=0.050, help="Max allowed time diff (s) for nearest match")
    args = ap.parse_args()

    base = Path(args.session)
    cam_rows = read_camera_ts(base / "camera" / "timestamps.csv")
    rob_rows = read_robot(base / "robot" / "state.csv")
    if not rob_rows:
        raise SystemExit("No robot samples found.")

    rob_t = np.array([r["t_mono"] for r in rob_rows], dtype=np.float64)

    out_path = base / "sync" / "associations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_id",
            "t_cam_mono",
            "robot_idx_nearest",
            "dt_nearest_s",
            # interpolated
            "a1","a2","a3","a4","a5","a6",
            "x","y","z","qx","qy","qz","qw"
        ])

        for c in cam_rows:
            t = c["t_mono"]
            idx = int(np.argmin(np.abs(rob_t - t)))
            dt = float(abs(rob_t[idx] - t))
            if dt > args.max_dt:
                # still write row but leave pose empty; you can filter later
                w.writerow([c["frame_id"], t, idx, dt] + [""] * 13)
                continue

            # find neighbors for interpolation
            if rob_t[idx] <= t and idx + 1 < len(rob_rows):
                r0, r1 = rob_rows[idx], rob_rows[idx + 1]
            elif rob_t[idx] > t and idx - 1 >= 0:
                r0, r1 = rob_rows[idx - 1], rob_rows[idx]
            else:
                r0 = r1 = rob_rows[idx]

            ip = interp_pose(r0, r1, t)

            w.writerow([
                c["frame_id"], t, idx, dt,
                ip["a1"], ip["a2"], ip["a3"], ip["a4"], ip["a5"], ip["a6"],
                ip["x"], ip["y"], ip["z"], ip["qx"], ip["qy"], ip["qz"], ip["qw"]
            ])

    print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
