"""Compute velocity/direction stats for all K1 motion CSVs.

Usage:
    python compute_motion_stats.py motions_k1/k1/
    python compute_motion_stats.py motions_k1/k1/ --fps 30 --output motion_stats.json
"""

import argparse
import json
from pathlib import Path

import numpy as np


def compute_stats(qpos_data, fps):
    """Compute motion statistics from K1 qpos data.

    Returns dict with velocity, direction, speed stats.
    MuJoCo coordinate: Z-up, X-forward.
    """
    root_pos = qpos_data[:, :3]  # (N, 3) = (x, y, z)
    n_frames = len(root_pos)
    duration = (n_frames - 1) / fps

    # Per-frame velocity
    vel = np.diff(root_pos, axis=0) * fps  # (N-1, 3)
    vel_xy = vel[:, :2]  # horizontal (x, y)

    # Per-frame speed (horizontal)
    speed = np.linalg.norm(vel_xy, axis=1)  # (N-1,)

    # Average velocity over entire clip
    displacement = root_pos[-1] - root_pos[0]
    avg_vel = displacement / duration if duration > 0 else np.zeros(3)

    # Average heading direction (from root quaternion, wxyz)
    # Use displacement direction as primary direction
    avg_heading_rad = np.arctan2(displacement[1], displacement[0])

    # Yaw rate: from root quaternion changes
    root_quat = qpos_data[:, 3:7]  # (N, 4) wxyz
    # Simple yaw extraction from quaternion: yaw = atan2(2(wz+xy), 1-2(y²+z²))
    w, x, y, z = root_quat[:, 0], root_quat[:, 1], root_quat[:, 2], root_quat[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    yaw_diff = np.diff(np.unwrap(yaw))
    yaw_rate = yaw_diff * fps  # rad/s

    return {
        "n_frames": int(n_frames),
        "duration_s": round(duration, 2),
        "displacement_m": round(float(np.linalg.norm(displacement[:2])), 3),
        "avg_speed_mps": round(float(np.mean(speed)), 3),
        "max_speed_mps": round(float(np.max(speed)), 3),
        "min_speed_mps": round(float(np.min(speed)), 3),
        "avg_vel_x_mps": round(float(avg_vel[0]), 3),
        "avg_vel_y_mps": round(float(avg_vel[1]), 3),
        "avg_heading_deg": round(float(np.degrees(avg_heading_rad)), 1),
        "total_yaw_change_deg": round(float(np.degrees(yaw[-1] - yaw[0])), 1),
        "avg_yaw_rate_dps": round(float(np.degrees(np.mean(yaw_rate))), 1),
        "max_yaw_rate_dps": round(float(np.degrees(np.max(np.abs(yaw_rate)))), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", help="Directory with K1 motion CSVs")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", default=None, help="Output JSON (default: <dir>/motion_stats.json)")
    args = parser.parse_args()

    csv_files = sorted(Path(args.dir).glob("*.csv"))
    print(f"Found {len(csv_files)} motion files\n")

    all_stats = {}
    for f in csv_files:
        qpos = np.loadtxt(f, delimiter=",")
        stats = compute_stats(qpos, args.fps)
        all_stats[f.stem] = stats

    # Print summary table
    print(f"{'Name':40s} {'Dur':>4s} {'AvgSpd':>7s} {'MaxSpd':>7s} {'Heading':>8s} {'YawChg':>8s}")
    print("-" * 80)
    for name, s in sorted(all_stats.items()):
        print(f"{name:40s} {s['duration_s']:4.1f}s {s['avg_speed_mps']:6.3f} {s['max_speed_mps']:6.3f}"
              f"  {s['avg_heading_deg']:+7.1f}° {s['total_yaw_change_deg']:+7.1f}°")

    # Speed distribution summary
    speeds = [s["avg_speed_mps"] for s in all_stats.values()]
    print(f"\n--- Speed distribution ---")
    print(f"  Min avg speed: {min(speeds):.3f} m/s")
    print(f"  Max avg speed: {max(speeds):.3f} m/s")
    print(f"  Mean avg speed: {np.mean(speeds):.3f} m/s")

    # Categorize
    standing = [n for n, s in all_stats.items() if s["avg_speed_mps"] < 0.1]
    walking = [n for n, s in all_stats.items() if 0.1 <= s["avg_speed_mps"] < 1.0]
    fast = [n for n, s in all_stats.items() if s["avg_speed_mps"] >= 1.0]
    turning = [n for n, s in all_stats.items() if abs(s["total_yaw_change_deg"]) > 45]

    print(f"\n--- Categories ---")
    print(f"  Standing (< 0.1 m/s): {len(standing)} clips")
    print(f"  Walking (0.1-1.0 m/s): {len(walking)} clips")
    print(f"  Fast (> 1.0 m/s): {len(fast)} clips")
    print(f"  Turning (|yaw| > 45°): {len(turning)} clips")

    # Save JSON
    out_path = args.output or str(Path(args.dir) / "motion_stats.json")
    with open(out_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
