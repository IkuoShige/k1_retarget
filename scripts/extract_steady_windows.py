"""Extract steady-speed windows from existing K1 motion clips.

Many clips have high max speed (1.5-2.3 m/s) but low average due to
acceleration/deceleration phases. This script finds contiguous windows
where speed stays above a threshold and extracts them as new clips.

Usage:
    python extract_steady_windows.py motions_k1/k1_fixed/ --dry-run
    python extract_steady_windows.py motions_k1/k1_fixed/
    python extract_steady_windows.py motions_k1/k1_fixed/ --min-speed 1.5 --min-duration 1.0
"""

import argparse
import json
from pathlib import Path

import numpy as np


def compute_frame_speed(qpos, fps, smooth_window=5):
    """Compute per-frame horizontal speed, optionally smoothed."""
    root_xy = qpos[:, :2]
    vel = np.diff(root_xy, axis=0) * fps
    speed = np.linalg.norm(vel, axis=1)
    # Smooth with moving average to avoid single-frame noise
    if smooth_window > 1 and len(speed) >= smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        speed = np.convolve(speed, kernel, mode="same")
    return speed


def find_best_window(speed, min_speed, min_frames):
    """Find the longest contiguous window where speed >= min_speed.

    Returns (start, end) frame indices, or None if no valid window.
    """
    above = speed >= min_speed
    best_start, best_len = 0, 0
    cur_start, cur_len = 0, 0

    for i, a in enumerate(above):
        if a:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
        else:
            if cur_len > best_len:
                best_start, best_len = cur_start, cur_len
            cur_len = 0
    if cur_len > best_len:
        best_start, best_len = cur_start, cur_len

    if best_len < min_frames:
        return None
    return (best_start, best_start + best_len)


def extract_window(qpos, start, end):
    """Extract qpos window and reset root XY to start from origin."""
    clip = qpos[start:end + 1].copy()  # +1 because speed has N-1 frames
    # Reset root XY so clip starts at origin
    clip[:, 0] -= clip[0, 0]
    clip[:, 1] -= clip[0, 1]
    return clip


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input_dir", help="Directory with K1 fixed CSVs")
    parser.add_argument("--output-dir", default=None,
                        help="Output dir (default: same as input)")
    parser.add_argument("--min-speed", type=float, default=1.2,
                        help="Minimum speed threshold for window (m/s, default 1.2)")
    parser.add_argument("--min-duration", type=float, default=1.5,
                        help="Minimum window duration (seconds, default 1.5)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--max-heading-deg", type=float, default=40.0,
                        help="Skip clips with heading > this (not forward)")
    parser.add_argument("--prefix", default="steady",
                        help="Prefix for output files (default: steady)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    min_frames = int(args.min_duration * args.fps)

    # Load stats to pre-filter candidates
    stats_path = input_dir / "motion_stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    else:
        stats = {}

    csv_files = sorted(input_dir.glob("*.csv"))
    # Skip already-extracted clips
    csv_files = [f for f in csv_files if not f.stem.startswith(args.prefix + "_")]

    candidates = []
    for f in csv_files:
        s = stats.get(f.stem, {})
        max_spd = s.get("max_speed_mps", 0)
        heading = abs(s.get("avg_heading_deg", 0))
        yaw = abs(s.get("total_yaw_change_deg", 0))
        # Only consider clips with sufficient max speed, forward-ish, low yaw
        if max_spd >= args.min_speed and heading < args.max_heading_deg and yaw < 60:
            candidates.append(f)

    print(f"Candidates (max_speed >= {args.min_speed}, forward): {len(candidates)}")
    print(f"Min window: {args.min_duration}s ({min_frames} frames)")

    results = []
    for f in candidates:
        qpos = np.loadtxt(f, delimiter=",")
        speed = compute_frame_speed(qpos, args.fps)
        window = find_best_window(speed, args.min_speed, min_frames)

        if window is None:
            continue

        start, end = window
        win_speed = speed[start:end]
        win_dur = len(win_speed) / args.fps

        results.append({
            "source": f.stem,
            "start": start,
            "end": end,
            "duration": win_dur,
            "avg_speed": float(np.mean(win_speed)),
            "min_speed": float(np.min(win_speed)),
            "max_speed": float(np.max(win_speed)),
            "std_speed": float(np.std(win_speed)),
        })

    # Sort by avg speed descending
    results.sort(key=lambda r: -r["avg_speed"])

    print(f"\nFound {len(results)} extractable windows:\n")
    print(f"{'Source':35s} {'Frames':>8s} {'Dur':>5s} {'AvgSpd':>7s} "
          f"{'MinSpd':>7s} {'MaxSpd':>7s} {'StdSpd':>7s}")
    print("-" * 90)
    for r in results:
        print(f"{r['source']:35s} {r['start']:3d}-{r['end']:3d}"
              f"  {r['duration']:4.1f}s {r['avg_speed']:6.3f} "
              f" {r['min_speed']:6.3f}  {r['max_speed']:6.3f}  {r['std_speed']:6.3f}")

    if args.dry_run:
        print("\n(dry-run, no files written)")
        return

    # Extract and save
    count = 0
    for i, r in enumerate(results):
        src = input_dir / f"{r['source']}.csv"
        qpos = np.loadtxt(src, delimiter=",")
        clip = extract_window(qpos, r["start"], r["end"])

        out_name = f"{args.prefix}_{r['avg_speed']:.1f}mps_{r['source']}.csv"
        out_path = output_dir / out_name
        np.savetxt(out_path, clip, delimiter=",")
        print(f"  Saved: {out_name} ({clip.shape[0]} frames, "
              f"avg {r['avg_speed']:.3f} m/s)")
        count += 1

    print(f"\nExtracted {count} steady clips -> {output_dir}")
    print("Next: run compute_motion_stats.py to update stats")


if __name__ == "__main__":
    main()
