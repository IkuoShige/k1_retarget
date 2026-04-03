"""Browse and play K1 (or G1) motion CSVs from a directory.

Usage:
    python visualize_all.py motions_k1/k1/              # K1 motions
    python visualize_all.py motions_k1/g1/ --g1          # G1 motions
    python visualize_all.py motions_k1/k1/ --filter turn  # only turn motions
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def play_motion(model, qpos_data, fps, name):
    """Play one motion in the viewer. Returns False if viewer was closed."""
    data = mujoco.MjData(model)
    n_frames = qpos_data.shape[0]
    dt = 1.0 / fps

    print(f"\n  Playing: {name} ({n_frames} frames, {n_frames/fps:.1f}s)")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            data.qpos[:] = qpos_data[frame]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(dt)

            frame += 1
            if frame >= n_frames:
                return True  # finished, go to next
    return False  # viewer closed


def main():
    parser = argparse.ArgumentParser(description="Browse motion CSVs")
    parser.add_argument("dir", help="Directory containing CSV files")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--g1", action="store_true", help="Use G1 model (default: K1)")
    parser.add_argument("--filter", type=str, default=None, help="Filter filenames")
    parser.add_argument("--loop", action="store_true", help="Loop each motion")
    args = parser.parse_args()

    if args.g1:
        xml_path = str(Path(__file__).resolve().parent.parent.parent / "kimodo" / "kimodo" / "assets" / "skeletons" / "g1skel34" / "xml" / "g1.xml")
    else:
        xml_path = str(Path(__file__).resolve().parent.parent / "robot" / "K1_22dof.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)

    csv_files = sorted(Path(args.dir).glob("*.csv"))
    if args.filter:
        csv_files = [f for f in csv_files if args.filter in f.name]

    print(f"Found {len(csv_files)} motions ({xml_path.split('/')[-1]})")
    print("Close the viewer window to advance to the next motion.")
    print("Ctrl+C to quit.\n")

    for i, csv_file in enumerate(csv_files):
        qpos_data = np.loadtxt(csv_file, delimiter=",")
        print(f"[{i+1}/{len(csv_files)}] {csv_file.name}", end="")

        while True:
            ok = play_motion(model, qpos_data, args.fps, csv_file.stem)
            if not ok or not args.loop:
                break

    print("\nDone.")


if __name__ == "__main__":
    main()
