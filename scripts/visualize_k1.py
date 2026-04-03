"""Visualize K1 motion CSV in MuJoCo viewer.

Usage:
    python visualize_k1.py walking_g1_k1.csv
    python visualize_k1.py walking_g1_k1.csv --fps 30
    python visualize_k1.py walking_g1_k1.csv --k1-xml path/to/K1_22dof.xml
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize K1 motion in MuJoCo")
    parser.add_argument("csv", help="K1 qpos CSV file (29 columns: 7 root + 22 joints)")
    parser.add_argument("--fps", type=float, default=30.0, help="Playback FPS (default: 30)")
    parser.add_argument(
        "--k1-xml",
        default=str(Path(__file__).resolve().parent.parent / "robot" / "K1_22dof.xml"),
        help="Path to K1 MuJoCo XML",
    )
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    args = parser.parse_args()

    # Load model and data
    model = mujoco.MjModel.from_xml_path(args.k1_xml)
    data = mujoco.MjData(model)
    qpos_data = np.loadtxt(args.csv, delimiter=",")

    n_frames = qpos_data.shape[0]
    print(f"Loaded {n_frames} frames, playing at {args.fps} FPS")
    print(f"Duration: {n_frames / args.fps:.1f}s")
    print("Controls: close the viewer window to exit")

    dt = 1.0 / args.fps

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame = 0
        while viewer.is_running():
            # Set qpos from CSV
            qpos = qpos_data[frame]
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(dt)

            frame += 1
            if frame >= n_frames:
                if args.loop:
                    frame = 0
                else:
                    print("Playback complete. Close viewer window to exit.")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
                    break


if __name__ == "__main__":
    main()
