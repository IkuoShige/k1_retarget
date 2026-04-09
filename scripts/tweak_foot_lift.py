"""Boost foot lift in an existing kimodo G1 motion via knee-bend keyframe constraints.

Reads an existing kimodo G1 NPZ, finds swing-peak frames using foot_contacts, builds
left-foot / right-foot end-effector constraints with extra knee flexion at those frames,
re-runs kimodo generate to produce a new motion that lifts the feet a bit higher,
then retargets the result to K1.

Usage:
    python tweak_foot_lift.py motions_k1/g1/turn_in_place_left/turn_in_place_left_00.npz \
        --prompt "a person turning around in place to the left" \
        --duration 4.0 --seed 1642 \
        --output-name turn_in_place_v2_left_00

By default writes:
    motions_k1/g1/turn_in_place_v2/<output-name>.{npz,csv}
    motions_k1/k1/<output-name>.csv
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

# G1Skeleton34 joint indices (verified via load_model('kimodo-g1-rp').skeleton.bone_index)
LEFT_ANKLE_ROLL = 6
RIGHT_ANKLE_ROLL = 13
LEFT_KNEE = 4
RIGHT_KNEE = 11

# foot_contacts columns: [left_heel, left_toe, right_heel, right_toe]


def find_swing_peaks(foot_contacts: np.ndarray, foot_y: np.ndarray, side: str):
    """Return list of frame indices where this foot is at peak swing height."""
    if side == "left":
        swing = ~foot_contacts[:, 0] & ~foot_contacts[:, 1]
    else:
        swing = ~foot_contacts[:, 2] & ~foot_contacts[:, 3]

    peaks = []
    in_group = False
    start = 0
    for i, s in enumerate(swing):
        if s and not in_group:
            start = i
            in_group = True
        elif not s and in_group:
            seg = foot_y[start:i]
            peaks.append(start + int(np.argmax(seg)))
            in_group = False
    if in_group:
        seg = foot_y[start:]
        peaks.append(start + int(np.argmax(seg)))
    return peaks


def build_constraint(local_rot_mats: np.ndarray, root_positions: np.ndarray,
                     peak_frames: list, knee_idx: int, knee_boost_rad: float,
                     constraint_type: str) -> dict:
    """Build a left-foot or right-foot constraint dict at the given peak frames.

    Modifies the knee joint at each peak by adding `knee_boost_rad` to the X-axis
    component of its axis-angle rotation. The G1 knee is essentially a pure X-axis
    hinge (verified empirically: yz components < 0.01 rad).
    """
    n_joints = local_rot_mats.shape[1]
    keyframe_aas = np.zeros((len(peak_frames), n_joints, 3), dtype=np.float32)

    for i, f in enumerate(peak_frames):
        rotvecs = R.from_matrix(local_rot_mats[f]).as_rotvec()  # (n_joints, 3)
        # boost knee X-axis (flexion)
        rotvecs[knee_idx, 0] += knee_boost_rad
        keyframe_aas[i] = rotvecs

    keyframe_roots = root_positions[peak_frames]  # (n_peaks, 3)
    smooth_root_2d = keyframe_roots[:, [0, 2]]  # kimodo Y-up → ground plane is XZ

    return {
        "type": constraint_type,
        "frame_indices": [int(f) for f in peak_frames],
        "local_joints_rot": keyframe_aas.tolist(),
        "root_positions": keyframe_roots.tolist(),
        "smooth_root_2d": smooth_root_2d.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("npz", help="Input kimodo G1 NPZ file (seed motion)")
    parser.add_argument("--prompt", required=True, help="Text prompt for regeneration")
    parser.add_argument("--duration", type=float, required=True, help="Duration in seconds")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--output-name", required=True,
                        help="Output stem (e.g. turn_in_place_v2_left_00)")
    parser.add_argument("--g1-out-dir",
                        default="motions_k1/g1/turn_in_place_v2",
                        help="Output dir for G1 NPZ/CSV")
    parser.add_argument("--k1-out-dir",
                        default="motions_k1/k1",
                        help="Output dir for K1 CSV")
    parser.add_argument("--knee-boost-rad", type=float, default=0.4,
                        help="Extra knee flexion at swing peak (rad). 0.4 ≈ 23°")
    parser.add_argument("--constraint-weight", type=float, default=4.0,
                        help="CFG constraint weight (default 4.0; vs text 2.0)")
    parser.add_argument("--text-weight", type=float, default=2.0)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--retarget-script",
                        default=str(Path(__file__).resolve().parent / "g1_to_k1.py"))
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Don't delete the constraint/meta tmp folder")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    data = np.load(npz_path, allow_pickle=True)
    local_rot = data["local_rot_mats"]      # (T, 34, 3, 3)
    root_pos = data["root_positions"]       # (T, 3)
    posed = data["posed_joints"]            # (T, 34, 3)
    contacts = data["foot_contacts"]        # (T, 4)

    left_foot_y = posed[:, LEFT_ANKLE_ROLL, 1]
    right_foot_y = posed[:, RIGHT_ANKLE_ROLL, 1]

    left_peaks = find_swing_peaks(contacts, left_foot_y, "left")
    right_peaks = find_swing_peaks(contacts, right_foot_y, "right")
    print(f"Detected swing peaks: left={left_peaks}, right={right_peaks}")
    if not left_peaks and not right_peaks:
        print("WARNING: no swing peaks found; constraints will be empty.")

    constraints = []
    if left_peaks:
        constraints.append(build_constraint(
            local_rot, root_pos, left_peaks, LEFT_KNEE,
            args.knee_boost_rad, "left-foot",
        ))
    if right_peaks:
        constraints.append(build_constraint(
            local_rot, root_pos, right_peaks, RIGHT_KNEE,
            args.knee_boost_rad, "right-foot",
        ))

    meta = {
        "text": args.prompt,
        "duration": args.duration,
        "num_samples": 1,
        "seed": args.seed,
        "diffusion_steps": args.diffusion_steps,
        "cfg": {
            "enabled": True,
            "text_weight": args.text_weight,
            "constraint_weight": args.constraint_weight,
        },
    }

    tmp_dir = Path(tempfile.mkdtemp(prefix="kimodo_tweak_"))
    try:
        (tmp_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        (tmp_dir / "constraints.json").write_text(json.dumps(constraints, indent=2))
        print(f"Wrote keyframes to {tmp_dir}")

        g1_out_dir = Path(args.g1_out_dir)
        g1_out_dir.mkdir(parents=True, exist_ok=True)
        g1_stem = g1_out_dir / args.output_name

        # Run kimodo generate from /tmp to avoid namespace collision
        cmd = [
            sys.executable, "-m", "kimodo.scripts.generate",
            "--model", "kimodo-g1-rp",
            "--input_folder", str(tmp_dir),
            "--output", str(g1_stem.resolve()),
        ]
        print(f"\n$ {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd="/tmp")
        if result.returncode != 0:
            print("ERROR: kimodo generation failed")
            return 1

        g1_csv = g1_stem.with_suffix(".csv")
        if not g1_csv.exists():
            print(f"ERROR: expected {g1_csv} not found")
            return 1

        # Retarget to K1
        k1_out_dir = Path(args.k1_out_dir)
        k1_out_dir.mkdir(parents=True, exist_ok=True)
        k1_csv = k1_out_dir / f"{args.output_name}.csv"
        cmd_rt = [sys.executable, args.retarget_script, str(g1_csv), str(k1_csv)]
        print(f"\n$ {' '.join(cmd_rt)}")
        result = subprocess.run(cmd_rt)
        if result.returncode != 0:
            print("ERROR: retargeting failed")
            return 1

        print(f"\nDone:")
        print(f"  G1 NPZ: {g1_stem.with_suffix('.npz')}")
        print(f"  G1 CSV: {g1_csv}")
        print(f"  K1 CSV: {k1_csv}")
    finally:
        if not args.keep_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
