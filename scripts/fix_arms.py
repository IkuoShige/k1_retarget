"""Fix arm angles in K1 motion CSVs to a neutral pose.

For AMP locomotion training, noisy/unnatural arm motion in reference data
is harmful. This script replaces arm joint angles with a fixed neutral pose,
keeping only root pose and leg motion from the original data.

Usage:
    python fix_arms.py motions_k1/k1/                    # fix all in-place
    python fix_arms.py motions_k1/k1/ --output-dir motions_k1/k1_fixed/
    python fix_arms.py motions_k1/k1/lateral*.csv         # fix specific files
    python fix_arms.py motions_k1/k1/ --preview lateral_left_shuffle_05_00.csv
"""

import argparse
from pathlib import Path

import numpy as np

# K1 joint order (22 DoF), indices within the 22 joint angles (after 7 root values)
# 0:  AAHead_yaw
# 1:  Head_pitch
# 2:  ALeft_Shoulder_Pitch
# 3:  Left_Shoulder_Roll
# 4:  Left_Elbow_Pitch
# 5:  Left_Elbow_Yaw
# 6:  ARight_Shoulder_Pitch
# 7:  Right_Shoulder_Roll
# 8:  Right_Elbow_Pitch
# 9:  Right_Elbow_Yaw
# 10-21: Leg joints (keep as-is)

ARM_INDICES = [2, 3, 4, 5, 6, 7, 8, 9]  # within 22-DoF
HEAD_INDICES = [0, 1]

# Neutral arm pose: arms relaxed at sides
# These values place the arms in a natural resting position on K1
NEUTRAL_ARM_POSE = {
    2: 0.0,     # L shoulder pitch: neutral
    3: -1.1,    # L shoulder roll: arm hanging down at side
    4: 0.2,     # L elbow pitch: slightly bent
    5: -0.5,    # L elbow yaw: slight inward
    6: 0.0,     # R shoulder pitch: neutral
    7: 1.1,     # R shoulder roll: arm hanging down at side
    8: 0.2,     # R elbow pitch: slightly bent
    9: 0.5,     # R elbow yaw: slight inward
}


def fix_arms(qpos_data, arm_pose=None):
    """Replace arm joint angles with fixed neutral pose."""
    if arm_pose is None:
        arm_pose = NEUTRAL_ARM_POSE

    result = qpos_data.copy()
    for joint_idx, angle in arm_pose.items():
        result[:, 7 + joint_idx] = angle

    # Also fix head to neutral
    for idx in HEAD_INDICES:
        result[:, 7 + idx] = 0.0

    return result


def main():
    parser = argparse.ArgumentParser(description="Fix arm angles in K1 motion CSVs")
    parser.add_argument("inputs", nargs="+", help="CSV files or directory")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: overwrite in-place)")
    parser.add_argument("--preview", default=None,
                        help="Print before/after for a specific file, don't write")
    args = parser.parse_args()

    # Collect CSV files
    csv_files = []
    for inp in args.inputs:
        p = Path(inp)
        if p.is_dir():
            csv_files.extend(sorted(p.glob("*.csv")))
        elif p.exists() and p.suffix == ".csv":
            csv_files.append(p)
        else:
            # glob pattern
            csv_files.extend(sorted(Path(".").glob(inp)))

    # Filter out non-motion files (like motion_stats.json)
    csv_files = [f for f in csv_files if f.suffix == ".csv"]
    print(f"Found {len(csv_files)} CSV files")

    if args.preview:
        target = [f for f in csv_files if args.preview in f.name]
        if not target:
            print(f"No file matching '{args.preview}'")
            return
        f = target[0]
        qpos = np.loadtxt(f, delimiter=",")
        fixed = fix_arms(qpos)
        print(f"\n{f.name} - Frame 0 arm angles:")
        names = ["L_Sh_P", "L_Sh_R", "L_El_P", "L_El_Y",
                 "R_Sh_P", "R_Sh_R", "R_El_P", "R_El_Y"]
        print(f"  {'Joint':8s} {'Before':>8s} {'After':>8s}")
        for name, idx in zip(names, ARM_INDICES):
            before = qpos[0, 7 + idx]
            after = fixed[0, 7 + idx]
            print(f"  {name:8s} {before:+8.3f} {after:+8.3f}")
        return

    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    count = 0
    for f in csv_files:
        qpos = np.loadtxt(f, delimiter=",")
        if qpos.shape[1] != 29:
            print(f"  Skip {f.name}: {qpos.shape[1]} cols (expected 29)")
            continue

        fixed = fix_arms(qpos)

        if out_dir:
            out_path = out_dir / f.name
        else:
            out_path = f  # overwrite

        np.savetxt(out_path, fixed, delimiter=",")
        count += 1

    print(f"Fixed arms in {count} files -> {out_dir or 'in-place'}")


if __name__ == "__main__":
    main()
