"""Retarget Kimodo G1 motion to Booster K1 robot.

Usage:
    python g1_to_k1.py input_g1.csv output_k1.csv
    python g1_to_k1.py input_g1.csv  # writes to input_g1_k1.csv
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

# G1 qpos layout: [root_xyz(3), root_quat_wxyz(4), 29 joint angles]
G1_JOINTS = {
    "left_hip_pitch": 0, "left_hip_roll": 1, "left_hip_yaw": 2,
    "left_knee": 3, "left_ankle_pitch": 4, "left_ankle_roll": 5,
    "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8,
    "right_knee": 9, "right_ankle_pitch": 10, "right_ankle_roll": 11,
    "waist_yaw": 12, "waist_roll": 13, "waist_pitch": 14,
    "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_shoulder_yaw": 17,
    "left_elbow": 18, "left_wrist_roll": 19, "left_wrist_pitch": 20, "left_wrist_yaw": 21,
    "right_shoulder_pitch": 22, "right_shoulder_roll": 23, "right_shoulder_yaw": 24,
    "right_elbow": 25, "right_wrist_roll": 26, "right_wrist_pitch": 27, "right_wrist_yaw": 28,
}

# Leg joints: direct mapping (same conventions between G1 and K1)
K1_LEG_MAP = {
    10: "left_hip_pitch", 11: "left_hip_roll", 12: "left_hip_yaw",
    13: "left_knee", 14: "left_ankle_pitch", 15: "left_ankle_roll",
    16: "right_hip_pitch", 17: "right_hip_roll", 18: "right_hip_yaw",
    19: "right_knee", 20: "right_ankle_pitch", 21: "right_ankle_roll",
}

_SHOULDER_ROLL_REST_OFFSET = np.pi / 2


def parse_k1_joint_limits(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    all_joints = root.find("worldbody").findall(".//joint")
    hinge_joints = [j for j in all_joints if j.get("type", "hinge") != "free"
                    and j.get("name") != "world_joint"]
    limits = np.full((len(hinge_joints), 2), [-np.pi, np.pi])
    for i, j in enumerate(hinge_joints):
        r = j.get("range")
        if r:
            lo, hi = [float(x) for x in r.split()]
            limits[i] = [lo, hi]
    return limits


def compute_k1_standing_height(k1_xml):
    model = mujoco.MjModel.from_xml_path(k1_xml)
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    min_foot_z = float("inf")
    for i in range(model.ngeom):
        bodyname = model.body(model.geom(i).bodyid).name
        if "foot" in bodyname.lower():
            pos = data.geom_xpos[i]
            size = model.geom(i).size
            gtype = model.geom(i).type[0] if hasattr(model.geom(i).type, '__len__') else model.geom(i).type
            if gtype == 6:
                min_foot_z = min(min_foot_z, pos[2] - size[2])
    return 1.0 - min_foot_z


def retarget_g1_to_k1(g1_qpos, k1_limits, k1_xml, clamp=True):
    n_frames = g1_qpos.shape[0]
    k1_qpos = np.zeros((n_frames, 29))

    # Root pose: scale XYZ by body proportion ratio
    k1_qpos[:, :7] = g1_qpos[:, :7]
    body_scale = compute_k1_standing_height(k1_xml) / 0.792
    k1_qpos[:, :3] = g1_qpos[:, :3] * body_scale

    # Joint angles
    g1 = g1_qpos[:, 7:]
    k1 = np.zeros((n_frames, 22))

    # Legs: direct mapping
    for k1_idx, g1_name in K1_LEG_MAP.items():
        k1[:, k1_idx] = g1[:, G1_JOINTS[g1_name]]

    # Left arm
    k1[:, 2] = g1[:, G1_JOINTS["left_shoulder_pitch"]]               # shoulder pitch
    k1[:, 3] = g1[:, G1_JOINTS["left_shoulder_roll"]] - _SHOULDER_ROLL_REST_OFFSET  # shoulder roll (+ offset)
    k1[:, 4] = g1[:, G1_JOINTS["left_elbow"]]                        # elbow pitch (bend)
    k1[:, 5] = g1[:, G1_JOINTS["left_shoulder_yaw"]]                 # elbow yaw ← G1 shoulder yaw (twist)

    # Right arm
    k1[:, 6] = g1[:, G1_JOINTS["right_shoulder_pitch"]]
    k1[:, 7] = g1[:, G1_JOINTS["right_shoulder_roll"]] + _SHOULDER_ROLL_REST_OFFSET
    k1[:, 8] = g1[:, G1_JOINTS["right_elbow"]]
    k1[:, 9] = g1[:, G1_JOINTS["right_shoulder_yaw"]]

    # Head: stays 0

    if clamp:
        k1 = np.clip(k1, k1_limits[:, 0], k1_limits[:, 1])

    k1_qpos[:, 7:] = k1
    return k1_qpos


def main():
    parser = argparse.ArgumentParser(description="Retarget G1 motion CSV to K1")
    parser.add_argument("input", help="G1 qpos CSV file")
    parser.add_argument("output", nargs="?", help="K1 qpos CSV output (default: <input>_k1.csv)")
    parser.add_argument("--no-clamp", action="store_true", help="Don't clamp to K1 joint limits")
    parser.add_argument(
        "--k1-xml",
        default=str(Path(__file__).resolve().parent.parent / "robot" / "K1_22dof.xml"),
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = str(Path(args.input).parent / f"{Path(args.input).stem}_k1.csv")

    g1_qpos = np.loadtxt(args.input, delimiter=",")
    assert g1_qpos.shape[1] == 36
    k1_limits = parse_k1_joint_limits(args.k1_xml)
    k1_qpos = retarget_g1_to_k1(g1_qpos, k1_limits, args.k1_xml, clamp=not args.no_clamp)
    np.savetxt(args.output, k1_qpos, delimiter=",")
    print(f"{g1_qpos.shape[0]} frames -> {args.output}")


if __name__ == "__main__":
    main()
