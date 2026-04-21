"""Retarget mjlab soccer NPZ motions from G1 to Booster K1.

Input NPZ keys (G1, 29 DoF):
  fps: (1,) int
  joint_pos: (T, 29) float32, spec joint order (excluding freejoint)
  joint_vel: (T, 29)
  body_pos_w: (T, 30, 3), spec body order (excluding world)
  body_quat_w: (T, 30, 4), wxyz
  body_lin_vel_w: (T, 30, 3)
  body_ang_vel_w: (T, 30, 3)
  kick_leg: str ('left' or 'right')

Output NPZ keys (K1, 22 DoF):
  fps: (1,) int
  joint_pos: (T, 22)
  joint_vel: (T, 22)
  body_pos_w: (T, 23, 3)
  body_quat_w: (T, 23, 4)
  body_lin_vel_w: (T, 23, 3)
  body_ang_vel_w: (T, 23, 3)
  kick_leg: str

Joint retarget follows g1_to_k1.py (22 DoF mapping, shoulder roll offset, body
scale). Body-space fields are recomputed from K1 forward kinematics; velocities
use central finite differences on positions / quaternions.

Usage:
  python soccer_npz_g1_to_k1.py motions/soccer-standard-mj/
  python soccer_npz_g1_to_k1.py motions/soccer-standard-mj/*.npz \\
      --output-dir motions/soccer-standard-mj-k1/
"""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent


def _compute_k1_standing_height(xml_path: str) -> float:
  """Distance from Trunk origin to the lowest foot point when qpos = rest pose
  with Trunk z = 1.0. Matches the g1_to_k1.py convention (~0.558)."""
  model = mujoco.MjModel.from_xml_path(xml_path)
  data = mujoco.MjData(model)
  data.qpos[:] = 0
  data.qpos[2] = 1.0
  # Unit quat.
  data.qpos[3] = 1.0
  mujoco.mj_forward(model, data)
  min_foot_z = float("inf")
  for i in range(model.ngeom):
    body_id = int(model.geom_bodyid[i])
    bodyname = model.body(body_id).name
    if "foot" not in bodyname.lower():
      continue
    gtype = int(model.geom_type[i])
    if gtype not in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_SPHERE):
      continue
    half_z = float(model.geom_size[i, 2]) if gtype == mujoco.mjtGeom.mjGEOM_BOX else float(
      model.geom_size[i, 0]
    )
    z_bottom = float(data.geom_xpos[i, 2]) - half_z
    if z_bottom < min_foot_z:
      min_foot_z = z_bottom
  return 1.0 - min_foot_z

# G1 joint index within NPZ joint_pos columns.
G1_JOINT = {
  "left_hip_pitch": 0, "left_hip_roll": 1, "left_hip_yaw": 2,
  "left_knee": 3, "left_ankle_pitch": 4, "left_ankle_roll": 5,
  "right_hip_pitch": 6, "right_hip_roll": 7, "right_hip_yaw": 8,
  "right_knee": 9, "right_ankle_pitch": 10, "right_ankle_roll": 11,
  "waist_yaw": 12, "waist_roll": 13, "waist_pitch": 14,
  "left_shoulder_pitch": 15, "left_shoulder_roll": 16, "left_shoulder_yaw": 17,
  "left_elbow": 18,
  "left_wrist_roll": 19, "left_wrist_pitch": 20, "left_wrist_yaw": 21,
  "right_shoulder_pitch": 22, "right_shoulder_roll": 23, "right_shoulder_yaw": 24,
  "right_elbow": 25,
  "right_wrist_roll": 26, "right_wrist_pitch": 27, "right_wrist_yaw": 28,
}  # fmt: skip

# K1 joint index within output joint_pos columns (mjlab booster_k1 spec order).
K1_JOINT = {
  "AAHead_yaw": 0, "Head_pitch": 1,
  "ALeft_Shoulder_Pitch": 2, "Left_Shoulder_Roll": 3,
  "Left_Elbow_Pitch": 4, "Left_Elbow_Yaw": 5,
  "ARight_Shoulder_Pitch": 6, "Right_Shoulder_Roll": 7,
  "Right_Elbow_Pitch": 8, "Right_Elbow_Yaw": 9,
  "Left_Hip_Pitch": 10, "Left_Hip_Roll": 11, "Left_Hip_Yaw": 12,
  "Left_Knee_Pitch": 13, "Left_Ankle_Pitch": 14, "Left_Ankle_Roll": 15,
  "Right_Hip_Pitch": 16, "Right_Hip_Roll": 17, "Right_Hip_Yaw": 18,
  "Right_Knee_Pitch": 19, "Right_Ankle_Pitch": 20, "Right_Ankle_Roll": 21,
}  # fmt: skip

# G1 pelvis -> K1 Trunk height ratio used by g1_to_k1.py.
G1_STANDING_HEIGHT = 0.792
_SHOULDER_ROLL_REST_OFFSET = np.pi / 2


def retarget_qpos(
  g1_joint_pos: np.ndarray,
  g1_root_pos: np.ndarray,
  g1_root_quat: np.ndarray,
  body_scale: float,
  k1_soft_limits: np.ndarray,
) -> np.ndarray:
  """Build K1 qpos (T, 29) from G1 joint/root data.

  g1_joint_pos: (T, 29)
  g1_root_pos:  (T, 3)
  g1_root_quat: (T, 4) wxyz
  k1_soft_limits: (22, 2) joint pos limits for K1 hinge joints.
  """
  n = g1_joint_pos.shape[0]
  qpos = np.zeros((n, 29), dtype=np.float32)
  qpos[:, 0:3] = g1_root_pos * body_scale
  qpos[:, 3:7] = g1_root_quat

  k1_joints = np.zeros((n, 22), dtype=np.float32)

  # Legs: direct G1 -> K1 (same conventions).
  leg_pairs = [
    ("Left_Hip_Pitch", "left_hip_pitch"),
    ("Left_Hip_Roll", "left_hip_roll"),
    ("Left_Hip_Yaw", "left_hip_yaw"),
    ("Left_Knee_Pitch", "left_knee"),
    ("Left_Ankle_Pitch", "left_ankle_pitch"),
    ("Left_Ankle_Roll", "left_ankle_roll"),
    ("Right_Hip_Pitch", "right_hip_pitch"),
    ("Right_Hip_Roll", "right_hip_roll"),
    ("Right_Hip_Yaw", "right_hip_yaw"),
    ("Right_Knee_Pitch", "right_knee"),
    ("Right_Ankle_Pitch", "right_ankle_pitch"),
    ("Right_Ankle_Roll", "right_ankle_roll"),
  ]
  for k1_name, g1_name in leg_pairs:
    k1_joints[:, K1_JOINT[k1_name]] = g1_joint_pos[:, G1_JOINT[g1_name]]

  # Left arm.
  k1_joints[:, K1_JOINT["ALeft_Shoulder_Pitch"]] = g1_joint_pos[
    :, G1_JOINT["left_shoulder_pitch"]
  ]
  k1_joints[:, K1_JOINT["Left_Shoulder_Roll"]] = (
    g1_joint_pos[:, G1_JOINT["left_shoulder_roll"]] - _SHOULDER_ROLL_REST_OFFSET
  )
  k1_joints[:, K1_JOINT["Left_Elbow_Pitch"]] = g1_joint_pos[:, G1_JOINT["left_elbow"]]
  k1_joints[:, K1_JOINT["Left_Elbow_Yaw"]] = g1_joint_pos[
    :, G1_JOINT["left_shoulder_yaw"]
  ]

  # Right arm.
  k1_joints[:, K1_JOINT["ARight_Shoulder_Pitch"]] = g1_joint_pos[
    :, G1_JOINT["right_shoulder_pitch"]
  ]
  k1_joints[:, K1_JOINT["Right_Shoulder_Roll"]] = (
    g1_joint_pos[:, G1_JOINT["right_shoulder_roll"]] + _SHOULDER_ROLL_REST_OFFSET
  )
  k1_joints[:, K1_JOINT["Right_Elbow_Pitch"]] = g1_joint_pos[
    :, G1_JOINT["right_elbow"]
  ]
  k1_joints[:, K1_JOINT["Right_Elbow_Yaw"]] = g1_joint_pos[
    :, G1_JOINT["right_shoulder_yaw"]
  ]

  # Head stays at 0.
  # Hands (Elbow_Yaw already set above; K1 has no wrist DOFs).

  k1_joints = np.clip(k1_joints, k1_soft_limits[:, 0], k1_soft_limits[:, 1])
  qpos[:, 7:] = k1_joints
  return qpos


def compute_body_kinematics(
  model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
  """Run mj_kinematics per frame. Returns (body_pos_w, body_quat_w) excluding
  world body."""
  n_frames = qpos.shape[0]
  n_bodies = model.nbody - 1  # exclude world
  body_pos = np.zeros((n_frames, n_bodies, 3), dtype=np.float32)
  body_quat = np.zeros((n_frames, n_bodies, 4), dtype=np.float32)
  for t in range(n_frames):
    data.qpos[:] = qpos[t]
    data.qvel[:] = 0
    mujoco.mj_kinematics(model, data)
    body_pos[t] = data.xpos[1:]
    body_quat[t] = data.xquat[1:]
  return body_pos, body_quat


def finite_diff_linear(pos: np.ndarray, dt: float) -> np.ndarray:
  """Central finite difference with edge padding. pos: (T, ..., 3)."""
  vel = np.zeros_like(pos)
  if pos.shape[0] < 2:
    return vel
  vel[1:-1] = (pos[2:] - pos[:-2]) / (2.0 * dt)
  vel[0] = (pos[1] - pos[0]) / dt
  vel[-1] = (pos[-1] - pos[-2]) / dt
  return vel


def quat_ang_vel_world(quat: np.ndarray, dt: float) -> np.ndarray:
  """World-frame angular velocity from a time series of unit quaternions (wxyz).

  For consecutive frames q0, q1 the relative rotation applied in the world
  frame is q_rel = q1 * q0^{-1}. Converting q_rel to axis-angle yields an
  angular displacement vector, which divided by dt approximates world angular
  velocity.
  """
  T = quat.shape[0]
  out = np.zeros((T, *quat.shape[1:-1], 3), dtype=np.float32)
  if T < 2:
    return out

  q = quat.astype(np.float64)
  # Ensure continuity: flip q[t] if it is on the opposite hemisphere from q[t-1].
  for t in range(1, T):
    dot = np.sum(q[t] * q[t - 1], axis=-1, keepdims=True)
    q[t] = np.where(dot < 0, -q[t], q[t])

  def _world_ang(q_prev: np.ndarray, q_next: np.ndarray, step: float) -> np.ndarray:
    # q_rel = q_next * q_prev^{-1}, with q = (w, x, y, z).
    w0, x0, y0, z0 = q_prev[..., 0], q_prev[..., 1], q_prev[..., 2], q_prev[..., 3]
    w1, x1, y1, z1 = q_next[..., 0], q_next[..., 1], q_next[..., 2], q_next[..., 3]
    # Inverse of unit quaternion: (w, -x, -y, -z).
    rw = w1 * w0 + x1 * x0 + y1 * y0 + z1 * z0
    rx = -w1 * x0 + x1 * w0 - y1 * z0 + z1 * y0
    ry = -w1 * y0 + x1 * z0 + y1 * w0 - z1 * x0
    rz = -w1 * z0 - x1 * y0 + y1 * x0 + z1 * w0
    rw = np.clip(rw, -1.0, 1.0)
    sin_half = np.sqrt(np.maximum(0.0, 1.0 - rw * rw))
    angle = 2.0 * np.arctan2(sin_half, rw)
    axis = np.stack([rx, ry, rz], axis=-1)
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis_safe = np.where(axis_norm > 1e-8, axis / np.maximum(axis_norm, 1e-8), 0.0)
    return (axis_safe * angle[..., None]) / step

  out[1:-1] = _world_ang(q[:-2], q[2:], 2.0 * dt).astype(np.float32)
  out[0] = _world_ang(q[0:1], q[1:2], dt).astype(np.float32)[0]
  out[-1] = _world_ang(q[-2:-1], q[-1:], dt).astype(np.float32)[0]
  return out


def retarget_file(
  src_path: Path,
  dst_path: Path,
  model: mujoco.MjModel,
  body_scale: float,
) -> None:
  data = mujoco.MjData(model)
  src = np.load(src_path, allow_pickle=True)

  fps = int(src["fps"][0])
  dt = 1.0 / fps
  joint_pos_g1 = src["joint_pos"]
  assert joint_pos_g1.shape[1] == 29, f"expected 29 G1 joints, got {joint_pos_g1.shape}"

  # G1 body 0 is pelvis (the root).
  root_pos = src["body_pos_w"][:, 0]
  root_quat = src["body_quat_w"][:, 0]

  # soft_joint_pos_limits are available via model.jnt_range for hinge joints.
  # K1: joint indices 1..22 in mujoco (index 0 is freejoint).
  k1_limits = model.jnt_range[1:].astype(np.float32)  # (22, 2)

  qpos = retarget_qpos(
    joint_pos_g1,
    root_pos.astype(np.float32),
    root_quat.astype(np.float32),
    body_scale,
    k1_limits,
  )

  body_pos_w, body_quat_w = compute_body_kinematics(model, data, qpos)
  body_lin_vel_w = finite_diff_linear(body_pos_w, dt)
  body_ang_vel_w = quat_ang_vel_world(body_quat_w, dt)

  joint_pos_k1 = qpos[:, 7:]
  joint_vel_k1 = finite_diff_linear(joint_pos_k1, dt)

  dst_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    dst_path,
    fps=src["fps"],
    joint_pos=joint_pos_k1.astype(np.float32),
    joint_vel=joint_vel_k1.astype(np.float32),
    body_pos_w=body_pos_w.astype(np.float32),
    body_quat_w=body_quat_w.astype(np.float32),
    body_lin_vel_w=body_lin_vel_w.astype(np.float32),
    body_ang_vel_w=body_ang_vel_w.astype(np.float32),
    kick_leg=src["kick_leg"],
  )
  print(f"  {src_path.name} -> {dst_path} ({qpos.shape[0]} frames)")


def _expand_inputs(paths: list[str]) -> list[Path]:
  out: list[Path] = []
  for p in paths:
    path = Path(p)
    if path.is_dir():
      out.extend(sorted(path.glob("*.npz")))
    else:
      out.append(path)
  return out


def _load_k1_model() -> tuple[mujoco.MjModel, float]:
  """Load the mjlab booster_k1 model and compute the body scale ratio."""
  repo_root = _SCRIPT_DIR.parent.parent
  k1_xml = (
    repo_root
    / "mjlab"
    / "src"
    / "mjlab"
    / "asset_zoo"
    / "robots"
    / "booster_k1"
    / "xmls"
    / "k1.xml"
  )
  assert k1_xml.exists(), f"K1 xml not found: {k1_xml}"
  model = mujoco.MjModel.from_xml_path(str(k1_xml))
  k1_standing = _compute_k1_standing_height(str(k1_xml))
  body_scale = k1_standing / G1_STANDING_HEIGHT
  return model, body_scale


def main() -> None:
  parser = argparse.ArgumentParser(description="Retarget G1 soccer NPZ to K1")
  parser.add_argument("inputs", nargs="+", help="NPZ file(s) or directory")
  parser.add_argument(
    "--output-dir",
    default=None,
    help="Output directory (default: append '-k1' to input dir name)",
  )
  args = parser.parse_args()

  files = _expand_inputs(args.inputs)
  if not files:
    print("No input files found.", file=sys.stderr)
    sys.exit(1)

  model, body_scale = _load_k1_model()
  print(f"body_scale (K1/G1) = {body_scale:.4f}")

  if args.output_dir is not None:
    out_dir = Path(args.output_dir)
  else:
    parents = {f.parent for f in files}
    if len(parents) == 1:
      src_dir = next(iter(parents))
      out_dir = src_dir.parent / f"{src_dir.name}-k1"
    else:
      out_dir = Path.cwd() / "retargeted_k1"

  print(f"Output dir: {out_dir}")
  for src in files:
    dst = out_dir / src.name
    retarget_file(src, dst, model, body_scale)

  print(f"Done. {len(files)} file(s) -> {out_dir}")


if __name__ == "__main__":
  main()
