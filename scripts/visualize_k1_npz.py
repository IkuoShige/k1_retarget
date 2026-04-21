"""Replay a retargeted K1 soccer NPZ inside MuJoCo's passive viewer.

Usage:
    python visualize_k1_npz.py motions/soccer-standard-mj-k1/<file>.npz [--loop]
    python visualize_k1_npz.py motions/soccer-standard-mj-k1/           # replays all

Keys inside the viewer: ESC to quit the current clip (moves to next).
"""

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer as viewer
import numpy as np


def _k1_xml() -> Path:
  repo_root = Path(__file__).resolve().parent.parent.parent
  p = repo_root / "mjlab" / "src" / "mjlab" / "asset_zoo" / "robots" / "booster_k1" / "xmls" / "k1.xml"
  assert p.exists(), f"K1 xml not found: {p}"
  return p


def replay(model: mujoco.MjModel, npz_path: Path, loop: bool) -> bool:
  """Returns False if the viewer window was closed (stop subsequent clips)."""
  data = np.load(npz_path, allow_pickle=True)
  fps = int(data["fps"][0])
  joint_pos = data["joint_pos"]
  body_pos_w = data["body_pos_w"]
  body_quat_w = data["body_quat_w"]
  n_frames = joint_pos.shape[0]
  dt = 1.0 / fps

  md = mujoco.MjData(model)
  md.qpos[0:3] = body_pos_w[0, 0]
  md.qpos[3:7] = body_quat_w[0, 0]
  md.qpos[7:] = joint_pos[0]
  md.qvel[:] = 0
  mujoco.mj_forward(model, md)

  print(f"[{npz_path.name}] {n_frames} frames @ {fps} fps ({n_frames * dt:.2f} s)")

  with viewer.launch_passive(model, md) as v:
    while v.is_running():
      start = time.time()
      for t in range(n_frames):
        if not v.is_running():
          return False
        md.qpos[0:3] = body_pos_w[t, 0]
        md.qpos[3:7] = body_quat_w[t, 0]
        md.qpos[7:] = joint_pos[t]
        md.qvel[:] = 0
        mujoco.mj_forward(model, md)
        v.sync()
        target = start + (t + 1) * dt
        while time.time() < target:
          time.sleep(0.001)
      if not loop:
        break
      start = time.time()
  return True


def main() -> None:
  parser = argparse.ArgumentParser(description="Replay K1 NPZ motion in MuJoCo")
  parser.add_argument("path", help="NPZ file or directory")
  parser.add_argument("--loop", action="store_true")
  args = parser.parse_args()

  model = mujoco.MjModel.from_xml_path(str(_k1_xml()))

  src = Path(args.path)
  if src.is_dir():
    files = sorted(src.glob("*.npz"))
  else:
    files = [src]
  if not files:
    print("No NPZ found.", file=sys.stderr)
    sys.exit(1)

  for f in files:
    cont = replay(model, f, loop=args.loop)
    if not cont:
      break


if __name__ == "__main__":
  main()
