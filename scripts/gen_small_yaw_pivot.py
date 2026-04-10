"""Generate small-yaw pivot motions with foot-lift boost.

Two-pass pipeline:
  1. Generate baseline G1 motions (text prompt only, no constraints)
  2. Apply foot-lift boost (tweak_foot_lift.py) using each baseline NPZ as seed
  3. Retarget to K1 + fix arms

Usage:
    python gen_small_yaw_pivot.py                     # generate all
    python gen_small_yaw_pivot.py --filter gentle      # only "gentle" prompts
    python gen_small_yaw_pivot.py --dry-run             # preview prompts
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS.parent / "motions_k1"

PROMPTS = [
    # (name, duration_s, prompt)
    ("pivot_slow_left",     5.0, "a person marching in place while slowly rotating to the left"),
    ("pivot_slow_right",    5.0, "a person marching in place while slowly rotating to the right"),
    ("pivot_gentle_left",   6.0, "a person stepping in place with a very slight turn to the left"),
    ("pivot_gentle_right",  6.0, "a person stepping in place with a very slight turn to the right"),
    ("pivot_quarter_left",  5.0, "a person turning 45 degrees in place to the left"),
    ("pivot_quarter_right", 5.0, "a person turning 45 degrees in place to the right"),
]


def run_cmd(cmd, label=""):
    """Run a command, print label, return success bool."""
    if label:
        print(f"  [{label}]")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp")
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate small-yaw pivot motions")
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=3000)
    parser.add_argument("--knee-boost-rad", type=float, default=0.4,
                        help="Foot lift boost (rad, default 0.4)")
    parser.add_argument("--constraint-weight", type=float, default=4.0)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    prompts = PROMPTS
    if args.filter:
        prompts = [(n, d, p) for n, d, p in prompts if args.filter in n]

    n_total = len(prompts) * args.samples_per_prompt
    print(f"Prompts: {len(prompts)}, Samples/prompt: {args.samples_per_prompt}, Total: {n_total}")

    if args.dry_run:
        for name, dur, prompt in prompts:
            print(f"  [{dur:.0f}s] {name:30s} | {prompt}")
        return

    g1_base_dir = BASE_DIR / "g1" / "pivot_small_yaw"
    g1_v2_dir = BASE_DIR / "g1" / "pivot_small_yaw_v2"
    k1_dir = BASE_DIR / "k1"
    k1_fixed_dir = BASE_DIR / "k1_fixed"
    for d in [g1_base_dir, g1_v2_dir, k1_dir, k1_fixed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    tweak_script = str(SCRIPTS / "tweak_foot_lift.py")
    fix_arms_script = str(SCRIPTS / "fix_arms.py")

    success = 0
    for i, (name, duration, prompt) in enumerate(prompts):
        seed = args.base_seed + i * 100
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(prompts)}] {name}: {prompt}")

        # === Pass 1: generate baseline (no constraints) ===
        baseline_stem = g1_base_dir / name
        cmd_gen = [
            sys.executable, "-m", "kimodo.scripts.generate",
            prompt,
            "--model", "kimodo-g1-rp",
            "--duration", str(duration),
            "--num_samples", str(args.samples_per_prompt),
            "--seed", str(seed),
            "--output", str(baseline_stem),
        ]
        if not run_cmd(cmd_gen, "pass1: baseline generation"):
            continue

        # Find generated NPZ files
        if args.samples_per_prompt == 1:
            npz_files = [baseline_stem.with_suffix(".npz")]
        else:
            npz_files = sorted(baseline_stem.parent.glob(f"{name}/{name}_*.npz"))
            if not npz_files:
                npz_files = sorted(baseline_stem.parent.glob(f"{name}_*.npz"))
                if not npz_files:
                    npz_files = [baseline_stem.with_suffix(".npz")]

        print(f"  baseline NPZ: {[f.name for f in npz_files if f.exists()]}")

        # === Pass 2: foot-lift boost for each sample ===
        for j, npz_file in enumerate(npz_files):
            if not npz_file.exists():
                print(f"  WARN: {npz_file} not found, skipping")
                continue

            out_name = f"{name}_{j:02d}"
            sample_seed = seed + j * 7  # slightly vary seed per sample

            cmd_tweak = [
                sys.executable, tweak_script,
                str(npz_file),
                "--prompt", prompt,
                "--duration", str(duration),
                "--seed", str(sample_seed),
                "--output-name", out_name,
                "--g1-out-dir", str(g1_v2_dir),
                "--k1-out-dir", str(k1_dir),
                "--knee-boost-rad", str(args.knee_boost_rad),
                "--constraint-weight", str(args.constraint_weight),
            ]
            if not run_cmd(cmd_tweak, f"pass2: boost {out_name}"):
                continue

            # === Fix arms ===
            k1_csv = k1_dir / f"{out_name}.csv"
            if k1_csv.exists():
                cmd_fix = [
                    sys.executable, fix_arms_script,
                    str(k1_csv),
                    "--output-dir", str(k1_fixed_dir),
                ]
                run_cmd(cmd_fix, f"fix arms {out_name}")

            success += 1

    print(f"\n{'='*60}")
    print(f"Done: {success}/{n_total} clips generated")
    print(f"  G1 baseline:  {g1_base_dir}")
    print(f"  G1 v2 (boost): {g1_v2_dir}")
    print(f"  K1:            {k1_dir}")
    print(f"  K1 fixed:      {k1_fixed_dir}")


if __name__ == "__main__":
    main()
