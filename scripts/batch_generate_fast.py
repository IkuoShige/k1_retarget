"""Batch generate steady high-speed forward motions (1.3-1.7 m/s) for AMP training.

Fills the gap in the motion pool: current clips > 1.3 m/s are only 3,
making it hard for AMP to learn clean fast-forward locomotion.

Usage:
    python batch_generate_fast.py --dry-run              # preview prompts
    python batch_generate_fast.py                        # generate all
    python batch_generate_fast.py --filter jog           # only jog prompts
    python batch_generate_fast.py --samples-per-prompt 3 # 3 samples each
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
BASE_DIR = SCRIPTS.parent / "motions_k1"

# Prompts targeting steady 1.3-1.7 m/s forward locomotion.
# Key: long duration + "steady pace" / "constant speed" phrasing
# to avoid acceleration/deceleration transients.
PROMPTS = [
    # === Fast walking (targeting ~1.3 m/s) ===
    ("walk_fast_steady_01", 6.0,
     "a person walking forward at a fast steady pace"),
    ("walk_fast_steady_02", 6.0,
     "a person walking forward quickly at a constant speed"),
    ("walk_fast_steady_03", 6.0,
     "a person walking forward briskly with long strides at a steady pace"),
    ("walk_fast_steady_04", 5.0,
     "a person power walking forward at a fast constant speed"),

    # === Jogging (targeting ~1.3-1.5 m/s) ===
    ("jog_steady_01", 6.0,
     "a person jogging forward at a steady moderate pace"),
    ("jog_steady_02", 6.0,
     "a person jogging forward steadily at a comfortable pace"),
    ("jog_steady_03", 5.0,
     "a person jogging forward at a constant speed"),
    ("jog_steady_04", 5.0,
     "a person jogging forward with a steady rhythmic stride"),

    # === Running (targeting ~1.5-1.7 m/s) ===
    ("run_steady_01", 5.0,
     "a person running forward at a steady pace"),
    ("run_steady_02", 5.0,
     "a person running forward at a constant moderate speed"),
    ("run_steady_03", 4.0,
     "a person running forward steadily"),
    ("run_steady_04", 5.0,
     "a person running forward at a steady fast pace with even strides"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Generate steady high-speed forward motions (1.3-1.7 m/s)")
    parser.add_argument("--samples-per-prompt", type=int, default=2)
    parser.add_argument("--base-seed", type=int, default=5000)
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=str(BASE_DIR))
    args = parser.parse_args()

    prompts = PROMPTS
    if args.filter:
        prompts = [(n, d, p) for n, d, p in prompts if args.filter in n]

    n_total = len(prompts) * args.samples_per_prompt
    print(f"Prompts: {len(prompts)}, Samples/prompt: {args.samples_per_prompt}, "
          f"Total: {n_total}")

    if args.dry_run:
        for name, dur, prompt in prompts:
            print(f"  [{dur:.0f}s] {name:30s} | {prompt}")
        return

    output_dir = Path(args.output_dir)
    g1_dir = output_dir / "g1"
    k1_dir = output_dir / "k1"
    k1_fixed_dir = output_dir / "k1_fixed"
    for d in [g1_dir, k1_dir, k1_fixed_dir]:
        d.mkdir(parents=True, exist_ok=True)

    retarget_script = str(SCRIPTS / "g1_to_k1.py")
    fix_arms_script = str(SCRIPTS / "fix_arms.py")

    success = 0
    for i, (name, duration, prompt) in enumerate(prompts):
        seed = args.base_seed + i * 100
        g1_stem = g1_dir / name

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(prompts)}] {name}")
        print(f"  Prompt: {prompt}")
        print(f"  Duration: {duration}s, Seed: {seed}")

        # --- Generate G1 motion ---
        cmd_gen = [
            sys.executable, "-m", "kimodo.scripts.generate",
            prompt,
            "--model", "kimodo-g1-rp",
            "--duration", str(duration),
            "--num_samples", str(args.samples_per_prompt),
            "--seed", str(seed),
            "--output", str(g1_stem),
        ]
        result = subprocess.run(cmd_gen, capture_output=True, text=True, cwd="/tmp")
        if result.returncode != 0:
            print(f"  ERROR (generate): {result.stderr[-500:]}")
            continue

        # --- Find generated CSVs ---
        # kimodo saves multi-sample output in a subdirectory:
        #   g1/<name>/<name>_00.csv, g1/<name>/<name>_01.csv, ...
        if args.samples_per_prompt == 1:
            csv_files = [g1_stem.with_suffix(".csv")]
        else:
            # Try subdirectory first (kimodo default for multi-sample)
            csv_files = sorted((g1_stem).glob(f"{name}_*.csv"))
            if not csv_files:
                # Fallback: flat layout
                csv_files = sorted(g1_stem.parent.glob(f"{name}_*.csv"))
            if not csv_files:
                csv_files = [g1_stem.with_suffix(".csv")]

        # --- Retarget + fix arms for each sample ---
        for csv_file in csv_files:
            if not csv_file.exists():
                print(f"  WARN: {csv_file} not found")
                continue

            k1_csv = k1_dir / csv_file.name
            k1_fixed_csv = k1_fixed_dir / csv_file.name

            # Retarget G1 -> K1
            cmd_rt = [sys.executable, retarget_script, str(csv_file), str(k1_csv)]
            rt = subprocess.run(cmd_rt, capture_output=True, text=True)
            if rt.returncode != 0:
                print(f"  ERROR (retarget): {rt.stderr[-300:]}")
                continue

            # Fix arms
            cmd_fix = [
                sys.executable, fix_arms_script,
                str(k1_csv), "--output-dir", str(k1_fixed_dir),
            ]
            subprocess.run(cmd_fix, capture_output=True, text=True)

            print(f"  -> {k1_fixed_csv}")
            success += 1

    print(f"\n{'='*60}")
    print(f"Done: {success}/{n_total} clips generated")
    print(f"  G1: {g1_dir}")
    print(f"  K1: {k1_dir}")
    print(f"  K1 fixed: {k1_fixed_dir}")
    print(f"\nNext: run compute_motion_stats.py to verify speed distribution")


if __name__ == "__main__":
    main()
