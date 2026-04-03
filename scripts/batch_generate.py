"""Batch generate locomotion motions with kimodo G1 and retarget to K1.

Usage:
    python batch_generate.py --output-dir motions_k1
    python batch_generate.py --output-dir motions_k1 --samples-per-prompt 3
    python batch_generate.py --output-dir motions_k1 --dry-run  # preview prompts only
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# Diverse locomotion prompts for AMP training data
PROMPTS = [
    # === Forward walking ===
    ("walk_forward_normal", 5.0, "a person walking forward naturally"),
    ("walk_forward_slow", 6.0, "a person walking forward slowly"),
    ("walk_forward_fast", 4.0, "a person walking forward quickly"),
    ("walk_forward_brisk", 4.0, "a person walking forward briskly with purpose"),
    ("walk_forward_casual", 6.0, "a person casually strolling forward"),

    # === Backward walking ===
    ("walk_backward", 5.0, "a person walking backward"),
    ("walk_backward_slow", 6.0, "a person walking backward slowly and carefully"),

    # === Lateral walking ===
    ("walk_left", 5.0, "a person side-stepping to the left"),
    ("walk_right", 5.0, "a person side-stepping to the right"),
    ("shuffle_left", 4.0, "a person shuffling sideways to the left"),
    ("shuffle_right", 4.0, "a person shuffling sideways to the right"),

    # === Turning ===
    ("turn_left_90", 4.0, "a person walking forward then turning left 90 degrees"),
    ("turn_right_90", 4.0, "a person walking forward then turning right 90 degrees"),
    ("turn_left_180", 5.0, "a person walking forward then making a U-turn to the left"),
    ("turn_right_180", 5.0, "a person walking forward then making a U-turn to the right"),
    ("turn_in_place_left", 4.0, "a person turning around in place to the left"),
    ("turn_in_place_right", 4.0, "a person turning around in place to the right"),

    # === Curved walking ===
    ("walk_circle_left", 6.0, "a person walking in a circle to the left"),
    ("walk_circle_right", 6.0, "a person walking in a circle to the right"),
    ("walk_curve_left", 5.0, "a person walking forward while gradually curving to the left"),
    ("walk_curve_right", 5.0, "a person walking forward while gradually curving to the right"),

    # === Speed transitions ===
    ("walk_accel", 5.0, "a person starting to walk slowly then speeding up"),
    ("walk_decel", 5.0, "a person walking fast then gradually slowing down to a stop"),
    ("walk_start_stop", 6.0, "a person standing still, then walking forward, then stopping"),

    # === Diagonal walking ===
    ("walk_diag_left", 5.0, "a person walking diagonally to the front-left"),
    ("walk_diag_right", 5.0, "a person walking diagonally to the front-right"),

    # === Jogging / running ===
    ("jog_forward", 4.0, "a person jogging forward"),
    ("jog_slow", 5.0, "a person jogging forward at a slow pace"),
    ("run_forward", 3.0, "a person running forward"),

    # === Standing / idle ===
    ("stand_idle", 4.0, "a person standing still naturally"),
    ("stand_shift_weight", 5.0, "a person standing and shifting weight from one foot to the other"),

    # === Complex locomotion ===
    ("walk_zigzag", 6.0, "a person walking forward in a zigzag pattern"),
    ("walk_stumble", 5.0, "a person walking forward and stumbling but recovering balance"),
    ("walk_uneven", 5.0, "a person walking carefully as if on uneven ground"),
    ("walk_confident", 5.0, "a person walking forward confidently with long strides"),
    ("walk_tired", 6.0, "a person walking forward slowly as if tired and exhausted"),
    ("walk_sneak", 5.0, "a person walking forward cautiously and sneaking"),
]


def generate_one(name, duration, prompt, output_dir, seed, samples_per_prompt):
    """Generate G1 motion and retarget to K1."""
    g1_stem = output_dir / "g1" / name
    k1_dir = output_dir / "k1"

    g1_stem.parent.mkdir(parents=True, exist_ok=True)
    k1_dir.mkdir(parents=True, exist_ok=True)

    # Generate G1 motion (run from /tmp to avoid kimodo namespace issue)
    cmd = [
        sys.executable, "-m", "kimodo.scripts.generate",
        prompt,
        "--model", "kimodo-g1-rp",
        "--duration", str(duration),
        "--num_samples", str(samples_per_prompt),
        "--output", str(g1_stem),
        "--seed", str(seed),
    ]

    print(f"\n{'='*60}")
    print(f"Generating: {name}")
    print(f"  Prompt: {prompt}")
    print(f"  Duration: {duration}s, Samples: {samples_per_prompt}, Seed: {seed}")

    result = subprocess.run(cmd, cwd="/tmp", capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[-500:]}")
        return False

    # Find generated CSV files
    if samples_per_prompt == 1:
        csv_files = [g1_stem.with_suffix(".csv")]
    else:
        csv_files = sorted(g1_stem.parent.glob(f"{name}_*.csv"))

    # Retarget each to K1
    retarget_script = Path(__file__).parent / "g1_to_k1.py"
    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"  WARNING: {csv_file} not found, skipping")
            continue

        k1_output = k1_dir / csv_file.name
        cmd_retarget = [
            sys.executable, str(retarget_script),
            str(csv_file), str(k1_output),
        ]
        result = subprocess.run(cmd_retarget, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Retarget ERROR: {result.stderr[-300:]}")
        else:
            print(f"  -> {k1_output}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Batch generate locomotion motions")
    parser.add_argument("--output-dir", default="motions_k1", help="Output directory")
    parser.add_argument("--samples-per-prompt", type=int, default=2,
                        help="Number of samples per prompt (default: 2)")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--dry-run", action="store_true", help="Just print prompts, don't generate")
    parser.add_argument("--filter", type=str, default=None,
                        help="Only generate prompts containing this string (e.g. 'turn')")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    prompts = PROMPTS
    if args.filter:
        prompts = [(n, d, p) for n, d, p in prompts if args.filter in n]

    print(f"Total prompts: {len(prompts)}")
    print(f"Samples per prompt: {args.samples_per_prompt}")
    print(f"Total motions: {len(prompts) * args.samples_per_prompt}")

    if args.dry_run:
        print("\nPrompts:")
        for name, dur, prompt in prompts:
            print(f"  [{dur:.0f}s] {name:30s} | {prompt}")
        return

    success = 0
    for i, (name, duration, prompt) in enumerate(prompts):
        seed = args.base_seed + i * 100
        ok = generate_one(name, duration, prompt, output_dir, seed, args.samples_per_prompt)
        if ok:
            success += 1

    print(f"\n{'='*60}")
    print(f"Done: {success}/{len(prompts)} prompts generated")
    print(f"G1 motions: {output_dir}/g1/")
    print(f"K1 motions: {output_dir}/k1/")


if __name__ == "__main__":
    main()
