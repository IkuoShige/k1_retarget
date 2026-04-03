"""Generate lateral movement motions with emphasis on no foot crossing.

Usage:
    python batch_lateral.py --output-dir motions_k1
    python batch_lateral.py --output-dir motions_k1 --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

PROMPTS = [
    # Side-step / shuffle (no crossing)
    ("lateral_left_shuffle_01", 5.0, "a person shuffling sideways to the left without crossing their feet"),
    ("lateral_left_shuffle_02", 5.0, "a person taking small side steps to the left keeping feet apart"),
    ("lateral_left_shuffle_03", 6.0, "a person moving sideways to the left with a defensive basketball stance"),
    ("lateral_left_shuffle_04", 5.0, "a person doing a lateral shuffle drill to the left"),
    ("lateral_left_shuffle_05", 4.0, "a person quickly sliding sideways to the left like a crab"),

    ("lateral_right_shuffle_01", 5.0, "a person shuffling sideways to the right without crossing their feet"),
    ("lateral_right_shuffle_02", 5.0, "a person taking small side steps to the right keeping feet apart"),
    ("lateral_right_shuffle_03", 6.0, "a person moving sideways to the right with a defensive basketball stance"),
    ("lateral_right_shuffle_04", 5.0, "a person doing a lateral shuffle drill to the right"),
    ("lateral_right_shuffle_05", 4.0, "a person quickly sliding sideways to the right like a crab"),

    # Slow/careful lateral
    ("lateral_left_slow", 6.0, "a person carefully stepping sideways to the left one step at a time"),
    ("lateral_right_slow", 6.0, "a person carefully stepping sideways to the right one step at a time"),

    # Fast lateral
    ("lateral_left_fast", 4.0, "a person rapidly shuffling sideways to the left in an athletic stance"),
    ("lateral_right_fast", 4.0, "a person rapidly shuffling sideways to the right in an athletic stance"),

    # Lateral with direction changes
    ("lateral_left_right", 6.0, "a person shuffling to the left then quickly changing direction to the right"),
    ("lateral_right_left", 6.0, "a person shuffling to the right then quickly changing direction to the left"),
]


def generate_one(name, duration, prompt, output_dir, seed, samples):
    g1_dir = output_dir / "g1"
    k1_dir = output_dir / "k1"
    g1_dir.mkdir(parents=True, exist_ok=True)
    k1_dir.mkdir(parents=True, exist_ok=True)

    g1_stem = g1_dir / name

    cmd = [
        sys.executable, "-m", "kimodo.scripts.generate",
        prompt,
        "--model", "kimodo-g1-rp",
        "--duration", str(duration),
        "--num_samples", str(samples),
        "--output", str(g1_stem),
        "--seed", str(seed),
    ]

    print(f"  Generating: {name} ...")
    result = subprocess.run(cmd, cwd="/tmp", capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-300:]}")
        return False

    # Find and retarget CSVs
    if samples == 1:
        csv_files = [g1_stem.with_suffix(".csv")]
    else:
        csv_files = sorted(g1_dir.glob(f"{name}/*.csv")) or sorted(g1_dir.glob(f"{name}_*.csv"))

    retarget_script = Path(__file__).parent / "g1_to_k1.py"
    for csv_file in csv_files:
        if not csv_file.exists():
            continue
        k1_output = k1_dir / csv_file.name
        subprocess.run([sys.executable, str(retarget_script), str(csv_file), str(k1_output)],
                       capture_output=True, text=True)
        print(f"    -> {k1_output.name}")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="motions_k1")
    parser.add_argument("--samples-per-prompt", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    print(f"Lateral motion prompts: {len(PROMPTS)}")
    print(f"Samples per prompt: {args.samples_per_prompt}")
    print(f"Total: {len(PROMPTS) * args.samples_per_prompt} clips\n")

    if args.dry_run:
        for name, dur, prompt in PROMPTS:
            print(f"  [{dur:.0f}s] {name:35s} | {prompt}")
        return

    for i, (name, dur, prompt) in enumerate(PROMPTS):
        generate_one(name, dur, prompt, output_dir, args.base_seed + i * 100, args.samples_per_prompt)

    print(f"\nDone. New K1 clips in {output_dir}/k1/")


if __name__ == "__main__":
    main()
