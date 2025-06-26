#!/usr/bin/env python3
"""
Sequentially play a list of trained policies.

Example
-------
python run_play.py \
    --policy_paths \
        logs/g1_policies/custom/g1/run_2025-06-26/model_400.pt \
        logs/g1_policies/custom/g1/run_2025-06-27/model_600.pt \
    -- --env_type custom --video_length 800 --headless
"""

import argparse
import os
import subprocess
import sys
from typing import List


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play a series of trained policies one after another.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Use '--' to separate the runner's own flags from those that should be
forwarded verbatim to play_policy.py.
""",
    )

    parser.add_argument(
        "--policy_paths",
        nargs="+",
        required=True,
        metavar="PATH",
        help="One or more checkpoint files or directories to play.",
    )

    # Split our arguments (parsed above) from those meant for play_policy.py
    args, passthrough = parser.parse_known_args()

    # --- locate play_policy.py in *this* folder --------------------------------
    script_dir   = os.path.dirname(os.path.realpath(__file__))
    play_script  = os.path.join(script_dir, "play_policy.py")
    if not os.path.exists(play_script):
        sys.exit(f"[ERROR] play_policy.py not found next to run_play.py ({play_script})")

    # ---------------------------------------------------------------------------
    print("\n━━━━━━━━━━ Policy Playback Sweep ━━━━━━━━━━\n")

    for idx, path in enumerate(args.policy_paths, 1):
        abs_ckpt = os.path.abspath(path)
        if not os.path.exists(abs_ckpt):
            print(f"[WARNING] Skipping '{abs_ckpt}' – file or directory not found.")
            continue

        print(f"[{idx}/{len(args.policy_paths)}] ▶ Playing: {abs_ckpt}")
        print("───────────────────────────────────────────")

        # Build the command: call play_policy.py and forward any extra flags
        cmd: List[str] = (
            [sys.executable, play_script, "--policy_paths", abs_ckpt] + passthrough
        )

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"[ERROR] play_policy.py exited with code {exc.returncode}")
            break
        except KeyboardInterrupt:
            print("\nPlayback sweep interrupted by user.")
            break

        print("───────────────────────────────────────────\n")

    print("Sweep complete.")


if __name__ == "__main__":
    main()
