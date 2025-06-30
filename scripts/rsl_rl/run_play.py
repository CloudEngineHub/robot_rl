#!/usr/bin/env python3
"""
Sequentially play trained policies, optionally sweeping multiple parameters.

Example Usage:
----------------
# Test one policy while sweeping both X and Y push velocities
python run_play.py \\
    --policy_paths path/to/model.pt \\
    --sweep \\
        "events.push_robot.params.velocity_range.x:0.5,1.0,1.5" \\
        "events.push_robot.params.velocity_range.y:0.2,0.4,0.6" \\
    -- --env_type rough --headless --save_summary
"""

import argparse
import os
import subprocess
import sys
from typing import List

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play policies, optionally sweeping multiple hyperparameters.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--policy_paths", nargs="+", required=True, metavar="PATH",
        help="One or more checkpoint files to play.",
    )
    # --- NEW, MORE POWERFUL SWEEP ARGUMENT ---
    parser.add_argument(
        "--sweep", nargs='+',
        help="Define a sweep over one or more parameters. Each parameter is a string in the format: 'param.path:val1,val2,...'. All lists of values must have the same length."
    )

    args, passthrough = parser.parse_known_args()
    
    script_dir = os.path.dirname(os.path.realpath(__file__))
    play_script = os.path.join(script_dir, "play_policy.py")
    if not os.path.exists(play_script):
        sys.exit(f"[ERROR] play_policy.py not found at {play_script}")

    # --- Parse the sweep arguments ---
    sweep_params = {}
    num_values = 0
    if args.sweep:
        print("Parsing sweep parameters...")
        for arg in args.sweep:
            try:
                param, values_str = arg.split(':', 1)
                values = values_str.split(',')
                if num_values == 0:
                    num_values = len(values)
                elif len(values) != num_values:
                    sys.exit(f"[ERROR] All parameter value lists in --sweep must have the same length. Expected {num_values}, but found {len(values)} for '{param}'.")
                sweep_params[param] = values
                print(f"  - Found param '{param}' with {len(values)} values.")
            except ValueError:
                sys.exit(f"[ERROR] Invalid format for --sweep argument: '{arg}'. Expected 'param.path:val1,val2,...'.")

    # If no sweep is defined, create a dummy loop that runs once
    num_runs = num_values if args.sweep else 1

    print("\n━━━━━━━━━━ Policy Playback Sweep ━━━━━━━━━━\n")

    for policy_path in args.policy_paths:
        abs_ckpt = os.path.abspath(policy_path)
        if not os.path.exists(abs_ckpt):
            print(f"[WARNING] Skipping '{abs_ckpt}' – not found.")
            continue

        for i in range(num_runs):
            run_env = os.environ.copy()
            run_description = f"Playing: {os.path.basename(abs_ckpt)}"
            override_descriptions = []

            # If sweeping, set the environment variables for this specific run
            if args.sweep:
                for j, (param_name, values) in enumerate(sweep_params.items()):
                    value = values[i]
                    run_env[f"PARAM_OVERRIDE_{j}"] = f"{param_name}={value}"
                    override_descriptions.append(f"{param_name.split('.')[-1]}={value}")
                run_description += f" with {', '.join(override_descriptions)}"

            print(f"🚀 {run_description}")
            print("───────────────────────────────────────────")

            cmd: List[str] = (
                [sys.executable, play_script, "--policy_paths", abs_ckpt] + passthrough
            )

            try:
                subprocess.run(cmd, env=run_env, check=True)
            except subprocess.CalledProcessError as exc:
                print(f"[ERROR] play_policy.py exited with code {exc.returncode}")
            except KeyboardInterrupt:
                print("\nPlayback sweep interrupted by user.")
                sys.exit(0)
            
            # Clean up env vars for the next potential loop
            for j in range(len(sweep_params)):
                run_env.pop(f"PARAM_OVERRIDE_{j}", None)

            print("───────────────────────────────────────────\n")

    print("Sweep complete.")

if __name__ == "__main__":
    main()