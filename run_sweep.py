import argparse
import os
import subprocess
import sys
from typing import List

def main():
    """
    Main function to parse arguments and run the hyperparameter sweep.
    """
    parser = argparse.ArgumentParser(
        description="Run multiple training sessions with varying hyperparameters for IsaacLab.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Usage:
----------------
# Run a sweep over the entropy coefficient for the 'custom' environment
python run_sweep.py \\
    --param_name "algorithm.entropy_coef" \\
    --values 0.006 0.008 0.01 \\
    -- \\
    --env_type custom --num_envs 4096

# Run a sweep over the learning rate
python run_sweep.py \\
    --param_name "algorithm.learning_rate" \\
    --values 1e-3 5e-4 \\
    -- \\
    --env_type custom --max_iterations 10000

NOTE: Use '--' to separate the sweep script's arguments from the arguments
      that need to be passed to the underlying 'train_policy.py' script.
"""
    )
    parser.add_argument(
        "--param_name",
        type=str,
        required=True,
        help="The full name of the hyperparameter to vary in the agent config (e.g., 'algorithm.entropy_coef')."
    )
    parser.add_argument(
        "--values",
        nargs='+',
        required=True,
        help="A list of values to assign to the hyperparameter for each run."
    )

    # Use parse_known_args to separate sweep-specific args from train_policy.py args
    args, pass_through_args = parser.parse_known_args()

    # Assuming this script is in the project root, construct the path to the training script
    project_root = os.path.dirname(os.path.abspath(__file__))
    train_script_path = os.path.join(project_root, "scripts", "rsl_rl", "train_policy.py")

    if not os.path.exists(train_script_path):
        print(f"Error: Training script not found at '{train_script_path}'")
        print("Please ensure this sweep script is located in the root directory of your project.")
        sys.exit(1)

    print(f"Starting hyperparameter sweep for: '{args.param_name}'")
    print(f"Values to test: {args.values}")
    print(f"Arguments passed to train_policy.py: {pass_through_args}")
    print("-" * 80)

    for value in args.values:
        print(f"🚀 Starting run with {args.param_name} = {value}")
        print("-" * 80)

        # Create a copy of the current environment variables
        run_env = os.environ.copy()
        # Set the special environment variable that train_policy.py will read
        run_env["PARAM_OVERRIDE"] = f"{args.param_name}={value}"

        # Construct the full command to execute
        command: List[str] = [sys.executable, train_script_path] + pass_through_args

        try:
            # Execute the training script as a subprocess, passing the modified environment
            subprocess.run(command, env=run_env, check=True)
            print(f"Finished run with {args.param_name} = {value}")
        except subprocess.CalledProcessError as e:
            print(f"Error during run with {args.param_name} = {value}. Exit code: {e.returncode}")
            print("Stopping sweep due to error.")
            break
        except KeyboardInterrupt:
            print("\nSweep interrupted by user. Exiting.")
            sys.exit(0)
        print("-" * 80)

    print("\nSweep complete.")

if __name__ == "__main__":
    main()
