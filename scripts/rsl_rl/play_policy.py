#!/usr/bin/env python3
# =============================================================================
#  play_policy.py
#  ---------------------------------------------------------------------------
#  Play back one **or many** trained RSL-RL policies in Isaac Lab, saving
#  optional videos, trajectory data and plots.
#
#  Usage (single policy)
#  ---------------------
#  python play_policy.py \
#      --policy_paths logs/g1_policies/custom/g1/run_A/model_400.pt \
#      --env_type custom --video
#
#  Usage (multiple policies, headless)
#  -----------------------------------
#  python play_policy.py \
#      --policy_paths \
#          logs/g1_policies/custom/g1/run_A/model_400.pt \
#          logs/g1_policies/custom/g1/run_B/model_600.pt \
#      --env_type custom --headless --video_length 600
#
#  Any extra RSL-RL / AppLauncher flags may be appended as usual.
# =============================================================================

import argparse
import glob
import subprocess
import os
import pickle
import sys
import time
from dataclasses import dataclass
from typing import List, Sequence
import matplotlib.pyplot as plt

import torch
from isaaclab.app import AppLauncher
import cli_args

# -----------------------------------------------------------------------------#
#  Constants                                                                    #
# -----------------------------------------------------------------------------#

SIM_ENVIRONMENTS = {
    "vanilla": "custom-Isaac-Velocity-Flat-G1-Play-v0",
    "custom": "G1-flat-vel-play",
    "clf": "G1-flat-ref-play",
    "ref_tracking": "G1-flat-ref-play",
    "clf_vdot": "G1-flat-ref-play",
    "stair": "G1-stair-play",
    "height-scan-flat": "G1-height-scan-flat-play",
}

EXPERIMENT_NAMES = {
    "vanilla": "g1_isaac",
    "custom": "g1",
    "clf": "g1",
    "ref_tracking": "g1",
    "clf_vdot": "g1",
    "stair": "g1",
    "height-scan-flat": "g1",
}

# -----------------------------------------------------------------------------#
#  Simple data-logger                                                           #
# -----------------------------------------------------------------------------#

# -------------------------------------------------------------
#  Hard-wired DataLogger  (records base_velocity + root_pos)
# -------------------------------------------------------------
import torch, pickle, os


class DataLogger:
    def __init__(self, enabled=True, log_dir=None):
        self.enabled = enabled
        self.log_dir = log_dir
        self.data = {"base_velocity": [], "root_pos": []}

        if enabled and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            print(f"[INFO] Logging data to {log_dir}")

    # ------------------------------------------------------------------
    def log_step(self, base_vel, root_pos):
        """Append one timestep of data (tensors or arrays)."""
        if not self.enabled:
            return
        self.data["base_velocity"].append(base_vel)
        self.data["root_pos"].append(root_pos)

    # ------------------------------------------------------------------
    def save(self):
        if not (self.enabled and self.log_dir):
            return

        for var, entries in self.data.items():
            cleaned = [
                x.detach().cpu().tolist() if isinstance(x, torch.Tensor) else x
                for x in entries
            ]
            path = os.path.join(self.log_dir, f"{var}.pkl")
            with open(path, "wb") as fh:
                pickle.dump(cleaned, fh)
            print(f"[INFO] Saved '{var}' to {path}")



# -----------------------------------------------------------------------------#
#  CLI                                                                          #
# -----------------------------------------------------------------------------#


def _parse_sim_speed(txt: str) -> list[float]:
    try:
        return [float(x) for x in txt.split(",")]
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            "sim_speed must be comma-separated floats, e.g. '1.0,0.0,0.0'"
        ) from err


def get_args():
    p = argparse.ArgumentParser(
        description="Play one or many trained policies in Isaac Lab."
    )

    # core
    p.add_argument(
        "--env_type", required=True, choices=SIM_ENVIRONMENTS, help="Environment label"
    )
    p.add_argument(
        "--policy_paths",
        nargs="+",
        metavar="PATH",
        help="Checkpoint file(s) (.pt) OR run directories. "
        "If omitted, the newest checkpoint in the experiment folder is used.",
    )
    p.add_argument(
        "--exp_name",
        help="Override default experiment name (folder inside logs/…)",
    )

    # playback options
    p.add_argument("--video", action="store_true", default=False)
    p.add_argument("--video_length", type=int, default=400)
    p.add_argument("--real_time", action="store_true", default=False)
    p.add_argument("--num_envs", type=int)
    p.add_argument("--sim_speed", type=_parse_sim_speed)

    # data logging
    p.add_argument("--log_data", action="store_true", default=False)
    p.add_argument("--play_log_dir", type=str)

    # export network
    p.add_argument("--export_policy", action="store_true", default=False)

    p.add_argument("--plot_graphs", action="store_true", default=False,
                   help="After playback, draw 6 graphs from the logger .pkl files.")

    # passthrough
    cli_args.add_rsl_rl_args(p)
    AppLauncher.add_app_launcher_args(p)

    return p.parse_known_args()


# -----------------------------------------------------------------------------#
#  Helpers                                                                      #
# -----------------------------------------------------------------------------#


def newest_checkpoint(run_dir: str) -> str | None:
    files = glob.glob(os.path.join(run_dir, "model_*.pt"))
    return max(files, key=os.path.getmtime) if files else None


def newest_run(exp_root: str) -> str | None:
    runs = glob.glob(os.path.join(exp_root, "*"))
    return max(runs, key=os.path.getmtime) if runs else None


def make_comparison_plots(play_dirs: list[str]) -> None:
    import pickle

    base_keys = ["vx", "vy", "vz", "root_x", "root_y", "root_z"]
    titles    = ["Velocity X", "Velocity Y", "Velocity Z",
                 "Root X",     "Root Y",     "Root Z"]

    series = {k: [] for k in base_keys}

    # read every run’s pickles -----------------------------------------
    for play_dir in play_dirs:
        label = os.path.basename(os.path.dirname(play_dir))   # “run_A”, …
        with open(os.path.join(play_dir, "base_velocity.pkl"), "rb") as fh:
            vel_raw = pickle.load(fh)
        with open(os.path.join(play_dir, "root_pos.pkl"), "rb") as fh:
            pos_raw = pickle.load(fh)

        vel = torch.tensor(vel_raw).mean(dim=1)   # (T,3)
        pos = torch.tensor(pos_raw).mean(dim=1)   # (T,3)

        for i, k in enumerate(("vx", "vy", "vz")):
            series[k].append((label, vel[:, i].tolist()))
        for i, k in enumerate(("root_x", "root_y", "root_z")):
            series[k].append((label, pos[:, i].tolist()))

    # one figure per component -----------------------------------------
    suffix = "_".join(os.path.basename(os.path.dirname(d)) for d in play_dirs)
    out_dir = os.path.join(os.getcwd(), "comparison_plots")
    os.makedirs(out_dir, exist_ok=True)

    for key, title in zip(base_keys, titles):
        plt.figure()
        for lbl, data in series[key]:
            plt.plot(data, label=lbl)
        plt.xlabel("timestep")
        plt.ylabel(title)
        plt.title(title)
        plt.legend()
        outfile = os.path.join(out_dir, f"{key}_{suffix}.png")
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Plot saved → {outfile}")


def find_all_play_dirs(active_run_dir: str) -> list[str]:
    """
    Return every sibling “…/playback” directory (including *this* run) that
    already contains both base_velocity.pkl and root_pos.pkl.  They all live
    one level above the timestamped run directory.
    """
    parent_dir = os.path.dirname(active_run_dir)         # …/g1/<runs…>/
    play_dirs: list[str] = []

    for entry in os.scandir(parent_dir):
        if not entry.is_dir():
            continue
        cand = os.path.join(entry.path, "playback")
        if os.path.isfile(os.path.join(cand, "base_velocity.pkl")) and \
           os.path.isfile(os.path.join(cand, "root_pos.pkl")):
            play_dirs.append(cand)

    return sorted(play_dirs)



# -----------------------------------------------------------------------------#
#  Main                                                                         #
# -----------------------------------------------------------------------------#


# -----------------------------------------------------------------------------#
#  Main                                                                         #
# -----------------------------------------------------------------------------#
def main() -> None:
    args, hydra_tail = get_args()

    # ─────────────────────────────────────────────────────────────────────────
    # 1) Detect whether we were given more than one policy checkpoint
    # ─────────────────────────────────────────────────────────────────────────
    multi_run = len(args.policy_paths or []) > 1

    # Flags that every (sub-)process should receive
    common_flags = [
        "--env_type", args.env_type,
        "--log_data",           # guarantees that *.pkl files are written out
    ]

    # In the single-policy case we can show graphs immediately
    if not multi_run and args.plot_graphs:
        common_flags.append("--plot_graphs")

    # ─────────────────────────────────────────────────────────────────────────
    # 2) Multi-policy sweep: spawn this very script once per checkpoint,
    #    wait for them to finish, then generate the combined plots and exit.
    # ─────────────────────────────────────────────────────────────────────────
    if multi_run:
        play_dirs: list[str] = []

        for ckpt in args.policy_paths:
            run_dir  = os.path.dirname(ckpt)
            out_dir  = os.path.join(run_dir, "playback")
            play_dirs.append(out_dir)

            cmd = [
                sys.executable,
                os.path.abspath(__file__),     # recursively call *this* script
                "--policy_paths", ckpt,        # one checkpoint at a time
                "--play_log_dir", out_dir,     # write logs to that folder
                *common_flags,
            ]
            subprocess.run(cmd, check=True)

        # once all children have finished we can aggregate their logs
        if args.plot_graphs:
            make_comparison_plots(play_dirs)

        # important: prevent the parent from running the single-policy logic
        sys.exit(0)

    # ─────────────────────────────────────────────────────────────────────────
    # 3) Single-policy run — continue with the original code path
    # ─────────────────────────────────────────────────────────────────────────
    # (Everything below this point is unchanged from your original file)
    # -----------------------------------------------------------------------
    # Absolute path to play root (logs/…)
    exp_name = args.exp_name or EXPERIMENT_NAMES[args.env_type]
    exp_root = os.path.abspath(
        os.path.join("logs", "g1_policies", args.env_type, exp_name)
    )

    # ------------------------------------------------------------------#
    #  Resolve list of checkpoints                                      #
    # ------------------------------------------------------------------#
    if args.policy_paths:
        ckpts: list[str] = [os.path.abspath(p) for p in args.policy_paths]
    else:
        run_dir = newest_run(exp_root)
        if not run_dir:
            sys.exit(f"[ERROR] No run directories found in {exp_root}")
        ckpt = newest_checkpoint(run_dir)
        if not ckpt:
            sys.exit(f"[ERROR] No checkpoints in latest run {run_dir}")
        ckpts = [ckpt]

    # sanity-check paths
    for c in ckpts:
        if not os.path.exists(c):
            sys.exit(f"[ERROR] Checkpoint not found: {c}")

    # ------------------------------------------------------------------#
    #  Launch Isaac Lab once                                            #
    # ------------------------------------------------------------------#
    sys.argv = [sys.argv[0]] + hydra_tail  # so hydra sees only its args
    if args.video:
        args.enable_cameras = True  # AppLauncher flag

    app = AppLauncher(args).app

    # late imports
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg, get_checkpoint_path
    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        RslRlOnPolicyRunnerCfg,
        export_policy_as_onnx,
        export_policy_as_jit,
    )
    from robot_rl.network.custom_policy_runner import CustomOnPolicyRunner
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

    # pytorch knobs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # env-wide constants
    task_name = SIM_ENVIRONMENTS[args.env_type]
    dt_env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=args.num_envs)

    if args.sim_speed:
        vx, vy, wz = args.sim_speed + [0.0] * (3 - len(args.sim_speed))
        dt_env_cfg.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
        dt_env_cfg.commands.base_velocity.ranges.lin_vel_y = (vy, vy)
        dt_env_cfg.commands.base_velocity.ranges.ang_vel_z = (wz, wz)

    # ------------------------------------------------------------------#
    #  Loop over checkpoints                                            #
    # ------------------------------------------------------------------#
    for n, ckpt in enumerate(ckpts, 1):
        print(f"\n━━━ [{n}/{len(ckpts)}] ▶ {ckpt}")

        # derive play-specific log dir
        run_dir = os.path.dirname(ckpt)
        play_dir = args.play_log_dir or os.path.join(run_dir, "playback")
        os.makedirs(play_dir, exist_ok=True)

        # clone / reset env_cfg each iteration
        from isaaclab_tasks.utils import parse_env_cfg
        env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=args.num_envs)
        if hasattr(env_cfg, "__prepare_tensors__"):
            env_cfg.__prepare_tensors__()

        env = gym.make(
            task_name,
            cfg=env_cfg,
            render_mode="rgb_array" if args.video else None,
        )
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)
            
            
                # video wrapper
        if args.video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=os.path.join(play_dir, "videos"),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length,
                disable_logger=True,
            )
        
        runner_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args)
        # Determine a numeric clipping bound (or None)
        clip_val = runner_cfg.clip_actions
        if isinstance(clip_val, bool):          # True → default to 1.0, False → None
            clip_val = 1.0 if clip_val else None

        env = RslRlVecEnvWrapper(env, clip_actions=clip_val)



        # prepare RSL-RL runner
        runner = CustomOnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # optional export
        if args.export_policy:
            export_dir = os.path.join(play_dir, "exported")
            os.makedirs(export_dir, exist_ok=True)
            try:
                policy_net = runner.alg.policy  # RSL-RL ≥2.3
            except AttributeError:
                policy_net = runner.alg.actor_critic
            export_policy_as_jit(policy_net, runner.obs_normalizer, export_dir, "policy.pt")
            export_policy_as_onnx(policy_net, runner.obs_normalizer, export_dir, "policy.onnx")

        # data logger
        log_vars = ["base_velocity", "robot"]
        logger = DataLogger(enabled=args.log_data or args.plot_graphs, log_dir=play_dir)

        # ------------------------------------------------------------------#
        #  Simulation loop                                                  #
        # ------------------------------------------------------------------#
        obs, _ = env.reset()               # ← kicks-off recording
        obs, _ = env.get_observations()
        dt = env.unwrapped.step_dt
        frame = 0
        while app.is_running():
            tic = time.time()
            with torch.inference_mode():
                act = policy(obs)
                obs, _, _, info = env.step(act)
                if args.log_data:
                    # minimal example; extend as needed
                    cmd_vel = env.unwrapped.command_manager.get_command("base_velocity")
                    root_pos = env.unwrapped.scene["robot"].data.root_pos_w
                    logger.log_step(cmd_vel.clone(), root_pos.clone())

            frame += 1
            if args.video and frame >= args.video_length:
                break
            if frame >= max(100, args.video_length):  # safety stop
                break

            if args.real_time:
                time.sleep(max(0.0, dt - (time.time() - tic)))

        env.close()

        if args.log_data:
            logger.save()
            
        if args.plot_graphs:
            play_dirs_ready = find_all_play_dirs(run_dir)

            if len(play_dirs_ready) >= 2:
                # overwrite / refresh the 6 composite figures
                make_comparison_plots(play_dirs_ready)
            elif len(play_dirs_ready) == 1:
                # still useful to have single-run plots while waiting
                make_comparison_plots(play_dirs_ready)

    app.close()
    print("\nAll playbacks done.")

if __name__ == "__main__":
    main()
