#!/usr/bin/env python3
# =============================================================================
#  play_policy.py
# =============================================================================

import argparse
import glob
import subprocess
import os
import pickle
import sys
import time
import csv
from dataclasses import asdict, is_dataclass
from typing import Any, List, Sequence
import matplotlib.pyplot as plt

import torch
import numpy as np
from isaaclab.app import AppLauncher
import cli_args
from omegaconf import OmegaConf, DictConfig, ListConfig

# -----------------------------------------------------------------------------#
#  Constants                                                                   #
# -----------------------------------------------------------------------------#

SIM_ENVIRONMENTS = {
    "vanilla": "custom-Isaac-Velocity-Flat-G1-Play-v0",
    "custom": "G1-flat-vel-play",
    "clf": "G1-flat-ref-play",
    "ref_tracking": "G1-flat-ref-play",
    "clf_vdot": "G1-flat-ref-play",
    "stair": "G1-stair-play",
    "height-scan-flat": "G1-height-scan-flat-play",
    "flat-hzd": "G1-flat-hzd-play",
    "flat-hzd-GL": "G1-flat-hzd-GL-play",
    "flat-hzd-no-dr": "G1-flat-hzd-play",
    "stair-hzd": "G1-stair-hzd-play",
    "stair-hzd-GL": "G1-stair-hzd-GL-play"
    "rough": "G1-rough-clf-play",
    "slight-rough": "G1-slight-rough-clf-play",
}

EXPERIMENT_NAMES = {
    "vanilla": "g1_isaac",
    "custom": "g1",
    "clf": "g1",
    "ref_tracking": "g1",
    "clf_vdot": "g1",
    "stair": "g1",
    "height-scan-flat": "g1",
    "rough": "g1",
    "slight-rough": "g1",
}

# -----------------------------------------------------------------------------#
#  Simple data-logger                                                          #
# -----------------------------------------------------------------------------#

class DataLogger:
    def __init__(self, enabled=True, log_dir=None):
        self.enabled = enabled
        self.log_dir = log_dir
        self.data = {
            "base_velocity": [], "root_pos": [], "root_velocity": [], "terminations": [],
            "root_ang_vel_w": [], "root_quat_w": [],
            "joint_torques": [], "joint_velocities": []  
        }
        if enabled and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            print(f"[INFO] Logging data to {log_dir}")

    def log_step(self, base_vel, root_pos, root_vel, terminated, root_ang_vel, root_quat, joint_torques, joint_vels):
        if not self.enabled: return
        self.data["base_velocity"].append(base_vel)
        self.data["root_pos"].append(root_pos)
        self.data["root_velocity"].append(root_vel)
        self.data["terminations"].append(terminated)
        self.data["root_ang_vel_w"].append(root_ang_vel)
        self.data["root_quat_w"].append(root_quat)
        self.data["joint_torques"].append(joint_torques)     
        self.data["joint_velocities"].append(joint_vels)

    def save(self):
        if not (self.enabled and self.log_dir): return
        for var, entries in self.data.items():
            cleaned = [x.detach().cpu().tolist() if isinstance(x, torch.Tensor) else x for x in entries]
            path = os.path.join(self.log_dir, f"{var}.pkl")
            with open(path, "wb") as fh:
                pickle.dump(cleaned, fh)
            print(f"[INFO] Saved '{var}' to {path}")

# -----------------------------------------------------------------------------#
#  CLI                                                                         #
# -----------------------------------------------------------------------------#

def _parse_sim_speed(txt: str) -> list[float]:
    try:
        return [float(x) for x in txt.split(",")]
    except ValueError as err:
        raise argparse.ArgumentTypeError("sim_speed must be comma-separated floats, e.g. '1.0,0.0,0.0'") from err

def get_args():
    p = argparse.ArgumentParser(description="Play one or many trained policies in Isaac Lab.")
    p.add_argument("--env_type", required=True, choices=SIM_ENVIRONMENTS, help="Environment label")
    p.add_argument("--policy_paths", nargs="+", metavar="PATH", help="Checkpoint file(s) (.pt) OR run directories.")
    p.add_argument("--exp_name", help="Override default experiment name (folder inside logs/…)")
    p.add_argument("--video", action="store_true", default=False)
    p.add_argument("--video_length", type=int, default=400)
    p.add_argument("--real_time", action="store_true", default=False)
    p.add_argument("--num_envs", type=int)
    p.add_argument("--sim_speed", type=_parse_sim_speed)
    p.add_argument("--log_data", action="store_true", default=False)
    p.add_argument("--play_log_dir", type=str)
    p.add_argument("--export_policy", action="store_true", default=False)
    p.add_argument("--plot_graphs", action="store_true", default=False, help="After playback, draw graphs from the logger .pkl files.")
    cli_args.add_rsl_rl_args(p)
    AppLauncher.add_app_launcher_args(p)
    return p.parse_known_args()

# -----------------------------------------------------------------------------#
#  Helpers                                                                     #
# -----------------------------------------------------------------------------#

def to_plain_dict(cfg: Any) -> dict:
    if is_dataclass(cfg): return asdict(cfg)
    if isinstance(cfg, (DictConfig, ListConfig)): return OmegaConf.to_container(cfg, resolve=True)
    raise TypeError(f"Unsupported config type: {type(cfg)}")

def set_nested_dict(dct: dict, keys: Sequence[str], value: Any) -> None:
    for key in keys[:-1]:
        dct = dct.setdefault(key, {})
    dct[keys[-1]] = value

def apply_override(root: Any, keys: Sequence[str], value: Any) -> None:
    obj = root
    for key in keys[:-1]:
        if isinstance(obj, (dict, DictConfig, ListConfig)): obj = obj[key]
        else: obj = getattr(obj, key)
    last = keys[-1]
    if isinstance(obj, (dict, DictConfig, ListConfig)): obj[last] = value
    else: setattr(obj, last, value)

def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    return yaw

def newest_checkpoint(run_dir: str) -> str | None:
    files = glob.glob(os.path.join(run_dir, "model_*.pt"))
    return max(files, key=os.path.getmtime) if files else None

def newest_run(exp_root: str) -> str | None:
    runs = glob.glob(os.path.join(exp_root, "*"))
    return max(runs, key=os.path.getmtime) if runs else None

def find_all_play_dirs(active_run_dir: str) -> list[str]:
    parent_dir = os.path.dirname(active_run_dir)
    play_dirs: list[str] = []
    for entry in os.scandir(parent_dir):
        if not entry.is_dir(): continue
        cand = os.path.join(entry.path, "playback")
        if (os.path.isfile(os.path.join(cand, "base_velocity.pkl")) and
            os.path.isfile(os.path.join(cand, "root_pos.pkl")) and
            os.path.isfile(os.path.join(cand, "root_velocity.pkl"))):
            play_dirs.append(cand)
    return sorted(play_dirs)

def make_comparison_plots(play_dirs: list[str]) -> None:
    DT = 0.02
    if len(play_dirs) == 1:
        out_dir = play_dirs[0]
        run_name_tag = os.path.basename(os.path.dirname(out_dir))
    else:
        out_dir = os.path.join(os.getcwd(), "comparison_plots")
        run_name_tag = "vs".join(sorted(os.path.basename(os.path.dirname(d)) for d in play_dirs))
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Saving plots to: {out_dir}")
    all_stats_data = {}
    for play_dir in play_dirs:
        label = os.path.basename(os.path.dirname(play_dir))
        try:
            with open(os.path.join(play_dir, "base_velocity.pkl"), "rb") as fh: vel_raw = torch.tensor(pickle.load(fh))
            with open(os.path.join(play_dir, "root_pos.pkl"), "rb") as fh: pos_raw = torch.tensor(pickle.load(fh))
            with open(os.path.join(play_dir, "root_velocity.pkl"), "rb") as fh: root_vel_raw = torch.tensor(pickle.load(fh))
            with open(os.path.join(play_dir, "root_ang_vel_w.pkl"), "rb") as f: root_ang_vel_raw = torch.tensor(pickle.load(f))
            with open(os.path.join(play_dir, "root_quat_w.pkl"), "rb") as f: root_quat_raw = torch.tensor(pickle.load(f))
            
            stats = {
                "cmd_vel_mean": vel_raw.mean(dim=1), "cmd_vel_std": vel_raw.std(dim=1),
                "root_pos_mean": pos_raw.mean(dim=1), "root_pos_std": pos_raw.std(dim=1),
                "root_vel_mean": root_vel_raw.mean(dim=1), "root_vel_std": root_vel_raw.std(dim=1),
                "root_ang_vel_mean": root_ang_vel_raw.mean(dim=1), "root_ang_vel_std": root_ang_vel_raw.std(dim=1),
            }
            linear_cmd_vel = torch.stack([stats["cmd_vel_mean"][:, 0], stats["cmd_vel_mean"][:, 1], torch.zeros_like(stats["cmd_vel_mean"][:, 0])], dim=1)
            stats["cmd_pos_mean"] = torch.cumsum(linear_cmd_vel * DT, dim=0)
            stats["cmd_pos_std"] = torch.zeros_like(stats["root_pos_std"])
            stats["root_pos_mean"] = stats["root_pos_mean"] - stats["root_pos_mean"][0]
            
            actual_yaw_pos = quat_to_yaw(root_quat_raw.mean(dim=1))
            cmd_yaw_vel = stats["cmd_vel_mean"][:, 2]
            stats["cmd_yaw_pos_mean"] = torch.cumsum(cmd_yaw_vel * DT, dim=0)
            stats["actual_yaw_pos_mean"] = torch.from_numpy(np.unwrap(actual_yaw_pos.cpu().numpy())) - actual_yaw_pos[0]
            stats["cmd_yaw_pos_std"] = torch.zeros_like(stats["cmd_yaw_pos_mean"])
            stats["actual_yaw_pos_std"] = torch.zeros_like(stats["actual_yaw_pos_mean"])

            all_stats_data[label] = stats
            stats_path = os.path.join(play_dir, "statistics.pkl")
            stats_to_save = {k: v.cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v for k, v in stats.items()}
            with open(stats_path, "wb") as fh: pickle.dump(stats_to_save, fh)
        except FileNotFoundError as e:
            print(f"[WARNING] Could not load data for '{label}': {e}. Skipping this run.")
            continue
    if not all_stats_data:
        print("[ERROR] No data found to plot.")
        return
    
    plot_definitions = [
        ("x_vel", "X Velocity", [("cmd_vel", 0, "Cmd"), ("root_vel", 0, "Actual")]),
        ("y_vel", "Y Velocity", [("cmd_vel", 1, "Cmd"), ("root_vel", 1, "Actual")]),
        ("yaw_vel", "Yaw Velocity", [("cmd_vel", 2, "Cmd"), ("root_ang_vel", 2, "Actual")]),
        ("x_pos", "X Position (Relative)", [("cmd_pos", 0, "Cmd"), ("root_pos", 0, "Actual")]),
        ("y_pos", "Y Position (Relative)", [("cmd_pos", 1, "Cmd"), ("root_pos", 1, "Actual")]),
        ("yaw_pos", "Yaw Position (Relative)", [("cmd_yaw_pos", 0, "Cmd"), ("actual_yaw_pos", 0, "Actual")]),
        ("z_pos", "Z Position (Relative)", [("root_pos", 2, "Actual")]),
    ]

    for f_key, title, series_list in plot_definitions:
        plt.figure(figsize=(10, 6))
        full_title = f"{run_name_tag}: {title}" if len(play_dirs) == 1 else title
        plt.title(full_title)
        for run_label, stats in all_stats_data.items():
            for data_key, index, legend_suffix in series_list:
                mean_key, std_key = f"{data_key}_mean", f"{data_key}_std"
                
                # --- START FIX ---
                mean_array = np.array(stats[mean_key])
                std_array = np.array(stats[std_key])

                if mean_array.ndim > 1:
                    # It's a 2D array (like root_vel), so slice it
                    mean_data = mean_array[:, index]
                    std_data = std_array[:, index]
                else:
                    # It's a 1D array (like yaw position), so use it directly
                    mean_data = mean_array
                    std_data = std_array
                # --- END FIX ---
                
                line_label = f"{run_label} {legend_suffix}".strip() if len(play_dirs) > 1 else f"{legend_suffix}".strip()
                line, = plt.plot(mean_data, label=line_label)
                plt.fill_between(range(len(mean_data)), mean_data - std_data, mean_data + std_data, color=line.get_color(), alpha=0.2)
        
        plt.xlabel("Timestep"); plt.ylabel("Value"); plt.legend(); plt.grid(True)
        filename = f"{f_key}.png" if len(play_dirs) == 1 else f"{f_key}_{run_name_tag}.png"
        outfile = os.path.join(out_dir, filename)
        plt.savefig(outfile, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[INFO] Plot saved → {outfile}")

# -----------------------------------------------------------------------------#
#  Main                                                                        #
# -----------------------------------------------------------------------------#

def main() -> None:
    args, hydra_tail = get_args()
    multi_run = len(args.policy_paths) > 1 if args.policy_paths else False
    common_flags = ["--env_type", args.env_type]
    if args.plot_graphs:
        common_flags.append("--log_data")
    if not multi_run:
        if args.plot_graphs: common_flags.append("--plot_graphs")

    if multi_run:
        play_dirs = []
        for ckpt in args.policy_paths:
            run_dir = os.path.dirname(ckpt)
            out_dir = os.path.join(run_dir, "playback")
            play_dirs.append(out_dir)
            cmd = [sys.executable, os.path.abspath(__file__), "--policy_paths", ckpt, "--play_log_dir", out_dir] + common_flags
            subprocess.run(cmd, check=True)
        if args.plot_graphs: make_comparison_plots(play_dirs)
        sys.exit(0)

    exp_name = args.exp_name or EXPERIMENT_NAMES[args.env_type]
    exp_root = os.path.abspath(os.path.join("logs", "g1_policies", args.env_type, exp_name))
    if args.policy_paths:
        ckpts = [os.path.abspath(p) for p in args.policy_paths]
    else:
        run_dir = newest_run(exp_root)
        if not run_dir: sys.exit(f"[ERROR] No run directories found in {exp_root}")
        ckpt = newest_checkpoint(run_dir)
        if not ckpt: sys.exit(f"[ERROR] No checkpoints in latest run {run_dir}")
        ckpts = [ckpt]
    for c in ckpts:
        if not os.path.exists(c): sys.exit(f"[ERROR] Checkpoint not found: {c}")

    sys.argv = [sys.argv[0]] + hydra_tail
    if args.video: args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, export_policy_as_jit, export_policy_as_onnx
    from robot_rl.network.custom_policy_runner import CustomOnPolicyRunner
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    task_name = SIM_ENVIRONMENTS[args.env_type]
    
    for n, ckpt in enumerate(ckpts, 1):
        print(f"\n━━━ [{n}/{len(ckpts)}] ▶ {ckpt}")
        run_dir = os.path.dirname(ckpt)
        env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=args.num_envs)
        
        param_overrides = [v for k, v in os.environ.items() if k.startswith("PARAM_OVERRIDE")]
        override_tags = []
        if param_overrides:
            print("[INFO] Applying parameter overrides...")
            for override_str in param_overrides:
                try:
                    full_path, raw_val = override_str.split("=", 1)
                    key_path = full_path.split(".")
                    try:
                        val = float(raw_val) if "." in raw_val else int(raw_val)
                    except ValueError:
                        val = raw_val
                    if key_path[-1] == "velocity_range":
                            val = (val, val)
                            print(f"  - Converted to range for velocity override: {val}")
                    apply_override(env_cfg, key_path, val)
                    print(f"  - Overrode env param: {'.'.join(key_path)} = {val}")
                    suffix = "_".join(full_path.split("."))
                    override_tags.append(f"{suffix}_{str(raw_val).replace('.', 'p')}")
                except Exception as e:
                    print(f"[WARNING] Failed to parse PARAM_OVERRIDE='{override_str}': {e}")
        
        play_dir_name = "playback" + (f"_{'_'.join(override_tags)}" if override_tags else "")
        play_dir = args.play_log_dir or os.path.join(run_dir, play_dir_name)
        os.makedirs(play_dir, exist_ok=True)
        
        if args.sim_speed:
            vx, vy, wz = args.sim_speed + [0.0] * (3 - len(args.sim_speed))
            env_cfg.commands.base_velocity.ranges.lin_vel_x = (vx, vx)
            env_cfg.commands.base_velocity.ranges.lin_vel_y = (vy, vy)
            env_cfg.commands.base_velocity.ranges.ang_vel_z = (wz, wz)
        
        if hasattr(env_cfg, "__prepare_tensors__"): env_cfg.__prepare_tensors__()
        env = gym.make(task_name, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv): env = multi_agent_to_single_agent(env)
        if args.video:
            # Use the playback directory name as the unique prefix for the video file.
            video_name_prefix = os.path.basename(play_dir)
            
            env = gym.wrappers.RecordVideo(
                env, 
                video_folder=os.path.join(play_dir, "videos"), 
                step_trigger=lambda step: step == 0, 
                video_length=args.video_length, 
                name_prefix=video_name_prefix,  
                disable_logger=True
            )
        runner_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args)
        clip_val = 1.0 if runner_cfg.clip_actions is True else (None if runner_cfg.clip_actions is False else runner_cfg.clip_actions)
        env = RslRlVecEnvWrapper(env, clip_actions=clip_val)
        runner = CustomOnPolicyRunner(env, runner_cfg.to_dict(), log_dir=None, device=runner_cfg.device)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        
        if args.export_policy:
            export_dir = os.path.join(play_dir, "exported")
            os.makedirs(export_dir, exist_ok=True)
            try: policy_net = runner.alg.policy
            except AttributeError: policy_net = runner.alg.actor_critic
            export_policy_as_jit(policy_net, runner.obs_normalizer, export_dir, "policy.pt")
            export_policy_as_onnx(policy_net, runner.obs_normalizer, export_dir, "policy.onnx")
        
        logger = DataLogger(enabled=args.log_data, log_dir=play_dir)
        print("[INFO] Resetting environment at the start of the simulation.")
        obs, _ = env.reset()
        obs, _ = env.get_observations()
        dt = env.unwrapped.step_dt
        frame = 0
        print(f"[INFO] Starting simulation loop for {args.video_length} steps...")
        while app.is_running():
            if frame >= args.video_length:
                print(f"[INFO] Reached frame limit ({args.video_length}), exiting simulation loop.")
                break
            tic = time.time()
            with torch.inference_mode():
                act = policy(obs)
                obs, _, dones, info = env.step(act)
                if args.log_data:
                    cmd_vel = env.unwrapped.command_manager.get_command("base_velocity")
                    robot_data = env.unwrapped.scene["robot"].data
                    root_pos = robot_data.root_pos_w
                    root_vel = robot_data.root_lin_vel_w
                    root_ang_vel = robot_data.root_ang_vel_w
                    root_quat = robot_data.root_quat_w
                    # Get joint data from the environment
                    joint_torques = robot_data.applied_torque   # <-- ADDED
                    joint_vels = robot_data.joint_vel            # <-- ADDED
                    
                    # Pass the new data to the logger
                    logger.log_step(
                        cmd_vel.clone(), root_pos.clone(), root_vel.clone(), 
                        dones.clone(), root_ang_vel.clone(), root_quat.clone(),
                        joint_torques.clone(), joint_vels.clone() # <-- ADDED
                    )
            frame += 1
            if args.real_time:
                time.sleep(max(0.0, dt - (time.time() - tic)))
        if frame < args.video_length:
            print(f"[WARN] Simulation ended early at frame {frame} before reaching the limit of {args.video_length}.")
        env.close()

        if args.log_data:
            logger.save()

        if args.plot_graphs:
            print("\n--- Generating Plots ---")
            make_comparison_plots([play_dir])

    app.close()
    print("\nPlayback of this policy complete.")

if __name__ == "__main__":
    main()