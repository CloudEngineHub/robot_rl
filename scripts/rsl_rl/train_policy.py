# SPDX-FileCopyrightText: Copyright (c) 2022-2025, The Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause
#

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from dataclasses import asdict, is_dataclass
from typing import Any, Sequence
from torch.utils.tensorboard import SummaryWriter 

# Isaac-Lab / project imports ---------------------------------------------------
from isaaclab.app import AppLauncher
import cli_args

# OmegaConf helpers -------------------------------------------------------------
from omegaconf import OmegaConf, DictConfig, ListConfig

# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def to_plain_dict(cfg: Any) -> dict:
    """Return a plain `dict`, regardless of *cfg* being dataclass / DictConfig."""
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    raise TypeError(f"Unsupported config type: {type(cfg)}")


def set_nested_dict(dct: dict, keys: Sequence[str], value: Any) -> None:
    """Recursively set ``dct[k1][k2]… = value`` creating sub-dicts as needed."""
    for key in keys[:-1]:
        dct = dct.setdefault(key, {})
    dct[keys[-1]] = value


def apply_override(root: Any, keys: Sequence[str], value: Any) -> None:
    """Apply an override to *root* following *keys* path.

    The traversal seamlessly handles both attribute access (dataclass/objects)
    and mapping access (dict / OmegaConf containers).
    """
    obj = root
    for key in keys[:-1]:
        if isinstance(obj, (dict, DictConfig, ListConfig)):
            obj = obj[key]
        else:
            obj = getattr(obj, key)
    last = keys[-1]
    if isinstance(obj, (dict, DictConfig, ListConfig)):
        obj[last] = value
    else:
        setattr(obj, last, value)

# -----------------------------------------------------------------------------
# CLI setup
# -----------------------------------------------------------------------------
ENVIRONMENTS = {
    "vanilla": "custom-Isaac-Velocity-Flat-G1-v0",
    "custom": "G1-flat-vel",
    "ref_tracking": "G1-flat-ref-tracking",
    "clf_vdot": "G1-flat-clf-vdot",
    "clf": "G1-flat-clf",
    "stair": "G1-stair",
    "height-scan-flat": "G1-height-scan-flat",
    "flat-hzd": "G1-flat-hzd",
    "rough": "G1-rough-clf",
    "slight-rough": "G1-slight-rough-clf",
    "flat-hzd-no-dr": "G1-flat-hzd-no-dr",
    "stair-hzd": "G1-stair-hzd",
    "flat-hzd-GL": "G1-flat-hzd-GL",
    "stair-hzd-GL": "G1-stair-hzd-GL",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL policies for different environments.")
    parser.add_argument("--env_type", type=str, choices=list(ENVIRONMENTS.keys()), help="Environment type.")
    parser.add_argument("--video", action="store_true", help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200, help="Frames per video.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Steps between recordings.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of parallel envs.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
    parser.add_argument("--distributed", action="store_true", help="Enable multi-GPU training.")
    parser.add_argument("--grid_sweep", action="store_true", help="Enable grid sweep mode.")

    # Isaac-Lab / RSL-RL pass-throughs
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_known_args()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args_cli, hydra_args = parse_args()

    if not args_cli.env_type:
        print("Please specify --env_type (choices:", ", ".join(ENVIRONMENTS.keys()), ")")
        sys.exit(1)

    args_cli.task = ENVIRONMENTS[args_cli.env_type]
    if args_cli.video:
        args_cli.enable_cameras = True

    # Forward unknown CLI fragments to Hydra
    sys.argv = [sys.argv[0]] + hydra_args

    # Boot up Omniverse / Isaac-Lab ------------------------------------------------
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Deferred imports (require active Omniverse app) ---------------------------
    import torch  # noqa: E402
    import gymnasium as gym  # noqa: E402
    from rsl_rl.runners import OnPolicyRunner  # noqa: F401, E402
    from robot_rl.network.custom_policy_runner import CustomOnPolicyRunner  # noqa: E402
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )  # noqa: E402
    from isaaclab.utils.dict import print_dict  # noqa: E402
    from isaaclab.utils.io import dump_yaml, dump_pickle  # noqa: E402
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
    from isaaclab_tasks.utils import get_checkpoint_path  # noqa: F401, E402
    from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
    import robot_rl.tasks  # noqa: F401, E402

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ----------------------------------------------------------------------
    # Hydra-wrapped training function --------------------------------------
    # ----------------------------------------------------------------------
    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def train(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlOnPolicyRunnerCfg,
    ):
        # 1. CLI overrides not handled by Hydra ------------------------------
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs

        agent_cfg_dict = to_plain_dict(agent_cfg)
        env_cfg_dict = to_plain_dict(env_cfg)

        if args_cli.max_iterations is not None:
            agent_cfg_dict["max_iterations"] = args_cli.max_iterations

        # 2. PARAM_OVERRIDE parsing -----------------------------------------
        param_overrides = [v for k, v in os.environ.items() if k.startswith("PARAM_OVERRIDE")]
        if param_overrides:
            run_name_parts = []
            for override in param_overrides:
                try:
                    full_path, raw_val = override.split("=", 1)

                    # heuristic type cast ------------------------------------------------
                    try:
                        val = float(raw_val) if "." in raw_val else int(raw_val)
                    except ValueError:
                        val = raw_val

                    # decide root (env vs agent) ----------------------------------------
                    if full_path.startswith("env."):
                        cfg_root, cfg_dict = env_cfg, env_cfg_dict
                        key_path = full_path[4:].split(".")
                    elif full_path.startswith("agent."):
                        cfg_root, cfg_dict = agent_cfg, agent_cfg_dict
                        key_path = full_path[6:].split(".")
                    else:
                        cfg_root, cfg_dict = agent_cfg, agent_cfg_dict
                        key_path = full_path.split(".")

                    # apply to both structured cfg & dict view -------------------------
                    apply_override(cfg_root, key_path, val)
                    set_nested_dict(cfg_dict, key_path, val)

                    # tag run name -------------------------------------------------------
                    # Use a shorter name for the override in the run name
                    if full_path not in ["agent.resume", "agent.resume_path"]:
                        # Use a shorter name for the override in the run name
                        name_path = full_path
                        if name_path.startswith("env.rewards."):
                            name_path = name_path.removeprefix("env.rewards.")
                        elif name_path.startswith("agent.algorithm."):
                            name_path = name_path.removeprefix("agent.algorithm.")
                        elif name_path.startswith("algorithm."):
                            name_path = name_path.removeprefix("algorithm.")

                        suffix = "_".join(name_path.split("."))
                        run_name_parts.append(f"{suffix}_{str(val).replace('.', 'p')}")
                    print(
                        f"[INFO] Overrode {'env' if cfg_root is env_cfg else 'agent'} "
                        f"param {'.'.join(key_path)} = {val}"
                    )
                except Exception as exc:
                    print(f"[WARNING] Failed to parse PARAM_OVERRIDE='{override}': {exc}")

            if run_name_parts:
                agent_cfg_dict["run_name"] = "_".join(run_name_parts)

        # 3. Seed/device handling -------------------------------------------
        env_cfg.seed = agent_cfg_dict["seed"]
        env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device

        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg_dict["device"] = env_cfg.sim.device
            env_cfg.seed = agent_cfg_dict["seed"] = env_cfg.seed + app_launcher.local_rank

        # 4. Log directory ---------------------------------------------------
        base_log = os.path.join("logs", "g1_policies", args_cli.env_type)
        log_root = os.path.abspath(os.path.join(base_log, agent_cfg_dict["experiment_name"]))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if run_name := agent_cfg_dict.get("run_name"):
            timestamp += f"_{run_name}"
        log_dir = os.path.join(log_root, timestamp)
        print(f"[INFO] Logging to: {log_dir}")

        # 5. Env creation ----------------------------------------------------
        if hasattr(env_cfg, "__prepare_tensors__"):
            env_cfg.__prepare_tensors__()
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # 6. Video wrapper ----------------------------------------------------
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # RSL-RL VecEnv wrapper ---------------------------------------------
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_dict.get("clip_actions"))

        # 7. Runner -----------------------------------------------------------
        runner = CustomOnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg_dict["device"])
        runner.add_git_repo_to_log(__file__)

        if agent_cfg_dict.get("resume") or (
            agent_cfg_dict.get("algorithm") and agent_cfg_dict["algorithm"].get("class_name") == "Distillation"
        ):
            print(f"[INFO] Resuming from checkpoint: {agent_cfg_dict['resume_path']}")
            runner.load(agent_cfg_dict["resume_path"])
            
            # --- ▼▼ CORRECTED RESUME LOGIC ▼▼ ---
            # 1. Force the runner to use the NEW log directory, not the old one.
            runner.log_dir = log_dir

            # 2. Reset the TensorBoard writer to point to the new directory.
            if hasattr(runner, 'writer') and runner.writer is not None:
                runner.writer.close() # Close the old writer first
                runner.writer = SummaryWriter(log_dir=runner.log_dir)

        # 8. Persist configs --------------------------------------------------
        reward_terms_to_check = [
            "clf_decreasing_condition",
            "clf_reward",
            "holonomic_constraint",
            "holonomic_constraint_vel",
        ]
        # Check if the rewards dictionary exists before trying to modify it
        if "rewards" in env_cfg_dict:
            for term_name in reward_terms_to_check:
                # Check if the live object has the reward term
                if hasattr(env_cfg.rewards, term_name):
                    term_instance = getattr(env_cfg.rewards, term_name)
                    # Add the term to the dictionary if it's not None
                    if term_instance is not None:
                        if term_name not in env_cfg_dict["rewards"]:
                            print(f"[INFO] Manually adding '{term_name}' to env.yaml log.")
                            # Use to_plain_dict to convert the RewTerm object to a dictionary
                            env_cfg_dict["rewards"][term_name] = to_plain_dict(term_instance)
        
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg_dict)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg_dict)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

        # 9. Train ------------------------------------------------------------
        runner.learn(num_learning_iterations=agent_cfg_dict["max_iterations"], init_at_random_ep_len=True)
        env.close()

    # Execute Hydra-wrapped training -----------------------------------------
    train()
    simulation_app.close()

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
