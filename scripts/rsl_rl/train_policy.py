# SPDX-FileCopyrightText: Copyright (c) 2022-2025, The Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause
#

import argparse
import os
import sys
from datetime import datetime
from dataclasses import asdict, is_dataclass

from isaaclab.app import AppLauncher
import cli_args

# ----------------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------------

from omegaconf import OmegaConf, DictConfig, ListConfig


def to_plain_dict(cfg):
    """Return a plain Python dict regardless of whether *cfg* is a dataclass
    or an OmegaConf DictConfig/ListConfig."""
    if is_dataclass(cfg):
        return asdict(cfg)
    if isinstance(cfg, (DictConfig, ListConfig)):
        return OmegaConf.to_container(cfg, resolve=True)
    raise TypeError(f"Unsupported config type: {type(cfg)}")


def set_nested_attr(obj, keys, value):
    """Recursively set `obj.key1.key2… = value` (works for nested dataclasses)."""
    for key in keys[:-1]:
        obj = getattr(obj, key)
    setattr(obj, keys[-1], value)


def set_nested_dict(dct, keys, value):
    """Recursively set `dct[key1][key2]… = value`, creating sub-dicts as needed."""
    for key in keys[:-1]:
        dct = dct.setdefault(key, {})
    dct[keys[-1]] = value


# ----------------------------------------------------------------------------------
# CLI setup
# ----------------------------------------------------------------------------------

ENVIRONMENTS = {
    "vanilla": "custom-Isaac-Velocity-Flat-G1-v0",
    "custom": "G1-flat-vel",
    "ref_tracking": "G1-flat-ref-tracking",
    "clf_vdot": "G1-flat-clf-vdot",
    "clf": "G1-flat-clf",
    "stair": "G1-stair",
    "height-scan-flat": "G1-height-scan-flat",
    "rough": "G1-rough-clf",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL policies for different environments.")
    parser.add_argument(
        "--env_type",
        type=str,
        choices=list(ENVIRONMENTS.keys()),
        help="Type of environment to train on.",
    )
    parser.add_argument("--video", action="store_true", default=False, help="Record videos.")
    parser.add_argument("--video_length", type=int, default=200, help="Length of each recorded video.")
    parser.add_argument("--video_interval", type=int, default=2000, help="Video recording interval.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--max_iterations", type=int, default=None, help="Training iterations.")
    parser.add_argument("--distributed", action="store_true", default=False, help="Enable multi-GPU training.")

    # RSL-RL + AppLauncher pass-through
    cli_args.add_rsl_rl_args(parser)
    AppLauncher.add_app_launcher_args(parser)

    return parser.parse_known_args()


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------

def main():
    args_cli, hydra_args = parse_args()

    if not args_cli.env_type:
        print("Please specify an environment type with --env_type")
        print("Available:", list(ENVIRONMENTS.keys()))
        sys.exit(1)

    args_cli.task = ENVIRONMENTS[args_cli.env_type]

    if args_cli.video:
        args_cli.enable_cameras = True

    # Pass Hydra args through sys.argv so hydra_task_config picks them up.
    sys.argv = [sys.argv[0]] + hydra_args

    # Boot up Omniverse / Isaac-Lab
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Late imports (Omniverse must be running first)
    import torch
    import gymnasium as gym
    from rsl_rl.runners import OnPolicyRunner  # noqa: F401 (import for type)
    from robot_rl.network.custom_policy_runner import CustomOnPolicyRunner
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_yaml, dump_pickle
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path  # noqa: F401
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import robot_rl.tasks  # noqa: F401

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    # ----------------------------------------------------------------------
    # Training wrapper (hydra provides env_cfg & agent_cfg)
    # ----------------------------------------------------------------------
    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def train(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        agent_cfg: RslRlOnPolicyRunnerCfg,
    ):

        # ------------------------------------------------------------------
        # 1. CLI overrides that are *not* handled by Hydra itself
        # ------------------------------------------------------------------
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # Dict views of the two configs
        agent_cfg_dict = to_plain_dict(agent_cfg)
        env_cfg_dict = to_plain_dict(env_cfg)

        # Max iterations CLI override
        agent_cfg_dict["max_iterations"] = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg_dict["max_iterations"]
        )

        # ------------------------------------------------------------------
        # 2. Environment variable override (PARAM_OVERRIDE)
        #    Supports agent.*  or env.* prefixes.  Prefix omitted → agent.*
        # ------------------------------------------------------------------
        param_override = os.environ.get("PARAM_OVERRIDE")
        if param_override:
            try:
                full_path, raw_val = param_override.split("=", 1)

                # best-effort numeric cast
                try:
                    val = float(raw_val) if "." in raw_val else int(raw_val)
                except ValueError:
                    val = raw_val

                if full_path.startswith("env."):
                    cfg_obj, cfg_dict = env_cfg, env_cfg_dict
                    key_path = full_path[len("env.") :].split(".")
                elif full_path.startswith("agent."):
                    cfg_obj, cfg_dict = agent_cfg, agent_cfg_dict
                    key_path = full_path[len("agent.") :].split(".")
                else:  # default → agent
                    cfg_obj, cfg_dict = agent_cfg, agent_cfg_dict
                    key_path = full_path.split(".")

                set_nested_attr(cfg_obj, key_path, val)
                set_nested_dict(cfg_dict, key_path, val)

                suffix = "_".join(full_path.split("."))
                agent_cfg_dict["run_name"] = f"{suffix}_{str(val).replace('.', 'p')}"
                print(f"[INFO] Overrode {'env' if cfg_obj is env_cfg else 'agent'} parameter: "
                    f"param {'.'.join(key_path)} = {val}")

            except Exception as exc:
                print(f"[WARNING] Failed to parse PARAM_OVERRIDE='{param_override}': {exc}")

        # ------------------------------------------------------------------
        # 3. Seed / device handling for multi-GPU
        # ------------------------------------------------------------------
        env_cfg.seed = agent_cfg_dict["seed"]
        env_cfg.sim.device = args_cli.device if args_cli.device else env_cfg.sim.device

        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg_dict["device"] = f"cuda:{app_launcher.local_rank}"
            new_seed = agent_cfg_dict["seed"] + app_launcher.local_rank
            env_cfg.seed = agent_cfg_dict["seed"] = new_seed

        # ------------------------------------------------------------------
        # 4. Logging directory
        # ------------------------------------------------------------------
        base_log = os.path.join("logs", "g1_policies", args_cli.env_type)
        log_root = os.path.abspath(os.path.join(base_log, agent_cfg_dict["experiment_name"]))

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg_dict.get("run_name"):
            timestamp += f"_{agent_cfg_dict['run_name']}"
        log_dir = os.path.join(log_root, timestamp)

        print(f"[INFO] Logging to: {log_dir}")

        # ------------------------------------------------------------------
        # 5. Environment creation
        # ------------------------------------------------------------------
        if hasattr(env_cfg, "__prepare_tensors__"):
            env_cfg.__prepare_tensors__()

        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # ------------------------------------------------------------------
        # 6. Video recording wrapper
        # ------------------------------------------------------------------
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

        # Wrap env for RSL-RL
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_dict.get("clip_actions"))

        # ------------------------------------------------------------------
        # 7. Runner creation & (optional) resume
        # ------------------------------------------------------------------
        runner = CustomOnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg_dict["device"])
        runner.add_git_repo_to_log(__file__)

        if agent_cfg_dict.get("resume") or (
            agent_cfg_dict.get("algorithm") and agent_cfg_dict["algorithm"].get("class_name") == "Distillation"
        ):
            resume_path = agent_cfg_dict["resume_path"]
            print(f"[INFO] Resuming from checkpoint: {resume_path}")
            runner.load(resume_path)

        # ------------------------------------------------------------------
        # 8. Save configs
        # ------------------------------------------------------------------
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg_dict)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg_dict)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

        # ------------------------------------------------------------------
        # 9. Training
        # ------------------------------------------------------------------
        runner.learn(num_learning_iterations=agent_cfg_dict["max_iterations"], init_at_random_ep_len=True)
        env.close()

    # ── run hydra-wrapped training ───────────────────────────────────────
    train()
    simulation_app.close()


# ----------------------------------------------------------------------------------
# Entry-point
# ----------------------------------------------------------------------------------

if __name__ == "__main__":
    main()
