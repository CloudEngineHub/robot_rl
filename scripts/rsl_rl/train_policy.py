import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher
import cli_args

from dataclasses import asdict, is_dataclass


# Environment names
ENVIRONMENTS = {
    "vanilla": "custom-Isaac-Velocity-Flat-G1-v0",
    "custom": "G1-flat-vel",
    "ref_tracking": "G1-flat-ref-tracking",
    "clf_vdot": "G1-flat-clf-vdot",
    "clf": "G1-flat-clf",
    "stair": "G1-stair",
    "height-scan-flat": "G1-height-scan-flat",
}

def parse_args():
    parser = argparse.ArgumentParser(description="Train RL policies for different environments.")
    parser.add_argument("--env_type", type=str, choices=list(ENVIRONMENTS.keys()),
                        help="Type of environment to train on (vanilla/custom/clf)")
    parser.add_argument("--video", action="store_true", default=False,
                        help="Record videos during training.")
    parser.add_argument("--video_length", type=int, default=200,
                        help="Length of the recorded video (in steps).")
    parser.add_argument("--video_interval", type=int, default=2000,
                        help="Interval between video recordings (in steps).")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed used for the environment")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="RL Policy training iterations.")
    parser.add_argument("--distributed", action="store_true", default=False,
                        help="Run training with multiple GPUs or nodes.")
    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)
    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_known_args()

def main():
    args_cli, hydra_args = parse_args()

    if not args_cli.env_type:
        print("Please specify an environment type using --env_type")
        print("Available options:", list(ENVIRONMENTS.keys()))
        sys.exit(1)

    # Set the task based on environment type
    args_cli.task = ENVIRONMENTS[args_cli.env_type]

    # always enable cameras to record video
    if args_cli.video:
        args_cli.enable_cameras = True

    sys.argv = [sys.argv[0]] + hydra_args

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import necessary modules after app launch
    import gymnasium as gym
    import torch
    from omegaconf import OmegaConf
    from rsl_rl.runners import OnPolicyRunner
    from robot_rl.network.custom_policy_runner import CustomOnPolicyRunner
    from isaaclab.envs import (
        DirectMARLEnv,
        DirectMARLEnvCfg,
        DirectRLEnvCfg,
        ManagerBasedRLEnvCfg,
        multi_agent_to_single_agent,
    )
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml
    from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config
    import robot_rl.tasks  # noqa: F401

    # Configure PyTorch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    @hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
    def train(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
              agent_cfg: RslRlOnPolicyRunnerCfg):
        """Train with RSL-RL agent."""
        # Override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        
        # Convert agent_cfg to a dictionary to avoid OmegaConf typing issues
        if is_dataclass(agent_cfg):
            agent_cfg_dict = asdict(agent_cfg)                # convert dataclass → dict
        else:                                                 # DictConfig or ListConfig
            agent_cfg_dict = OmegaConf.to_container(agent_cfg, resolve=True)

        # Update max_iterations in the dictionary
        agent_cfg_dict['max_iterations'] = (
            args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg_dict['max_iterations']
        )
        
        # -- NEW CODE: HYPERPARAMETER OVERRIDE ON DICTIONARY --
        param_override = os.environ.get("PARAM_OVERRIDE")
        if param_override:
            try:
                param_name, param_value_str = param_override.split("=", 1)
                try:
                    if "." in param_value_str:
                         param_value = float(param_value_str)
                    else:
                         param_value = int(param_value_str)
                except ValueError:
                    param_value = param_value_str

                # Helper function to set nested dictionary values
                def set_nested_dict_value(d, keys, value):
                    for key in keys[:-1]:
                        d = d.setdefault(key, {})
                    d[keys[-1]] = value

                keys = param_name.split('.')
                set_nested_dict_value(agent_cfg_dict, keys, param_value)
                
                print(f"[INFO] Successfully overrode hyperparameter '{param_name}' with value: {param_value}")
                
                # Update run_name for clearer log directories
                run_name_suffix = param_name.replace('.', '_')
                run_name_value = str(param_value).replace('.', 'p')
                new_run_name = f"{run_name_suffix}_{run_name_value}"
                
                current_run_name = agent_cfg_dict.get('run_name') or ""
                agent_cfg_dict['run_name'] = f"{current_run_name}_{new_run_name}" if current_run_name else new_run_name

            except Exception as e:
                print(f"[WARNING] Could not parse or apply PARAM_OVERRIDE environment variable: '{param_override}'. Error: {e}")
        # -- END OF NEW CODE --

        # Set the environment seed from the dictionary
        env_cfg.seed = agent_cfg_dict['seed']
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        # Multi-gpu training configuration
        if args_cli.distributed:
            env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
            agent_cfg_dict['device'] = f"cuda:{app_launcher.local_rank}"
            seed = agent_cfg_dict['seed'] + app_launcher.local_rank
            env_cfg.seed = seed
            agent_cfg_dict['seed'] = seed

        # Create organized directory structure for logging
        base_log_path = os.path.join("logs", "g1_policies", args_cli.env_type)
        log_root_path = os.path.join(base_log_path, agent_cfg_dict['experiment_name'])
        log_root_path = os.path.abspath(log_root_path)
        print(f"[INFO] Logging experiment in directory: {log_root_path}")
        
        # Create timestamp-based run directory
        log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if agent_cfg_dict.get('run_name'):
            log_dir += f"_{agent_cfg_dict['run_name']}"
        log_dir = os.path.join(log_root_path, log_dir)

        # Create environment
        if hasattr(env_cfg, "__prepare_tensors__") and callable(getattr(env_cfg, "__prepare_tensors__")):
            env_cfg.__prepare_tensors__()
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # Convert to single-agent if needed
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # Check for resume path in the dictionary
        if agent_cfg_dict.get('resume_path') or (agent_cfg_dict.get('algorithm') and agent_cfg_dict['algorithm'].get('class_name') == "Distillation"):
            resume_path = agent_cfg_dict['resume_path']
            agent_cfg_dict['resume'] = True
        
        # Setup video recording if enabled
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "train"),
                "step_trigger": lambda step: step % args_cli.video_interval == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during training.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # Wrap environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg_dict.get('clip_actions'))

        # Create and configure runner using the dictionary
        runner = CustomOnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg_dict['device'])
        runner.add_git_repo_to_log(__file__)

        # Load checkpoint if resuming
        if agent_cfg_dict.get('resume') or (agent_cfg_dict.get('algorithm') and agent_cfg_dict['algorithm'].get('class_name') == "Distillation"):
            print(f"[INFO]: Loading model checkpoint from: {resume_path}")
            runner.load(resume_path)
        env_cfg_dict = asdict(env_cfg) if is_dataclass(env_cfg) \
            else OmegaConf.to_container(env_cfg, resolve=True)
        # Save configurations
        dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg_dict)
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg_dict)
        dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg) # save original for reference

        # Run training
        runner.learn(num_learning_iterations=agent_cfg_dict['max_iterations'], init_at_random_ep_len=True)

        # Cleanup
        env.close()

    # Run training
    train()
    # Close sim app
    simulation_app.close()

if __name__ == "__main__":
    main()
