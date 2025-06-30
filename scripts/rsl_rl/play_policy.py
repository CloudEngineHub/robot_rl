#!/usr/bin/env python3
#  ─────────────────────────────────────────────────────────────────────────────
#  play_policy.py  – single- or multi-policy playback + per-run plots
#  ─────────────────────────────────────────────────────────────────────────────
import argparse, glob, os, pickle, sys, time, subprocess
from typing import List
import matplotlib.pyplot as plt
import torch
from isaaclab.app import AppLauncher
import cli_args

# ╭───────────────────────────────────────────────────────────────────────────╮
# │  CONFIG                                                                   │
# ╰───────────────────────────────────────────────────────────────────────────╯
SIM_ENVIRONMENTS = {
    "vanilla": "custom-Isaac-Velocity-Flat-G1-Play-v0",
    "custom":  "G1-flat-vel-play",
    "clf":     "G1-flat-ref-play",
    "ref_tracking": "G1-flat-ref-play",
    "clf_vdot":     "G1-flat-ref-play",
    "stair":        "G1-stair-play",
    "height-scan-flat": "G1-height-scan-flat-play",
    "rough": "G1-rough-clf-play",
}
EXPERIMENT_NAMES = {k: "g1" for k in SIM_ENVIRONMENTS} | {"vanilla": "g1_isaac"}

# ╭───────────────────────────────────────────────────────────────────────────╮
# │  DATA LOGGER                                                              │
# ╰───────────────────────────────────────────────────────────────────────────╯
class DataLogger:
    """Stores cmd-vel, root pos, root vel → *.pkl inside <play_dir>."""
    def __init__(self, enabled: bool, log_dir: str | None):
        self.enabled, self.log_dir = enabled, log_dir
        self.data = {"base_velocity": [], "root_pos": [], "root_velocity": []}
        if enabled and log_dir:
            os.makedirs(log_dir, exist_ok=True)
            print(f"[INFO] logging → {log_dir}")

    def log_step(self, cmd_vel, root_pos, root_vel):
        if self.enabled:
            self.data["base_velocity"].append(cmd_vel)
            self.data["root_pos"].append(root_pos)
            self.data["root_velocity"].append(root_vel)

    def save(self):
        if not (self.enabled and self.log_dir):
            return
        for k, lst in self.data.items():
            cleaned = [x.cpu().tolist() if isinstance(x, torch.Tensor) else x
                       for x in lst]
            with open(os.path.join(self.log_dir, f"{k}.pkl"), "wb") as fh:
                pickle.dump(cleaned, fh)

# ╭───────────────────────────────────────────────────────────────────────────╮
# │  CLI                                                                      │
# ╰───────────────────────────────────────────────────────────────────────────╯
def _parse_sim_speed(txt): return [float(x) for x in txt.split(",")]

def get_args():
    p = argparse.ArgumentParser("Play trained RSL-RL policies in Isaac Lab")
    p.add_argument("--env_type", required=True, choices=SIM_ENVIRONMENTS)
    p.add_argument("--policy_paths", nargs="+")
    p.add_argument("--exp_name")
    p.add_argument("--video", action="store_true")
    p.add_argument("--video_length", type=int, default=400)
    p.add_argument("--real_time", action="store_true")
    p.add_argument("--num_envs", type=int)
    p.add_argument("--sim_speed", type=_parse_sim_speed)
    p.add_argument("--log_data", action="store_true")
    p.add_argument("--play_log_dir")
    p.add_argument("--export_policy", action="store_true")
    p.add_argument("--plot_graphs", action="store_true")
    cli_args.add_rsl_rl_args(p); AppLauncher.add_app_launcher_args(p)
    return p.parse_known_args()

# ╭───────────────────────────────────────────────────────────────────────────╮
# │  PLOTTING – **per single run**                                           │
# ╰───────────────────────────────────────────────────────────────────────────╯
def make_policy_plots(play_dir: str, dt: float = 0.02) -> None:
    """Create six PNGs from the pickles inside *play_dir*."""
    def _load(name):
        with open(os.path.join(play_dir, f"{name}.pkl"), "rb") as fh:
            return torch.tensor(pickle.load(fh))

    vel = _load("base_velocity")      # (T,E,3)
    pos = _load("root_pos")
    root_vel = _load("root_velocity")

    stats = {
        "cmd_vel":  vel.mean(1),              # (T,3)
        "root_vel": root_vel.mean(1),
        "root_pos": pos.mean(1),
    }
    lin_cmd = torch.stack([stats["cmd_vel"][:,0],
                           stats["cmd_vel"][:,1],
                           torch.zeros_like(stats["cmd_vel"][:,0])], 1)
    stats["cmd_pos"] = torch.cumsum(lin_cmd * dt, 0)

    out_dir = os.path.join(play_dir, "plots"); os.makedirs(out_dir, exist_ok=True)
    plots = [
        ("x_vel","X Velocity",("cmd_vel",0,"Cmd"),("root_vel",0,"Actual")),
        ("y_vel","Y Velocity",("cmd_vel",1,"Cmd"),("root_vel",1,"Actual")),
        ("z_vel","Z Velocity",("root_vel",2,"Actual")),
        ("x_pos","X Position",("cmd_pos",0,"Cmd"),("root_pos",0,"Actual")),
        ("y_pos","Y Position",("cmd_pos",1,"Cmd"),("root_pos",1,"Actual")),
        ("z_pos","Z Position",("root_pos",2,"Actual")),
    ]
    for fkey,title,*series in plots:
        plt.figure(figsize=(8,5)); plt.title(title)
        for sk,idx,label in series:
            y = stats[sk][:,idx].cpu()
            plt.plot(y,label=label)
        plt.xlabel("timestep"); plt.legend(); plt.grid(True)
        fn = os.path.join(out_dir,f"{fkey}.png")
        plt.savefig(fn,dpi=150,bbox_inches="tight"); plt.close()
        print(f"[INFO]  → {fn}")

# ╭───────────────────────────────────────────────────────────────────────────╮
# │  MAIN                                                                     │
# ╰───────────────────────────────────────────────────────────────────────────╯
def main():
    args, hydra_tail = get_args()
    multi_run = len(args.policy_paths or []) > 1

    # ── parent handles multi-policy sweep ────────────────────────────────
    if multi_run:
        for ckpt in args.policy_paths:
            run_dir = os.path.dirname(ckpt)
            play_dir = os.path.join(run_dir, "playback")
            cmd = [sys.executable, os.path.abspath(__file__),
                   "--policy_paths", ckpt,
                   "--play_log_dir", play_dir,
                   "--env_type", args.env_type,
                   "--log_data"]
            if args.plot_graphs:
                cmd.append("--plot_graphs")        # child will plot for itself
            subprocess.run(cmd, check=True)
        print("\nAll playbacks done.")
        sys.exit(0)

    # ── single-policy execution begins here ──────────────────────────────
    exp_name = args.exp_name or EXPERIMENT_NAMES[args.env_type]
    run_dir  = os.path.dirname(args.policy_paths[0])
    play_dir = args.play_log_dir or os.path.join(run_dir, "playback")
    os.makedirs(play_dir, exist_ok=True)

    # ---------- standard Isaac Lab boiler-plate (unchanged) ---------- #
    sys.argv = [sys.argv[0]] + hydra_tail
    if args.video: args.enable_cameras = True
    app = AppLauncher(args).app

    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, RslRlOnPolicyRunnerCfg, \
                                    export_policy_as_onnx, export_policy_as_jit
    from robot_rl.network.custom_policy_runner import CustomOnPolicyRunner
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent

    task_name = SIM_ENVIRONMENTS[args.env_type]
    env_cfg = parse_env_cfg(task_name, device=args.device, num_envs=args.num_envs)
    env = gym.make(task_name, cfg=env_cfg,
                   render_mode="rgb_array" if args.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    runner_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args)
    env = RslRlVecEnvWrapper(env, clip_actions=1.0)

    runner = CustomOnPolicyRunner(env, runner_cfg.to_dict(),
                                  log_dir=None, device=runner_cfg.device)
    runner.load(args.policy_paths[0])
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    logger = DataLogger(args.log_data or args.plot_graphs, log_dir=play_dir)
    obs,_ = env.reset(); obs,_ = env.get_observations()
    dt = env.unwrapped.step_dt; frame = 0
    while app.is_running():
        with torch.inference_mode():
            act = policy(obs)
            obs,_,_,_ = env.step(act)
            if args.log_data:
                cm = env.unwrapped.command_manager.get_command("base_velocity")
                rp = env.unwrapped.scene["robot"].data.root_pos_w
                rv = env.unwrapped.scene["robot"].data.root_lin_vel_w
                logger.log_step(cm.clone(), rp.clone(), rv.clone())
        frame += 1
        if frame >= max(100, args.video_length): break

    env.close(); app.close()
    if args.log_data: logger.save()
    if args.plot_graphs: make_policy_plots(play_dir)
    print("\nPlayback finished.")

if __name__ == "__main__":
    main()
