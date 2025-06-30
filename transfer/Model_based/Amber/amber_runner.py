# amber_runner.py

import argparse
import yaml
from pathlib import Path

from isaaclab.app import AppLauncher   # must be before any isaaclab imports
from rl_policy_wrapper import RLPolicy
from amber_simulation import run_simulation

def main():
    parser = argparse.ArgumentParser(description="Run Amber with a trained RL policy")
    parser.add_argument("--config_file", type=str, required=True,
                        help="YAML with keys: checkpoint_path, dt, num_obs, num_action, cmd_scale, period, "
                             "action_scale, default_angles, qvel_scale, ang_vel_scale")
    parser.add_argument("--num_envs", type=int, default=1,
                        help="Number of parallel Amber envs")
    parser.add_argument("--csv", type=Path, default=Path("amber_policy_log.csv"),
                        help="Where to dump joint logs")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    # 1) Launch the simulator app
    app_launcher = AppLauncher(args)
    sim_app = app_launcher.app

    # 2) Parse RL policy config
    with open(args.config_file, "r") as f:
        cfg = yaml.safe_load(f)
    policy = RLPolicy(
        dt=cfg["dt"],
        checkpoint_path=cfg["checkpoint_path"],
        num_obs=cfg["num_obs"],
        num_action=cfg["num_action"],
        cmd_scale=cfg["command_scale"],
        period=cfg["period"],
        action_scale=cfg["action_scale"],
        default_angles=cfg["default_angles"],
        qvel_scale=cfg["qvel_scale"],
        ang_vel_scale=cfg["ang_vel_scale"],
    )

    # 3) Setup IsaacLab SimulationContext and Scene
    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from add_amber import NewRobotsSceneCfg

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        render_interval=5,
        device=args.device
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene_cfg = NewRobotsSceneCfg(args.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    scene.reset()
    print("[INFO] Amber setup complete — running policy...")

    # 4) Hand off to our policy‐driven simulator loop
    run_simulation(sim, scene, policy, args.num_envs, args.csv)

    # 5) Clean up
    sim_app.close()

if __name__ == "__main__":
    main()
