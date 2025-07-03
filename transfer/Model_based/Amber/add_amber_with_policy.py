# add_amber_main.py
import argparse
from pathlib import Path
import yaml
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Add Amber to Isaac Lab, but drive it with a trained policy."
)
parser.add_argument(
    "--config_file", type=Path, required=True,
    help="YAML with keys: checkpoint_path, dt, num_obs, num_action, period, "
            "action_scale, default_angles, qvel_scale, ang_vel_scale, command_scale",
)
parser.add_argument(
    "--csv_out", type=Path, default=Path("amber_joint_log.csv"),
    help="Where to write joint‐position logs",
)
parser.add_argument(
    "--desired_vel", type=float, nargs=3, default=[0.2, 0.0, 0.0],
    help="Desired base command [vx, vy, vyaw]"
)
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="Number of environments to spawn (policy will be applied to env 0 only)."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# ─── launch Kit & IsaacLab ───
app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app
import torch
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene

from transfer.Model_based.Amber.rl_policy_wrapper import RLPolicy
from transfer.Model_based.Amber.amber_cfg import NewRobotsSceneCfg
from transfer.Model_based.Amber.amber_utils import get_projected_gravity, run_simulator

def main():
    
    # now that Kit is up, import other IsaacLab bits
    from isaaclab.scene import InteractiveScene

    # ─── build simulation & scene ───
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005, render_interval=5, device=args_cli.device
    )
    sim   = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    scene = InteractiveScene(NewRobotsSceneCfg(args_cli.num_envs, env_spacing=4.0))
    sim.reset(); scene.reset()
    print("[INFO] Isaac & scene initialized.")

    # ─── Load your policy ───
    with open(args_cli.config_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    policy = RLPolicy(
        dt=cfg["dt"],
        checkpoint_path=str(cfg["checkpoint_path"]),
        num_obs=cfg["num_obs"],
        num_action=cfg["num_action"],
        cmd_scale=cfg["command_scale"],
        period=cfg["period"],
        action_scale=cfg["action_scale"],
        default_angles=np.array(cfg["default_angles"], dtype=np.float32),
        qvel_scale=cfg["qvel_scale"],
        ang_vel_scale=cfg["ang_vel_scale"],
    )
    print(f"[INFO] Loaded policy from {cfg['checkpoint_path']}")
    # 1) Show the actual JIT module
    print("[DEBUG] JIT policy module:", policy.policy)
    # 2) List its parameters / state_dict keys to confirm non-empty weights
    try:
        sd = policy.policy.state_dict()
        print("[DEBUG] Policy state_dict keys (first 10):", list(sd.keys())[:10])
    except Exception as e:
        print("[DEBUG] Can't get state_dict:", e)

    # (optional) quick policy sanity check omitted for brevity…
    # 3) Run a dummy forward with zeros to see if it spits out anything non-zero
    dummy_obs = torch.zeros(1, policy.num_obs)
    if torch.cuda.is_available():
        dummy_obs = dummy_obs.cuda()
    with torch.inference_mode():
        out = policy.policy(dummy_obs)
    print("[DEBUG] policy(dummy_obs) →", out.detach().cpu().numpy().squeeze())
    # ─── run! ───
    run_simulator(sim, scene, policy, simulation_app, args_cli)

    simulation_app.close()

if __name__ == "__main__":
    main()