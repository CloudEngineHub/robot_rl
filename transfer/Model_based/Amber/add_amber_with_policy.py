# add_amber_with_policy.py

import argparse
import csv
from pathlib import Path
import yaml
import numpy as np
from rl_policy_wrapper import RLPolicy

# -- add argparse arguments --
parser = argparse.ArgumentParser(
    description="Add Amber to Isaac Lab, but drive it with a trained policy."
)
parser.add_argument(
    "--config_file",
    type=Path,
    required=True,
    help="YAML with keys: checkpoint_path, dt, num_obs, num_action, period, action_scale, default_angles, qvel_scale, ang_vel_scale, command_scale",
)
parser.add_argument(
    "--csv_out",
    type=Path,
    default=Path("amber_joint_log.csv"),
    help="Where to write joint‐position logs",
)
parser.add_argument(
    "--desired_vel",
    type=float,
    nargs=3,
    default=[0.5, 0.0, 0.0],
    help="Desired base command [vx, vy, vyaw]",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments to spawn (policy will be applied to env 0 only)."
)
# append AppLauncher cli args
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# -- launch OmniVerse and then Isaac‐lab imports --
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
import omni.physics.tensors as physx
import omni.usd
from pxr import Gf, UsdPhysics, UsdGeom

# — your Amber USD & ArticulationCfg (unchanged) —
from amber_cfg import NewRobotsSceneCfg
# ───────────────────────────────────────────────────────────
# Helper: compute projected gravity in policy’s expected [qw,qx,qy,qz] ordering
def get_projected_gravity(quat: np.ndarray) -> np.ndarray:
    # quat = [qw, qx, qy, qz]
    qw, qx, qy, qz = quat
    pg = np.zeros(3, dtype=np.float32)
    pg[0] = 2 * (-qz * qx + qw * qy)
    pg[1] = -2 * (qz * qy + qw * qx)
    pg[2] = 1 - 2 * (qw * qw + qz * qz)
    return pg

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene, policy: RLPolicy):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    amber = scene["Amber"]
    device = amber.data.default_root_state.device
    n_envs = args_cli.num_envs
    assert n_envs == 1, "Policy loop only supports a single env (0)."

    # ─── CSV SET-UP ───
    joint_names = amber.data.joint_names
    csv_path = args_cli.csv_out.expanduser()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_fh = open(csv_path, "w", newline="")
    writer = csv.writer(csv_fh)
    writer.writerow(["step", "sim_time", "env_id", *joint_names])

    try:
        while simulation_app.is_running():
            simulation_app.update()

            # ─── Gather sensor data from env 0 ───
            # joint positions & velocities
            qpos = amber.data.joint_pos.cpu().numpy()[0]   # shape (7,)
            qvel = amber.data.joint_vel.cpu().numpy()[0]   # shape (7,)
            # base state: [x,y,z, qx,qy,qz,qw, linVelX,linVelY,linVelZ, angVelX,angVelY,angVelZ]
            root = amber.data.root_state_w.cpu().numpy()[0]   # (13,)
            # reorder quaternion to [qw, qx, qy, qz]
            ori = root[3:7]
            quat = np.array([ori[3], ori[0], ori[1], ori[2]], dtype=np.float32)
            body_ang_vel = root[10:13].astype(np.float32)     # (3,)
            # desired velocity command
            des_vel = np.array(args_cli.desired_vel, dtype=np.float32)

            # ─── Build observation & get action ───
            obs = policy.create_obs(
                qjoints=qpos,             # length-7 now
                body_ang_vel=body_ang_vel,
                qvel=qvel,                # length-7 now
                time=sim_time,
                projected_gravity=get_projected_gravity(quat),
                des_vel=des_vel,
            )
            action_isaac = policy.get_action_isaac()  # numpy (n_actions,)
            print(f"[DEBUG] step={count:04d} sim_time={sim_time:.3f}")
            print(f"        action_isaac (len={len(action_isaac)}): {action_isaac}")

            # ─── Convert to torch targets ───
            default_all = amber.data.default_joint_pos.clone()  # (1, n_joints)
            # create a (1, n_joints) tensor of targets
            target = torch.from_numpy(action_isaac).to(device).unsqueeze(0)  # (1, n_actions)
            # fill into full target (others stay at default)
            n_joints = default_all.shape[1]
            joint_targets = default_all.clone()
            # scatter into exactly those 4 actuated joints (no assumption on ordering):
            actuated_names = ["q1_left","q2_left","q1_right","q2_right"]
            all_names     = list(amber.data.joint_names)
            for i, name in enumerate(actuated_names):
                idx = all_names.index(name)
                joint_targets[:, idx] = target[0, i]

            # ─── Send to sim ───
            amber.set_joint_position_target(joint_targets)

            # ─── Log CSV ───
            cur_pos = amber.data.joint_pos.cpu().numpy()
            for env_id in range(n_envs):
                writer.writerow([count, sim_time, env_id, *cur_pos[env_id]])
            if count % 100 == 0:
                csv_fh.flush()

            # ─── Step physics ───
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            sim_time += sim_dt
            count += 1

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl-C – CSV saved at:", csv_path)
    finally:
        csv_fh.close()

def main():
    # ─── Initialize sim context & scene ───
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, render_interval=5, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])

    # build scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()
    print("[INFO] Isaac & scene initialized.")

    # ─── Load policy from YAML ───
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

    # ─── Run! ───
    run_simulator(sim, scene, policy)

if __name__ == "__main__":
    main()
    simulation_app.close()
