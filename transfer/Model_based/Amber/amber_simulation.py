# amber_simulation.py

import torch
import numpy as np
from pathlib import Path
import sys

from rl_policy_wrapper import RLPolicy

def get_projected_gravity(quat):
    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    pg = np.zeros(3)

    pg[0] = 2 * (-qz * qx + qw * qy)
    pg[1] = -2 * (qz * qy + qw * qx)
    pg[2] = 1 - 2 * (qw * qw + qz * qz)

    return pg

def run_simulation(sim, scene, policy: RLPolicy, num_envs: int, csv_out: Path):
    """Step through the IsaacLab sim, querying the RL policy each frame."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step = 0

    amber = scene["Amber"]
    device = amber.data.default_root_state.device

    # --- CSV logger setup ---
    csv_out.parent.mkdir(exist_ok=True, parents=True)
    csv_fh = open(csv_out, "w", newline="")
    writer = __import__("csv").writer(csv_fh)
    writer.writerow(["step", "sim_time", "env_id", *amber.data.joint_names])

    try:
        while sim.app.is_running():
            # write state → sim, step physics, read back
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)

            # for each env, get obs, query policy, apply joint targets
            for env_id in range(num_envs):
                # 1) Extract robot state
                jpos = amber.data.joint_pos[env_id].cpu().numpy()            # (4,)
                jvel = amber.data.joint_vel[env_id].cpu().numpy()            # (4,)
                root = amber.data.root_state_w[env_id].cpu().numpy()         # (13,)
                body_ang_vel = root[10:13]                                   # (3,)
                quat = root[3:7]                                             # (4,)
                pg = get_projected_gravity(quat)                             # (3,)
                des_vel = np.zeros(3, dtype=np.float32)                      # you can swap in real commands

                # 2) Build and query policy
                obs = policy.create_obs(
                    qjoints=jpos,
                    body_ang_vel=body_ang_vel,
                    qvel=jvel,
                    time=sim_time,
                    projected_gravity=pg,
                    des_vel=des_vel
                )
                policy_action_mj = policy.get_action(obs)                    # MuJoCo order
                isaac_act = policy.get_action_isaac()                        # Isaac order (len 21)

                # 3) Pick out your 4 Amber joints by name
                joint_names = amber.data.joint_names                          # e.g. ["q1_left",...]
                targets = []
                for name in joint_names:
                    # map each Amber joint to your isaac_act index:
                    # e.g. if RLPolicy.isaac_to_mujoco mapping key=0 is q1_left, then index0→this value
                    # adjust these mappings if your RLPolicy wrapper differs!
                    idx = policy.isaac_to_mujoco_inv[name]  # you'll need to build this inverse map
                    targets.append(isaac_act[idx])
                targ = torch.tensor([targets], device=device)

                # 4) Send target to sim
                amber.set_joint_position_target(targ)

                # 5) Log
                writer.writerow([step, sim_time, env_id, *jpos.tolist()])

            # advance time
            if step % 100 == 0:
                csv_fh.flush()
            sim_time += sim_dt
            step += 1

    except KeyboardInterrupt:
        print(f"\n[INFO] Interrupted—saved CSV to {csv_out}")
    finally:
        csv_fh.close()
