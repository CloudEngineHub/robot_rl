# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, RewardsCfg, ObservationsCfg, EventCfg
from robot_rl.tasks.manager_based.robot_rl import mdp

##
# Pre-defined configs
##
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG # isort: skip


##
# MDP settings
##

# Constants
PERIOD = 0.8 # (0.4 s swing phase)

@configclass
class HumanoidActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)
    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)

@configclass
class HumanoidObservationsCfg(ObservationsCfg):
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""
        base_lin_vel = None # Removed - no sensor
        height_scan = None # Removed - not supported yet

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=1, scale=0.25)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"}, history_length=1, scale=(2.0, 2.0, 0.25))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5), history_length=1, scale=0.05)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=1)

        # Phase clock
        sin_phase = ObsTerm(func=mdp.sin_phase, params={"period": PERIOD})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"period": PERIOD})

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1), history_length=1, scale=2.0)
        height_scan = None # Removed - not supported yet

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class HumanoidEventsCfg(EventCfg):
    """Event configuration."""
    randomize_ground_contact_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "static_friction_range": (0.1, 1.25),
            "dynamic_friction_range": (0.1, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True, # ensures dynamic friction <= static friction
        },
    )

@configclass
class HumanoidRewardCfg(RewardsCfg):
    """Reward terms for the MDP."""

    ##
    # Termination
    ###
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)

    ##
    # Tracking
    ##
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_yaw_frame_exp,
    #     weight=1.0,
    #     params={"command_name": "base_velocity", "std": 0.5},
    # )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_world_exp, weight=2.0, params={"command_name": "base_velocity", "std": 0.5}
    # )

    # # Track the heading
    # track_heading = RewTerm(
    #     func=mdp.track_heading,
    #     weight=1.0,
    #     params={"command_name": "base_velocity", "std": 0.2},
    # )

    ##
    # Feet
    ##
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": PERIOD / 2.,
        },
    )

    # feet_slide = RewTerm(
    #     func=mdp.feet_slide,
    #     weight=-0.3,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     },
    # )

    # phase_feet_contacts = RewTerm(
    #     func=mdp.phase_feet_contacts,
    #     weight=0, #10
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
    #         "period": period,
    #         "std": 0.2,
    #         "nom_height": 0.78,
    #         "Tswing": period/2.,
    #         "command_name": "base_velocity",
    #         "wdes": 0.3,
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
    #     }
    # )
    
    phase_contact = RewTerm(
        func=mdp.phase_contact,
        weight=0.18,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )

    ##
    # Joint limits
    ##
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
    )

    ##
    # Penalize deviation from default of the joints that are not essential for locomotion
    ##
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                ],
            )
        },
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names="waist_yaw_joint")},
    )

    ##
    # Torso Height
    ##
    height_torso = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.78},
    )

    undesired_contacts = None

    ##
    # Regularization
    ##
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    ##
    # Feet
    ##
    feet_clearance = RewTerm(
        func=mdp.foot_clearance,
        weight=0.0,
        params={
            "target_height": 0.08,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    contact_no_vel = RewTerm(
        func=mdp.contact_no_vel,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )


##
# Environment configuration
##

@configclass
class G1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the G1 Flat environment."""

    rewards: HumanoidRewardCfg = HumanoidRewardCfg()
    observations: HumanoidObservationsCfg = HumanoidObservationsCfg()
    events: HumanoidEventsCfg = HumanoidEventsCfg()
    actions: HumanoidActionsCfg = HumanoidActionsCfg()


    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # No height scanner for now
        self.scene.height_scanner = None

        ##
        # Randomization
        ##
        # self.events.push_robot = None
        self.events.push_robot.params["velocity_range"] = {"x": (-1, 1), "y": (-1, 1), "roll": (-0.4, 0.4),
                                                           "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        # self.events.randomize_ground_contact_friction.params["static_friction_range"] = (0.1, 1.25)
        # self.events.randomize_ground_contact_friction.params["dynamic_friction_range"] = (0.1, 1.25)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

        ##
        # Commands
        ##
        self.commands.base_velocity.ranges.lin_vel_x = (-1.5, 1.5) # 0 - 1
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["pelvis_link"]


        ##
        # Rewards
        ##
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_vel_l2.weight = -1.0e-3
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.flat_orientation_l2.weight = -1.0
        self.rewards.dof_pos_limits.weight = -5.0
        self.rewards.alive.weight = 0.15
        self.rewards.contact_no_vel.weight = -0.2
        self.rewards.joint_deviation_hip.weight = -1.0
        self.rewards.height_torso.weight = -10.0
        self.rewards.feet_clearance.weight = -20.0
        self.rewards.phase_contact.weight = 0.25

        self.rewards.joint_deviation_arms.weight = -0.5
        self.rewards.joint_deviation_torso.weight = -1.0

        self.rewards.height_torso.params["target_height"] = 0.75
        self.rewards.feet_clearance.params["target_height"] = 0.12

        # # -- Regularization
        # self.rewards.dof_torques_l2.weight = -1e-4                  # Joint torques
        # # self.rewards.torque_lim.weight = -1e-2                      # Torque limits
        # self.rewards.joint_vel.weight = -1e-3                       # Joint velocity
        # self.rewards.dof_pos_limits.weight = -1 #-10                    # Joint limits
        # self.rewards.joint_reg.weight = 0. #0.25                    # Regularize positions of leg joints (relative to a nominal)
        # self.rewards.ang_vel_xy_l2.weight = -0.05                   # Base x-y angular velocity
        # self.rewards.lin_vel_z_l2.weight = -2.0                     # Base z linear velocity
        # self.rewards.flat_orientation_l2.weight = -1.5              # Tilting
        # self.rewards.action_rate_l2.weight = -0.005                 # Action smoothing
        # self.rewards.joint_deviation_hip.weight = -1.0              # Hip yaw and roll regularization
        # self.rewards.joint_deviation_arms.weight = -0.5             # Arms regularization
        # self.rewards.joint_deviation_torso.weight = -1.0
        # self.rewards.phase_feet_contacts.weight = 0.25 #1.               # Contact location
        # self.rewards.height_torso.weight = -2.                     # Base height
        # self.rewards.height_torso.params["target_height"] = 0.76
        # self.rewards.feet_clearance.weight = -20.
        # self.rewards.feet_clearance.params["target_height"] = 0.1
        # self.rewards.feet_slide.weight = -0.3
        # self.rewards.dof_acc_l2.weight = -1.25e-7
        # self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint"]
        # )
        # self.rewards.dof_torques_l2.weight = -1.5e-7
        # self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
        #     "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        # )
        #
        # # -- Task
        # self.rewards.track_lin_vel_xy_exp.weight = 2 #2.5
        # self.rewards.track_ang_vel_z_exp.weight = 0.5 #0.75  # 0
        #
        # # -- Unused
        # self.rewards.track_heading.weight = 0.                     # Base heading
        # self.rewards.feet_air_time.weight = 0.


    def __prepare_tensors__(self):
        """Move tensors to GPU"""
        # self.rewards.joint_reg.params["joint_des"] = torch.tensor(
        #     self.rewards.joint_reg.params["joint_des"],
        #     device=self.sim.device
        # )
        #
        # self.rewards.joint_reg.params["joint_weight"] = torch.tensor(
        #     self.rewards.joint_reg.params["joint_weight"],
        #     device=self.sim.device
        # )
        self.current_des_step = torch.zeros(self.scene.num_envs, 3, device=self.sim.device)
        self.com_lin_vel_avg = torch.zeros(self.scene.num_envs, 3, device=self.sim.device)

    def define_markers(self) -> VisualizationMarkers:
        """Define markers with various different shapes."""
        self.footprint_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/footprint",
            markers={
                "des_foot": sim_utils.CuboidCfg(
                    size=(0.2, 0.065, 0.018),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
                # "stance_foot": sim_utils.CuboidCfg(
                #     size=(0.2, 0.065, 0.018),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                # ),
            }
        )
        self.footprint_visualizer = VisualizationMarkers(self.footprint_cfg)
    
    # def post_physics_step(self):
    #     super().post_physics_step()
    #
    #     # Re-compute the desired foot step location
    #     if ()