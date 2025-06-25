import math
import torch

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import isaaclab.sim as sim_utils

# Managers
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm

# Base environment configuration
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    RewardsCfg,
    ObservationsCfg,
    EventCfg,
    CommandsCfg,
)

# MDP and asset definitions
from robot_rl.tasks.manager_based.robot_rl import mdp
from robot_rl.tasks.manager_based.robot_rl.mdp.cmd_cfg import HLIPCommandCfg
from robot_rl.assets.robots.g1_21j import G1_MINIMAL_CFG  # isort: skip


##
# Base Humanoid Configurations (previously in humanoid_env_cfg.py)
##

# Constants
PERIOD = 0.8  # (0.4 s swing phase)

@configclass
class HumanoidActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True)

@configclass
class HumanoidEventsCfg(EventCfg):
    """Base event configuration for the humanoid."""
    randomize_ground_contact_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_ankle_roll_link"]),
            "static_friction_range": (0.1, 1.25),
            "dynamic_friction_range": (0.1, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,  # ensures dynamic friction <= static friction
        },
    )

@configclass
class HumanoidRewardCfg(RewardsCfg):
    """Base reward terms for the MDP."""
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.25,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "threshold": PERIOD / 2.0,
        },
    )
    phase_contact = RewTerm(
        func=mdp.phase_contact,
        weight=0.18,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
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
    height_torso = RewTerm(func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.78})
    undesired_contacts = None
    dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    alive = RewTerm(func=mdp.is_alive, weight=0.15)
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
# G1 LIP Configurations
##

@configclass
class G1RoughLipCommandsCfg(CommandsCfg):
    """Commands for the G1 Flat environment."""
    hlip_ref = HLIPCommandCfg()


@configclass
class G1RoughLipObservationsCfg(ObservationsCfg):
    """Observation specifications for the G1 Flat environment."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        """Observations for policy group."""
        base_lin_vel = None     # Removed - no sensor
        height_scan = None      # Removed - not supported yet

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2),scale=0.25)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"},scale=(2.0,2.0,0.25))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5),scale=0.05)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))

        # Phase clock
        sin_phase = ObsTerm(func=mdp.sin_phase, params={"period": PERIOD})
        cos_phase = ObsTerm(func=mdp.cos_phase, params={"period": PERIOD})

        # des_foot_pos = ObsTerm(func=mdp.generated_commands, params={"command_name": "hlip_ref"},history_length=1,scale=(1.0,1.0))

    @configclass
    class CriticCfg(PolicyCfg):
        """Observations for critic group."""
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1),scale=1.0)
        foot_vel = ObsTerm(func=mdp.foot_vel, params={"command_name": "hlip_ref"},scale=1.0)
        foot_ang_vel = ObsTerm(func=mdp.foot_ang_vel, params={"command_name": "hlip_ref"},scale=1.0)
        ref_traj = ObsTerm(func=mdp.ref_traj, params={"command_name": "hlip_ref"},scale=1.0)
        act_traj = ObsTerm(func=mdp.act_traj, params={"command_name": "hlip_ref"},scale=1.0)
        ref_traj_vel = ObsTerm(func=mdp.ref_traj_vel, params={"command_name": "hlip_ref"},clip=(-20.0,20.0,),scale=0.1)
        act_traj_vel = ObsTerm(func=mdp.act_traj_vel, params={"command_name": "hlip_ref"},clip=(-20.0,20.0,),scale=0.1)
        height_scan = None      # Removed - not supported yet
        # v_dot = ObsTerm(func=mdp.v_dot, params={"command_name": "hlip_ref"},clip=(-1000.0,1000.0),scale=0.001)
        # v = ObsTerm(func=mdp.v, params={"command_name": "hlip_ref"},clip=(0.0,500.0),scale=0.01)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    clf_curriculum = CurrTerm(func=mdp.clf_curriculum, params={"update_interval": 1000})

@configclass
class G1RoughLipRewards(HumanoidRewardCfg):
    """Rewards specific to LIP Model"""
    holonomic_constraint = RewTerm(
        func=mdp.holonomic_constraint,
        weight=4.0,
        params={
            "command_name": "hlip_ref",
            "z_offset": 0.036,
        }
    )
    holonomic_constraint_vel = RewTerm(
        func=mdp.holonomic_constraint_vel,
        weight=2.0,
        params={
            "command_name": "hlip_ref",
        }
    )
    clf_reward = RewTerm(
        func=mdp.clf_reward,
        weight=10.0,
        params={
            "command_name": "hlip_ref",
            "max_clf": 100.0,
        }
    )
    clf_decreasing_condition = RewTerm(
        func=mdp.clf_decreasing_condition,
        weight=-2.0,
        params={
            "command_name": "hlip_ref",
            "max_clf_decreasing": 200.0,
        }
    )


##
# Environment configuration
##

@configclass
class G1RoughLipEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for the G1 Flat LIP environment."""
    
    # Assign the LIP-specific configurations
    rewards: G1RoughLipRewards = G1RoughLipRewards()
    observations: G1RoughLipObservationsCfg = G1RoughLipObservationsCfg()
    commands: G1RoughLipCommandsCfg = G1RoughLipCommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    
    # Assign the base humanoid configurations
    events: HumanoidEventsCfg = HumanoidEventsCfg()
    actions: HumanoidActionsCfg = HumanoidActionsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        ##
        # Scene
        ##
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis_link"

        # No height scanner for now
        self.scene.height_scanner = None

        ##
        # Randomization
        ##
        # self.events.push_robot = None
        self.events.push_robot.params["velocity_range"] = {"x": (-1, 1), "y": (-1, 1), "roll": (-0.4, 0.4),
                                                           "pitch": (-0.4, 0.4), "yaw": (-0.4, 0.4)}
        # self.events.push_robot.params["velocity_range"] = {"x": (-0, 0), "y": (-0, 0), "roll": (-0.0, 0.0),
        #                                                    "pitch": (-0., 0.), "yaw": (-0.0, 0.0)}
        self.events.add_base_mass.params["asset_cfg"].body_names = ["pelvis_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.add_base_mass.params["operation"] = "scale"
        # self.events.randomize_ground_contact_friction.params["static_friction_range"] = (0.1, 1.25)
        # self.events.randomize_ground_contact_friction.params["dynamic_friction_range"] = (0.1, 1.25)
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)
        # self.events.base_external_force_torque.params["asset_cfg"].body_names = ["pelvis_link"]
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

        self.events.base_external_force_torque = None
        
        ##
        # Commands
        ##
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.3, 0.3)

        ##
        # Terminations
        ##
        self.terminations.base_contact.params["sensor_cfg"].body_names = "waist_yaw_link"
        # self.terminations.base_contact.params["sensor_cfg"].body_names = ["pelvis_link"]

        ##
        # Rewards
        ##
        self.rewards.feet_air_time = None
        self.rewards.phase_contact = None
        self.rewards.lin_vel_z_l2 = None
        # self.rewards.height_torso = None
        self.rewards.feet_clearance = None
        self.rewards.ang_vel_xy_l2 = None
        self.rewards.termination_penalty = None
        self.rewards.flat_orientation_l2 = None
        self.rewards.joint_deviation_hip = None
        self.rewards.contact_no_vel = None
        self.rewards.alive = None
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        # self.rewards.track_ang_vel_z_exp.weight = 1.0

        # torque, acc, vel, action rate regularization
        self.rewards.dof_torques_l2.weight = -1.0e-5
        self.rewards.dof_pos_limits.weight = -1.0
        self.rewards.dof_acc_l2.weight = -2.5e-7
        self.rewards.dof_vel_l2.weight = -1.0e-5
        self.rewards.action_rate_l2.weight = -0.001
        # self.rewards.joint_deviation_arms.weight = -1.0             # Arms regularization
        # self.rewards.joint_deviation_torso.weight = -1.0
        
        self.rewards.joint_deviation_arms = None
        self.rewards.joint_deviation_torso = None
        # self.rewards.dof_pos_limits = None
        # self.rewards.dof_vel_l2 = None
        # self.rewards.dof_acc_l2 = None
        # self.rewards.dof_torques_l2 = None
        # self.rewards.action_rate_l2 = None   
        self.rewards.height_torso = None
        
        # self.rewards.alive.weight = 0.15
        # self.rewards.contact_no_vel.weight = -0.2
        # self.rewards.lip_gait_tracking.weight = 2
        # self.rewards.joint_deviation_hip.weight = -0.0
        # self.rewards.ang_vel_xy_l2.weight = -0.05
        # self.rewards.height_torso.weight = -1.0 #-10.0
        # self.rewards.feet_clearance.weight = -20.0
        # self.rewards.lin_vel_z_l2.weight =  -2.0
        # self.rewards.track_lin_vel_xy_exp.weight = 3.5 #1
        # self.rewards.phase_contact.weight = 0 #0.25
        
        # self.rewards.lip_feet_tracking.weight = 10.0 #10.0
        # self.rewards.flat_orientation_l2.weight = -1.0
        # self.rewards.height_torso.params["target_height"] = 0.75
        # self.rewards.feet_clearance.params["target_height"] = 0.12

    def __prepare_tensors__(self):
        """Prepare tensors for the environment."""
        self.current_des_step = torch.zeros(self.scene.num_envs, 3, device=self.sim.device)
        self.com_lin_vel_avg = torch.zeros(self.scene.num_envs, 3, device=self.sim.device)

    def define_markers(self) -> VisualizationMarkers:
        """Define markers for visualization."""
        self.footprint_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/footprint",
            markers={
                "des_foot": sim_utils.CuboidCfg(
                    size=(0.2, 0.065, 0.018),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            }
        )
        self.footprint_visualizer = VisualizationMarkers(self.footprint_cfg)