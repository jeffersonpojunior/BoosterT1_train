"""Locomotion velocity-tracking environment for Booster T1."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from dataclasses import MISSING

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from booster_train.assets.robots.booster import BOOSTER_T1_CFG, T1_ACTION_SCALE

# Scale without Head joints (deploy controls 21 joints, no Head_Pitch/Head_Yaw)
T1_ACTION_SCALE_NO_HEAD = {k: v for k, v in T1_ACTION_SCALE.items() if "Head" not in k}

import isaaclab.terrains as terrain_gen


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with the T1 robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = MISSING

    # Height scanner attached to T1's Trunk link
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["(?!.*Head).*"],
        scale=T1_ACTION_SCALE_NO_HEAD,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy (no privileged info)."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "base_velocity"}
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(?!.*Head).*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(?!.*Head).*"])},
        )
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for randomization events."""

    # --- startup ---
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 0.9),
            "dynamic_friction_range": (0.4, 0.9),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # --- reset ---
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # --- interval ---
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # --- task rewards ---
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # --- stability penalties ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)

    # --- effort / smoothness penalties ---
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # --- gait rewards ---
    # Reward feet spending time in the air (encourages walking gait)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_foot_link", "right_foot_link"]
            ),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    # --- contact penalties ---
    # Penalize undesired body contacts (anything except feet and hands)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "Trunk",
                    "Waist",
                    "Shank_Left",
                    "Shank_Right",
                    "Hip_Roll_Left",
                    "Hip_Roll_Right",
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Terminate if the trunk touches the ground
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="Trunk"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Base environment configuration
##


@configclass
class T1LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    """Base locomotion environment configuration for Booster T1."""

    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Enable terrain curriculum when terrain_levels curriculum is active
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True

        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

        self.scene.robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

"""







"""



##
# Flat terrain variant (training)
##


@configclass
class T1LocomotionFlatEnvCfg(T1LocomotionEnvCfg):
    """T1 locomotion on flat terrain with state estimation disabled (no base_lin_vel)."""

    def __post_init__(self):
        super().__post_init__()

        # Flat terrain - disable curriculum
        self.curriculum.terrain_levels = None

        # Remove base linear velocity from observations (simulates no state estimator)
        self.observations.policy.base_lin_vel = None

        # Disable height scan on flat terrain
        self.observations.policy.height_scan = None
        self.scene.height_scanner = None


##
# Rough terrain variant (training)
##


@configclass
class T1LocomotionRoughEnvCfg(T1LocomotionEnvCfg):
    """T1 locomotion on rough terrain with state estimation disabled."""

    def __post_init__(self):
        super().__post_init__()

        # Remove base linear velocity from observations
        self.observations.policy.base_lin_vel = None

        # Switch to rough terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.debug_vis = False
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(10.0, 10.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            curriculum=True,
            sub_terrains={
                "flat": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.3,
                    noise_range=(0.0, 0.005),
                    noise_step=0.005,
                    border_width=0.25,
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.4,
                    noise_range=(-0.02, 0.02),
                    noise_step=0.005,
                    border_width=0.25,
                ),
                "pyramid_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
                    proportion=0.15,
                    step_height_range=(0.05, 0.15),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                ),
                "pyramid_stairs_inv": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
                    proportion=0.15,
                    step_height_range=(0.05, 0.15),
                    step_width=0.3,
                    platform_width=3.0,
                    border_width=1.0,
                ),
            },
        )


##
# Play variant (inference / evaluation)
##


@configclass
class T1LocomotionPlayEnvCfg(T1LocomotionFlatEnvCfg):
    """T1 locomotion play environment (no pushes, single env)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene = MySceneCfg(num_envs=1, env_spacing=2.5)
        self.scene.robot = BOOSTER_T1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner = None

        self.events.push_robot = None
        self.curriculum.terrain_levels = None

        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
