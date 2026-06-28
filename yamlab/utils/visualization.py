"""Debug visualization helpers for inspecting USD assets in OmniGibson."""
import math
import torch as th

from omnigibson.macros import gm
gm.USE_GPU_DYNAMICS = True

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson.utils.ui_utils import choose_from_options

def load_usd_object(usd_path, short_exec=False):
    """Load a USD object into an OmniGibson scene and step the simulation.

    Args:
        usd_path (str): Path to the asset's .usd file.
        short_exec (bool): If True, run only 100 sim steps instead of 10000.
    """
    scene_options = ["Scene", "InteractiveTraversableScene"]
    scene_type = choose_from_options(options=scene_options, name="scene type")


    # Create and load this object into the simulator
    obj_cfg = dict(
        type="USDObject",
        name="obj",
        usd_path=usd_path,
        position=[0, 0, 0],
    )

    cfg = {
        "scene": {
            "type": scene_type,
        },
        "objects": [obj_cfg],
    }
    if scene_type == "InteractiveTraversableScene":
        cfg["scene"]["scene_model"] = "Rs_int"

    # Create the environment
    env = og.Environment(configs=cfg)

    # Step through the environment
    max_steps = 100 if short_exec else 10000
    for i in range(max_steps):
        env.step(th.empty(0))

    # Always close the environment at the end
    og.clear()

def visualize_usd_object(usd_path, visualize_only=False, short_exec=False):
    """Visualize a USD object in an empty scene, rotating it (and any joints) in place.

    Args:
        usd_path (str): Path to the asset's .usd file.
        visualize_only (bool): If True, load only the visual geometry (no collision/physics).
        short_exec (bool): If True, run only 100 sim steps instead of 10000.
    """
    # Define objects to load
    light0_cfg = dict(
        type="LightObject",
        light_type="Sphere",
        name="sphere_light0",
        radius=0.01,
        intensity=1e5,
        position=[-2.0, -2.0, 2.0],
    )

    light1_cfg = dict(
        type="LightObject",
        light_type="Sphere",
        name="sphere_light1",
        radius=0.01,
        intensity=1e5,
        position=[-2.0, 2.0, 2.0],
    )

    kwargs = {
        "type": "USDObject",
        "usd_path": usd_path,
    }

    # Import the desired object
    obj_cfg = dict(
        **kwargs,
        name="obj",
        visual_only=visualize_only,
        position=[0, 0, 10.0],
    )

    # Create the scene config to load -- empty scene
    cfg = {
        "scene": {
            "type": "Scene",
        },
        "objects": [light0_cfg, light1_cfg, obj_cfg],
    }

    # Create the environment
    env = og.Environment(configs=cfg)

    # Set camera to appropriate viewing pose
    og.sim.viewer_camera.set_position_orientation(
        position=th.tensor([-0.00913503, -1.95750906, 1.36407314]),
        orientation=th.tensor([0.6350064, 0.0, 0.0, 0.77250687]),
    )

    # Grab the object references
    obj = env.scene.object_registry("name", "obj")

    # Standardize the scale of the object so it fits in a [1,1,1] box -- note that we have to stop the simulator
    # in order to set the scale
    # extents = obj.aabb_extent
    og.sim.stop()
    obj.scale = 3.0
    og.sim.play()
    env.step(th.empty(0))

    # Move the object so that its center is at [0, 0, 1]
    # center_offset = obj.get_position_orientation()[0] - obj.aabb_center + th.tensor([0, 0, 1.0])
    center_offset = th.tensor([0, 0, 1.0])
    obj.set_position_orientation(position=center_offset)

    # Allow the user to easily move the camera around
    og.sim.enable_viewer_camera_teleoperation()

    # Rotate the object in place
    steps_per_rotate = 360
    steps_per_joint = steps_per_rotate / 10
    max_steps = 100 if short_exec else 10000
    for i in range(max_steps):
        z_angle = 2 * math.pi * (i % steps_per_rotate) / steps_per_rotate
        quat = T.euler2quat(th.tensor([0, 0, z_angle]))
        pos = T.quat2mat(quat) @ center_offset
        if obj.n_dof > 0:
            frac = (i % steps_per_joint) / steps_per_joint
            j_frac = -1.0 + 2.0 * frac if (i // steps_per_joint) % 2 == 0 else 1.0 - 2.0 * frac
            obj.set_joint_positions(positions=j_frac * th.ones(obj.n_dof), normalized=True, drive=False)
            obj.keep_still()
        obj.set_position_orientation(position=pos, orientation=quat)
        env.step(th.empty(0))

    # Shut down at the end
    og.clear()