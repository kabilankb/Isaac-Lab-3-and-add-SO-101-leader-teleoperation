# Robot Definition

This package splits into **generic robot models** (usable by any parallel-jaw robot) and a
**`yam/` package** holding everything YAM-specific. The rest of `yamlab` reads the robot
through one object, `ROBOT`, and never hard-codes a pose, gain, or finger name of its own.

```
robot/
├── ../configs/robot/yam.yaml   robot facts: arm/camera poses, table & finger geometry,
│                               gripper limits, controller gain sets   (edit to recalibrate)
├── spec.py                     RobotSpec — typed, IsaacLab-free view over a robot YAML  (generic)
├── robot.py                    Robot/Arm (end-effector-agnostic) + Gripper/Finger (parallel-jaw)
├── bimanual_robot.py           BimanualRobot — two-arm specialization of Robot           (generic)
└── yam/                        the YAM robot (a package; Python at root, USDs in subfolders)
    ├── __init__.py             ROBOT (spec), ArticulationCfgs, YamRobot, USD dir constants
    ├── arm/                    arm USD (yam.usd)
    ├── workstation/            workstation-only USD (table + walls)
    ├── yam_station/            combined arm + workstation USD
    └── yam_old/                legacy assets
```

`spec.py`, `robot.py`, and `bimanual_robot.py` are generic — they take a `RobotSpec` and
contain no YAM constants. Everything YAM-specific (which spec to load, the arm
`ArticulationCfg`s, the `YamRobot` class, the action layout) lives in the `yam/` package.

## The two layers

**`RobotSpec` (`spec.py`)** is a thin, typed, read-only view over `configs/robot/<name>.yaml`.
It imports nothing from IsaacLab, so any module — including a CLI tool that runs before the
simulator launches — can read calibration from it. Get one with `get_robot("yam")`.

```python
from yamlab.robot.yam import ROBOT          # the loaded YAM spec

ROBOT.table_position                          # (x, y, z)
ROBOT.arm_position("left")                     # arm base pose
ROBOT.arm_quaternion("left")                   # wxyz
ROBOT.finger_open("left") / ROBOT.finger_closed("left")
ROBOT.camera_position("top") / ROBOT.camera_intrinsics("top")
ROBOT.intrinsic_resolution                     # (w, h) the intrinsics were calibrated at
ROBOT.fingers                                  # {lf, rf}: contact-sensor path + pad keypoints
ROBOT.controller_gains("high_pd")              # {group: {stiffness, damping}}
```

**The `yam/` package** builds the IsaacLab side from that spec and exposes it (import all of
these from `yamlab.robot.yam`):

| Symbol | Meaning |
| --- | --- |
| `ROBOT` | the YAM `RobotSpec` (this is the import everyone uses for calibration) |
| `YAM_CONFIG` | single-arm `ArticulationCfg` with the **base** gains |
| `YAM_CONFIG_HIGH_PD_CFG` | same arm with the **high-PD** position-control gains |
| `YAM_CONFIG_DEFAULT` | whichever of the two `controller.default` selects |
| `DEFAULT_CONTROLLER` | `"high_pd"` or `"base"` |
| `YamRobot` | the runtime robot model (generic `BimanualRobot` bound to `ROBOT`) |

## Grasp is a robot capability

`RobotSpec` is *static* (geometry, gains). The *runtime* model (`robot.py` +
`bimanual_robot.py`) binds to a live scene and answers grasp queries on the robot, its arms,
and their end-effectors. The target object is an argument (defaulting to the robot's `grasp_target`):

```python
robot = self.robot                            # YamRobot, built by the base env
left, right = robot.is_grasping()              # both arms, default target object
robot.is_grasping("obj_1")                      # both arms, another object
robot.left_arm.is_grasping()                    # one arm
robot.left_arm.end_effector.is_grasping("obj_1")     # one end-effector, explicit target
```

Detection is a capability of the parts themselves: a `Finger` reads its contact sensor
(contact force + finger-pad position), a `Gripper` reports a grasp when both its fingers hold
the target, and `BimanualRobot` returns the pair of arm results. The finger links,
contact-sensor names, and pad keypoints each part needs come from `RobotSpec`
(`ROBOT.fingers`, `ROBOT.contact_sensor_topology()`, `ROBOT.arm_names`), so another
*parallel-jaw* robot is described by its YAML, not by new code. (`utils/grasp.py` holds only
the teleop grasp-ray overlay.)

`Robot` and `Arm` are end-effector-agnostic: each arm's end-effector is `Arm.END_EFFECTOR_CLS` and
each robot's arm type is `Robot.ARM_CLS`. `Gripper`/`Finger` are the **parallel-jaw**
implementation (grasp = two opposing fingers in pad contact). A different end-effector — e.g.
a multi-finger dexterous hand with its own grasp definition — is added by writing a new
`EndEffector` subclass and a small `Arm`/`Robot` subclass that points
`END_EFFECTOR_CLS`/`ARM_CLS` at it; the arm/robot/scene/spec scaffolding is reused unchanged.

## Controller gains are a robot property

Both gain sets and the active one are declared once, in `configs/robot/yam.yaml`:

```yaml
controller:
  default: high_pd            # high_pd | base
  base:    { yam_shoulder: {stiffness: 40.0,  damping: 2.5}, ... }
  high_pd: { yam_shoulder: {stiffness: 800.0, damping: 50.0}, ... }
```

The scene reads actuator gains at build time, so the choice is made at exactly one place —
`YAM_CONFIG_DEFAULT`, which the scene cfgs build their arms from. Flip `controller.default`
to switch every arm; there is no per-task controller override and no per-file gain literal.

## Adding or recalibrating a robot

Edit `configs/robot/yam.yaml` to recalibrate the existing YAM. To add another *parallel-jaw*
robot, drop a `configs/robot/<name>.yaml` with the same schema, load it with
`get_robot("<name>")`, and add a `robot/<name>/` package with its `ArticulationCfg`s and robot
class. A different morphology (DoF count, number of arms) also needs its own action layout and
a robot-specific base environment. A non-parallel-jaw end-effector (e.g. a dexterous hand)
additionally needs a new `EndEffector` subclass and an `Arm`/`Robot` subclass pointing
`END_EFFECTOR_CLS`/`ARM_CLS` at it (see "Grasp is a robot capability").
