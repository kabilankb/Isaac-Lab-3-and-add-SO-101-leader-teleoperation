# Experiment Configuration

This directory holds the tunable experiment knobs for every task as layered
YAML. You change *what* a task does, *how its scene is randomized*, or *how a
mode runs* (device, parallel envs) by editing the YAML files here — no code
change. You normally edit just two files:

- `defaults.yaml` — values common to every task, plus per-mode settings.
- `tasks/<task>.yaml` — one file per task, overriding the defaults where the
  task differs.

Configs are validated when they load, so a misspelled key or a wrong value type
fails immediately with a clear error rather than being silently ignored.

The repo currently supports two tasks — **PutPotOnCooktop** (`put_pot_on_cooktop.yaml`) and
**HangMugOnTree** (`hang_mug_on_tree.yaml`); add your own by dropping a new file in `tasks/`
(see [Adding a new task](#adding-a-new-task)).

> Robot/hardware **calibration** (camera intrinsics & poses, arm base poses,
> table geometry, finger geometry, gripper limits) lives separately in
> `configs/robot/yam.yaml`. It is kept under the same `configs/` root so
> everything tunable is in one place, but it is *not* an experiment knob — those
> are physical facts of the setup, not values you tune and share to reproduce a
> run. Edit that file only to recalibrate.

## Layout

```
configs/
├── defaults.yaml              global defaults + per-mode (sim.device, sim.num_envs, DR)
├── tasks/                     one YAML per task (edit these)
│   ├── hang_mug_on_tree.yaml
│   └── put_pot_on_cooktop.yaml
├── robot/
│   └── yam.yaml               robot calibration (edit only to recalibrate)
├── loader.py                  merges + validates the layers (machinery; you don't edit this)
└── schema.py                  declares the allowed config keys/types (machinery)
```

## How overrides work

To change a value, set it at the most specific layer that should be affected.
Layers are merged lowest → highest precedence:

1. `defaults.yaml` (applies to every task)
2. `defaults.yaml` → `modes:`[mode] (per execution mode)
3. `tasks/<task>.yaml` (this task, every mode)
4. `tasks/<task>.yaml` → `modes:`[mode] (this task, one mode)
5. `tasks/<task>.yaml` → `variants:`[env_name] (one gym ID, every mode)
6. `tasks/<task>.yaml` → `variants:`[env_name] → `modes:`[mode] (one gym ID, one mode)
7. a CLI flag (highest — wins over all YAML)

Nested dicts merge key-by-key; scalars and lists replace wholesale. So putting a
value in a task file overrides the global default, a per-mode block overrides
the task's plain value for that mode, and a CLI flag overrides everything.

As a worked example (illustrative — the shipped task files don't all set these), here is how the
**rendering** settings could be overridden at each layer. We follow two of them:
`render_width`/`render_height` (the resolution, naturally set per task) and
`image_downsample_factor` (how much the recorder shrinks frames, naturally set per mode). The
global default in `defaults.yaml` is `640x480`, downsample `2`.

**2. `defaults.yaml` → `modes:`[mode]** — applies to every task, but only in
that mode:

```yaml
# defaults.yaml
modes:
  evaluation:
    rendering: {image_downsample_factor: 1}   # evaluation keeps full-res frames (any task)
```

**3. `tasks/<task>.yaml`** — this task, every mode (overrides the global
`640x480`):

```yaml
# tasks/put_pot_on_cooktop.yaml
rendering: {render_width: 1280, render_height: 960}   # this task renders 1280x960 in all modes
```

**4. `tasks/<task>.yaml` → `modes:`[mode]** — this task, one mode (overrides the
global per-mode value from layer 2):

```yaml
# tasks/put_pot_on_cooktop.yaml
modes:
  mimicgen:
    rendering: {image_downsample_factor: 4}   # this task shrinks frames 4x in mimicgen only
```

**5. `tasks/<task>.yaml` → `variants:`[env_name]** — one gym ID only (overrides
the shared task body from layer 3):

```yaml
# tasks/put_pot_on_cooktop.yaml
variants:
  PutPotOnCooktop-v1:
    rendering: {render_width: 640, render_height: 480}   # only v1; v0 stays at 1280x960
```

So for `PutPotOnCooktop-v1` in mimicgen mode, the resolved rendering is
`640x480` (from layer 5) with `image_downsample_factor: 4` (from layer 4) — and
a `--camera_width 320 --camera_height 240` flag would still override the
resolution for that run.

## Modes

Every run uses one of four execution modes: `teleoperation`, `replay`,
`mimicgen`, `evaluation`. `defaults.yaml` sets `sim.device` and `sim.num_envs` per mode
(teleop/replay → cpu, 1 env; mimicgen → cpu, 4 envs; evaluation → cuda, 64
envs), and a task may override these under its own `modes:` block. Domain
randomization is off by default in every mode; the entry script's
`--enable_domain_randomization` flag turns it on for a run.

## Registering a task or variant: `env_names` + `variants`

Task files are matched by the gym IDs they declare, not by filename. Each task
file lists the exact IDs it serves:

```yaml
env_names:
  - PutPotOnCooktop-v0
  - PutPotOnCooktop-Mimic-v0
```

To register a new variant (`PutPotOnCooktop-v1`), add its ID here. If the
variant needs *different* values from the shared task body, add a `variants:`
block keyed by the exact gym ID — it overrides the shared body for that ID only,
and may carry its own `modes:` sub-block:

```yaml
variants:
  PutPotOnCooktop-v1:
    sim: {dt: 0.0333333, decimation: 1}   # this variant only
```

## What you can set

Common knobs (defined in `defaults.yaml`, overridable per task):

- `sim` — simulation execution: physics (`dt`, `decimation`, `render_interval`,
  `enable_scene_query_support`, `physx` iteration counts), `episode_length_s` (timeout),
  and the mode-dependent `device` / `num_envs`. `dt` and `decimation` set the control rate
  (`control_hz = 1/dt / decimation`).
- `rendering` — camera render output: `render_width`, `render_height` (sensor render
  resolution), `image_downsample_factor` (recorded image = render resolution / this). (Camera
  poses and intrinsics are calibration in `robot/yam.yaml`; only `intrinsic_resolution` is fixed.)
- `recording` — LeRobot dataset frame trimming: `discard_first_n_frames`, `discard_last_n_frames`.
- `domain_randomization` — material + lighting randomization, applied during replay, MimicGen,
  and evaluation. `randomize_interval_steps` (re-randomize every N control steps; 0 = at reset
  only); `materials` (which category folders to `include`/`exclude`, `num_variants_per_material`,
  `randomize_workstation` / `randomize_robot_arms`, texture rotation/translation and color-tint
  ranges, and an optional `per_object` category map); `lighting` (`dome_intensity_range`,
  `dome_color_temperature_range`, `hdri_rotation_range_deg`). See the fully-commented block in
  `defaults.yaml`.
- `modes` — per-mode overrides of any of the above (e.g. `sim.device`, `sim.num_envs`), plus
  `enable_domain_randomization`.

> Arm controller gains and grasp-detection thresholds (`normal_force_thresh`, `check_pad`,
> `pad_min_frac`, `pad_max_frac`) are **robot properties** in `configs/robot/yam.yaml`
> (`controller:` / `grasp:`), not experiment knobs — recalibrate there, not here.

Task-specific sections (set in the task file):

- `objects` — one entry per scene object (key = object name). Per object:
  `mass` (kg; omit for articulated objects), `asset` (optional default instance
  dir; the CLI `--asset name=path` overrides it), and `randomization`
  (`region_size` XY box in m **or** `position_range` direct ± distance in m;
  `orientation_range_deg` ± yaw; `scale_range`).
- `grasp_detect` — which object(s) each arm detects a grasp on:
  `grasp_detect: {left_arm: [pot], right_arm: [pot]}`. Only the listed arms get finger
  contact sensors; the force/pad thresholds come from `configs/robot/yam.yaml` (`grasp:`).
- `kwargs` — runtime arguments forwarded to the task (with per-mode overrides).
- `success` — stage-success thresholds.
- `friction` — finger/object friction values applied during a run.
- `mimic_signals` — MimicGen subtask signal thresholds and time delays.
- `pose_schedule` — a deterministic spawn-pose sequence for systematic demo
  collection. **Teleop only** — put it under `modes.teleoperation`; other modes
  randomize instead.

The numeric blocks `kwargs`, `success`, `friction`, and `mimic_signals` accept
whatever key names your task needs. All other keys must be ones the config
schema knows about — a typo or an unrecognized global key is rejected on load.

## Adding a new task

1. Create `tasks/my_task.yaml`. Start from an existing file (e.g.
   `hang_mug_on_tree.yaml`), give it an `env_names:` list (e.g.
   `[MyTask-v0, MyTask-Mimic-v0]`), an `objects:` block, a `grasp_detect` map,
   and any task-specific sections above. Override common knobs (`sim`,
   `rendering`, …) only where the task differs from `defaults.yaml`.
2. Implement and register the task environment — copy the `put_pot_on_cooktop` env files in
   `../envs/tasks/` (and `../envs/mimic/` for a MimicGen variant) and adapt them.

If your task needs a brand-new *global* knob (one that should exist for every
task, not just a `kwargs`/`success`/`friction`/`mimic_signals` entry), it must
also be added to the config schema in `schema.py`, or it will be rejected on
load.
