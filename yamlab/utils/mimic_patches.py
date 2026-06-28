"""MimicGen monkey-patches for IsaacLab / IsaacLab Mimic.

Import this module early (from the mimic env files) so the patches are active
before the data generator runs.
"""
import numpy as np
import isaaclab.utils.math as PoseUtils

from yamlab.utils.transforms import get_delta_object_pose

# Patch 1: register get_delta_object_pose on PoseUtils (required by the
# COORDINATION TRANSFORM scheme in data_generator.py::get_delta_pose_with_scheme()).
PoseUtils.get_delta_object_pose = get_delta_object_pose


# Patch 2: Fix SEQUENTIAL constraint to hold at step_ind == 0.
#
# The original code only decrements step_ind when step_ind > 0, so at step 0
# the arm executes the first waypoint, advances to 1, then gets pushed back to 0
# - oscillating between the first two source waypoints instead of freezing.
# This patch skips waypoint execution entirely while the constraint is pending.

_PATCHED = False


def hold_sequential_constraint_at_zero():
    """Patch DataGenerator so SEQUENTIAL_LATTER holds the arm frozen at step 0.

    Must be called after isaaclab_mimic is importable (i.e. after AppLauncher).
    Safe to call multiple times - only patches once.
    """
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    from isaaclab_mimic.datagen.data_generator import DataGenerator

    def _randomize_subtask_boundaries(self):
        """Randomize subtask boundaries, clamping each to at least 1 step.

        The original asserts end - start > 0 after randomisation; with
        aggressive offsets some subtasks collapse to zero steps. This patch
        clamps every subtask to a minimum length of 1.
        """
        result = {}
        for eef_name in self.env_cfg.subtask_configs:
            subtask_boundaries = np.array(
                self.src_demo_datagen_info_pool.subtask_boundaries[eef_name]
            )

            for i in range(subtask_boundaries.shape[1]):
                if i == 0:
                    low = self.env_cfg.subtask_configs[eef_name][i].first_subtask_start_offset_range[0]
                    high = self.env_cfg.subtask_configs[eef_name][i].first_subtask_start_offset_range[1]
                    start_offsets = np.random.randint(low, high + 1, size=subtask_boundaries.shape[0])
                    subtask_boundaries[:, i, 0] += start_offsets
                elif self.env_cfg.datagen_config.use_skillgen:
                    cfg_i = self.env_cfg.subtask_configs[eef_name][i]
                    if hasattr(cfg_i, "subtask_start_offset_range") and cfg_i.subtask_start_offset_range is not None:
                        low_s = cfg_i.subtask_start_offset_range[0]
                        high_s = cfg_i.subtask_start_offset_range[1]
                        start_offset = np.random.randint(low_s, high_s + 1, size=subtask_boundaries.shape[0])
                        subtask_boundaries[:, i, 0] += start_offset
                elif i > 0:
                    subtask_boundaries[:, i, 0] = subtask_boundaries[:, i - 1, 1]

                end_offsets = np.random.randint(
                    self.env_cfg.subtask_configs[eef_name][i].subtask_term_offset_range[0],
                    self.env_cfg.subtask_configs[eef_name][i].subtask_term_offset_range[1] + 1,
                    size=subtask_boundaries.shape[0],
                )
                subtask_boundaries[:, i, 1] = subtask_boundaries[:, i, 1] + end_offsets

            for i in range(subtask_boundaries.shape[1]):
                subtask_boundaries[:, i, 1] = np.maximum(
                    subtask_boundaries[:, i, 1],
                    subtask_boundaries[:, i, 0] + 1,
                )
                if i < subtask_boundaries.shape[1] - 1:
                    subtask_boundaries[:, i + 1, 0] = np.maximum(
                        subtask_boundaries[:, i + 1, 0],
                        subtask_boundaries[:, i, 1],
                    )

            result[eef_name] = subtask_boundaries

        return result

    DataGenerator.randomize_subtask_boundaries = _randomize_subtask_boundaries


# Patch 3: Override first-subtask source-demo selection to use nearest-neighbor
# by a specified anchor object, regardless of that subtask's own object_ref.
#
# Useful when the first subtask uses a fixed reference (random selection by
# default) but the source demos vary mainly in one object's pose: redirecting to
# nearest-neighbor by that anchor object selects the demo whose anchor pose is
# closest to the generated episode's, improving transfer.

_PATCHED_FIRST_SUBTASK_SEL = False


def select_first_subtask_by_object(
    anchor_object: str,
    nn_k: int = 1,
    pos_weight: float = 1.0,
    rot_weight: float = 0.0,
):
    """Redirect first-subtask source-demo selection to nearest-neighbor on ``anchor_object``.

    Only intercepts calls where ``subtask_object_name == "fixed_ref"`` and
    ``selection_strategy_name == "random"`` so per-subtask NN calls are unaffected.

    Args:
        anchor_object: Key in ``get_object_poses()`` to use as the NN target (e.g. ``"obj_1"``).
        nn_k: Number of nearest neighbours to sample from.
        pos_weight: Positional weight for the NN distance metric.
        rot_weight: Rotational weight. Set to 0 when only position varies in the source pool.

    Safe to call multiple times - only patches once.
    """
    global _PATCHED_FIRST_SUBTASK_SEL
    if _PATCHED_FIRST_SUBTASK_SEL:
        return
    _PATCHED_FIRST_SUBTASK_SEL = True

    from isaaclab_mimic.datagen.data_generator import DataGenerator

    _orig_select = DataGenerator.select_source_demo

    def _select_source_by_object(
        self,
        eef_name,
        eef_pose,
        object_pose,
        src_demo_current_subtask_boundaries,
        subtask_object_name,
        selection_strategy_name,
        selection_strategy_kwargs=None,
    ):
        if subtask_object_name == "fixed_ref" and selection_strategy_name == "random":
            try:
                # Keep as torch.Tensor - selection_strategy.py asserts this type.
                anchor_pose = self.env.get_object_poses()[anchor_object][0]
            except Exception:
                return _orig_select(
                    self, eef_name, eef_pose, object_pose,
                    src_demo_current_subtask_boundaries,
                    subtask_object_name, selection_strategy_name,
                    selection_strategy_kwargs,
                )
            return _orig_select(
                self,
                eef_name=eef_name,
                eef_pose=eef_pose,
                object_pose=anchor_pose,
                src_demo_current_subtask_boundaries=src_demo_current_subtask_boundaries,
                subtask_object_name=anchor_object,
                selection_strategy_name="nearest_neighbor_object",
                selection_strategy_kwargs={
                    "nn_k": nn_k,
                    "pos_weight": pos_weight,
                    "rot_weight": rot_weight,
                },
            )
        return _orig_select(
            self, eef_name, eef_pose, object_pose,
            src_demo_current_subtask_boundaries,
            subtask_object_name, selection_strategy_name,
            selection_strategy_kwargs,
        )

    DataGenerator.select_source_demo = _select_source_by_object


# Patch 4: Stop the DataGenerator from accumulating the full per-step observations
# (camera images) and scene states for every parallel env's whole trajectory — they
# are never consumed in the async + RecorderManager pipeline and exhaust RAM/VRAM.

def drop_unused_trajectory_buffers():
    """Stop the DataGenerator from retaining unused per-step observations and states.

    ``isaaclab_mimic.datagen.waypoint.MultiWaypoint.execute`` returns, every step,
    the full ``env.obs_buf`` (which for visuomotor tasks includes the camera images)
    and the full scene state. ``DataGenerator.generate`` then accumulates these into
    ``generated_obs`` / ``generated_states`` for the *entire trajectory of every
    parallel environment*. In the async + RecorderManager pipeline used here these
    returned buffers are never consumed: only the success flag is read downstream,
    and the dataset is written by the RecorderManager / LeRobot recorder through
    their own per-step hooks. For image observations this accumulation grows with
    (num_envs x episode_length x image_size), exhausting system RAM on the cpu device
    and GPU memory on the cuda device, which causes the progressive slowdown / OOM.

    This wraps ``execute`` so the returned ``states`` and ``observations`` are
    lightweight placeholders. The list *lengths* are preserved so the generator's
    ``len(states) > 0`` gate and success accumulation keep working; ``actions`` (which
    is concatenated downstream) and ``success`` are left untouched. Idempotent.
    """
    from isaaclab_mimic.datagen.waypoint import MultiWaypoint

    if getattr(MultiWaypoint.execute, "_dc_drops_buffers", False):
        return

    _orig_execute = MultiWaypoint.execute

    async def _execute_without_buffers(self, *args, **kwargs):
        result = await _orig_execute(self, *args, **kwargs)
        n = len(result.get("states", []))
        result["states"] = [None] * n
        result["observations"] = [None] * n
        return result

    _execute_without_buffers._dc_drops_buffers = True
    MultiWaypoint.execute = _execute_without_buffers
