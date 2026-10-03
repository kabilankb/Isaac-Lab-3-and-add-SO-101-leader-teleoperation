# MimicGen data-generation workflow (YAMLab on Isaac Sim 6)

How bimanual YAM demonstrations are generated with MimicGen in Isaac Sim 6,
merged into one LeRobot v2.0 dataset, and published to Hugging Face.

- **Task:** PutPotOnCooktop. Both YAM arms grasp the pot by its handles, lift
  it, and set it on the cooktop. HangMugOnTree is supported by the same scripts
  but has not been run on Isaac Sim 6.
- **Demonstrator:** MimicGen, via Isaac Lab Mimic. It takes a few annotated
  human teleop demos, randomizes the object poses, and adapts each demo's
  object-relative segments to the new poses. Only rollouts that pass the task's
  success check are saved.
- **Output:** a LeRobot v2.0 dataset with three cameras (top, left wrist,
  right wrist) and 14-D joint actions (two arms × (6 joints + gripper)).

Published result: [`kabilanKB/yam_put_pot`](https://huggingface.co/datasets/kabilanKB/yam_put_pot),
20 episodes (7,779 frames at 30 FPS) from 10 object pairs (`putpot_000`–`009`,
2 episodes each), domain randomization on. All 10 parts succeeded on the
first try, taking 69–108 s each and about 14 minutes in total on this laptop.

This is the YAMLab counterpart of the reBot workflow in
`~/rebot-sim-pipeline/WORKFLOW.md`. There, a scripted expert with IK is the
demonstrator. Here MimicGen fills that role, starting from real human demos.

---

## 1. How it fits together

```
 yamlab_datasets/tasks_data/PutPotOnCooktop/
 ├── annotated/annotated_putpot_NNN.hdf5   9 human demos per object pair, with subtask boundaries
 └── objects/{Pot,Cooktop}/..._NNN         USD assets (40 pairs)
                 │
                 ▼  scripts/pipeline/generate.py        (one "part" per object pair)
 ┌──────────────────────────────────────────────────────────────────────────────────┐
 │ scripts/mimic/generate_dataset.py  ×  N parts   (Isaac Sim 6, headless, 1 env)   │
 │   reset → random object poses (+ HDRI lighting, object materials)                │
 │   MimicGen: pick a source demo, transform its subtask segments to the new poses, │
 │             execute; keep only if the task succeeds                              │
 │   LeRobotRecorder → datasets/<name>_parts/putpot_NNN/  (LeRobot v2.0)            │
 └──────────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼  scripts/pipeline/merge_lerobot_v2.py
          datasets/<name>/   one LeRobot v2.0 dataset, statistics recomputed over all frames
                 │
                 ▼  scripts/pipeline/push_to_hub.py     (checks first, then uploads)
          huggingface.co/datasets/<user>/<name>
```

### Why parts and a merge

- Each annotated source file belongs to **one object pair** (for example
  `pot_003` + `cooktop_003`), so a single `generate_dataset.py` run only ever
  shows one pot and one cooktop.
- Spreading episodes over several source files gives object variety. The
  official `yamlab/yamlab_datasets` does the same: one dataset per pair.
- YAMLab's `LeRobotRecorder` **deletes** an existing dataset at
  `--output_root` (`yamlab/utils/recorders.py`) instead of appending. So each
  part gets its own folder, and `merge_lerobot_v2.py` joins them. The merge
  renumbers episodes, rewrites the parquet `episode_index`/`index` columns and
  the video names, and recomputes `stats.json` (including q01/q99) with the
  fork's own `compute_stats_from_parquet`. Merging a single dataset reproduces
  its original statistics exactly.

### How MimicGen makes a new episode

1. **Reset:** the pot and cooktop are placed at new random poses.
   Domain randomization picks an indoor HDRI (210 in `HDRIs/indoor/train`), its
   rotation, the light intensity and colour temperature, and object materials
   (`materials/train`), re-randomized every 30 control steps. The workstation
   and the robot keep their look (`randomize_workstation: false` in
   `yamlab/configs/defaults.yaml`), as in the official data.
2. **Plan:** for each subtask (grasp the handles, lift and carry, place), take
   the matching segment of a source demo. Express it relative to the object it
   manipulates, and re-anchor it to the object's new pose.
3. **Execute:** step the arms through the transformed end-effector targets,
   with the gripper clamp preventing the fingers from over-closing on the
   handles.
4. **Check:** the task's staged success check. Pick: pot lifted > 5 cm with
   both grippers holding it. Place: pot centred on the cooktop, upright, both
   grippers released.
5. **Keep:** `generation_guarantee` is on for this task, so
   `--generation_num_trials N` means **N successful episodes**; failed
   rollouts are never written.

---

## 2. Prerequisites

| What | Where (this laptop) |
|---|---|
| Code | `~/yamlab6` (Isaac Sim 6 port of YAMLab; see `README_ISAACSIM6.md`) |
| Conda env | `env_yamlab6`: Python 3.12, Isaac Sim 6.0.1, Isaac Lab 3 (`~/IsaacLab`), YAMLab's LeRobot v2.0 fork |
| Data | `~/yamlab6/yamlab_datasets` → `~/yamlab/yamlab_datasets` (annotated demos, objects, HDRIs, materials) |
| Hugging Face login | `huggingface-cli login` (token is shared across envs) |

The original `~/yamlab` (Isaac Sim 5.1, env `yam_lab`) **does not run on this
laptop**. Isaac Sim 5.1 segfaults while starting its RTX renderer on driver
595.84, before any YAMLab code runs. Use `~/yamlab6`.

Always start with:

```bash
cd ~/yamlab6
conda activate env_yamlab6
```

---

## 3. Run it

### 3.1 Smoke test (one part, no randomization, ~2 min)

```bash
python scripts/mimic/generate_dataset.py --task PutPotOnCooktop-Mimic-v0 \
  --input_file yamlab_datasets/tasks_data/PutPotOnCooktop/annotated/annotated_putpot_000.hdf5 \
  --output_root /tmp/yam_smoke --generation_num_trials 2 --num_envs 1 \
  --assets_root_path yamlab_datasets/tasks_data/PutPotOnCooktop/objects \
  --camera_width 320 --camera_height 240 --image_downsample_factor 1 \
  --task_description "Put the pot on the cooktop" --enable_gripper_clamp --seed 1 \
  --enable_cameras --headless
```

Expect `2/2 (100.0%) successful demos generated by mimic` and
`[LeRobotRecorder] Dataset consolidated: 2 episodes`.

### 3.2 Generate

```bash
python scripts/pipeline/generate.py --episodes 20 --parts 10 --seed 100 \
  --output datasets/yam_put_pot
```

| Flag | Effect |
|---|---|
| `--episodes N` | successful episodes in the final dataset |
| `--parts P` | object pairs to spread them over (source files `--first-source` … `+P-1`); episodes are split evenly |
| `--first-source K` | first annotated file index (0–39); use a new range for more object variety |
| `--seed S` | part *k* uses seed S+*k* |
| `--no-dr` | no domain randomization |
| `--task HangMugOnTree` | the other task (untested on Isaac Sim 6) |
| `--camera-width/height` | recorded resolution (default 320×240, as in the official data) |
| `--work-dir` | where parts and their logs go (default `<output>_parts/`) |

Progress:

```
[generate] part 3/10 putpot_002: 2 episodes, seed 102 (log .../yam_put_pot_parts/putpot_002.log)
[generate] part 3/10 putpot_002: exit 0, 2/2 episodes, 79s
[generate] merged 10 parts -> .../datasets/yam_put_pot: 20 episodes, 7779 frames
```

**Resuming:** if a part fails or the run is interrupted, run the same
command again. Parts that already hold their episodes are skipped. Failed
parts are regenerated.

**More data:** generate a second dataset from other object pairs
(`--first-source 10 --seed 200 --output datasets/yam_put_pot_b`) and merge the
two:

```bash
python scripts/pipeline/merge_lerobot_v2.py --output datasets/yam_put_pot_40 \
  datasets/yam_put_pot datasets/yam_put_pot_b
```

### 3.3 Check

```bash
python scripts/pipeline/push_to_hub.py datasets/yam_put_pot \
  --repo-id kabilanKB/yam_put_pot --expect-episodes 20 --check-only
```

This loads the dataset and checks the episode count, that each episode's
parquet rows match `episodes.jsonl`, and that each episode has all three
videos.

| Feature | Shape | Contents |
|---|---|---|
| `observation.images.top_rgb` | 240×320×3, video | overhead camera |
| `observation.images.left_rgb` | 240×320×3, video | left wrist camera |
| `observation.images.right_rgb` | 240×320×3, video | right wrist camera |
| `observation.state` | 14 | left arm 6 + gripper, right arm 6 + gripper |
| `action` | 14 | same layout (`meta/modality.json`) |
| task | text | "Put the pot on the cooktop" |

`meta/embodiment.json`, `meta/modality.json` and `meta/metadata.json` are
YAMLab's extra files for its downstream training pipeline. They are kept in
the merge.

### 3.4 Publish

```bash
python scripts/pipeline/push_to_hub.py datasets/yam_put_pot \
  --repo-id kabilanKB/yam_put_pot --expect-episodes 20        # add --private for a private repo
```

The same checks run first, and nothing is uploaded if any fail. It then
uploads the folder (data, videos, all `meta/` files), writes a dataset card
(license `apache-2.0`), and creates the `v2.0` branch that LeRobot v2.0 loaders
look for.

---

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Segfault in `librtx.scenedb` right after launch | You are in `~/yamlab` / `yam_lab` (Isaac Sim 5.1). Use `~/yamlab6` / `env_yamlab6` |
| `RepositoryNotFoundError ... local/...` when loading a dataset | The LeRobot fork asks the Hub for refs; pass `local_files_only=True` |
| `part putpot_NNN produced k/2 episodes` | See that part's log in `<output>_parts/`; rerun the same command to retry only that part |
| `... already exists; choose a new --output` | Neither the merge nor `generate.py` overwrites a finished dataset |
| Pot and cooktop look the same in every episode of a part | Expected: one part = one object pair. Use more `--parts` for variety |
| Workstation looks identical with DR on | Expected: `randomize_workstation: false`; lighting and object materials vary |

---

## 5. Files

| File | Role |
|---|---|
| `scripts/pipeline/generate.py` | runs one MimicGen generation per object pair, resumable, then merges |
| `scripts/pipeline/merge_lerobot_v2.py` | merges LeRobot v2.0 datasets; recomputes statistics |
| `scripts/pipeline/push_to_hub.py` | checks a dataset, then publishes it |
| `scripts/mimic/generate_dataset.py` | YAMLab's MimicGen entry point (one dataset per run) |
| `yamlab/envs/mimic/put_pot_on_cooktop_mimic_env_cfg.py` | subtasks, `generation_guarantee`, `generation_keep_failed` |
| `yamlab/configs/defaults.yaml` | `domain_randomization:` ranges, mode defaults |
| `yamlab/utils/recorders.py` | `LeRobotRecorder` (writes the dataset, YAMLab metadata files) |
