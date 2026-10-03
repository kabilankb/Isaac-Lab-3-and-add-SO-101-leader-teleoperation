# ACT + M3: architecture and rationale

ACT + M3 is the standard ACT policy (an encoder–decoder transformer that predicts a chunk of
future actions) with two kinds of attention masking switched on **during training only**. The
masking is adapted from the Modality Masking Mechanism (M3) in "Robust Bimanual
Vision-Language-Action Models via Embarrassingly Simple Modality Masking"
([arXiv 2608.22419](https://arxiv.org/abs/2608.22419)).

The network itself is unchanged: same parameters, same checkpoint format, and at evaluation time
it is exactly ACT. A checkpoint trained with M3 loads into a stock `ACTPolicy`.

| File | Role |
|---|---|
| `yamlab/training/m3_act.py` | The masking: `M3Config`, `M3ACT`, `M3ACTPolicy` |
| `scripts/train/train_act_m3.py` | Training entry point (wraps the LeRobot fork's `train.py`) |
| `scripts/eval/eval_act.py` | Closed-loop evaluation in the simulator, with or without the GUI |

---

## 1. The architecture as trained

```
top cam ──┐                                   state (14) ─┐   latent z (32) ─┐
left cam ─┼─ ResNet-18 ─ 80 tokens per camera ────────────┴──────────────────┤
right cam ┘   (shared)   (8×10 feature map)                                  ▼
                                                     Transformer encoder (4 layers)
                                                     242 tokens: z, state, 240 image
                                                                             │
100 learned action queries ──► Transformer decoder (4 layers) ◄─ cross-attn ─┘
                                           │
                                  linear head ──► 100 × 14 joint targets
```

| Part | Setting |
|---|---|
| Inputs | Three 320×240 RGB cameras (`top_rgb`, `left_rgb`, `right_rgb`) and the 14-value joint state |
| Vision backbone | Shared ImageNet-pretrained ResNet-18; 8×10 feature map = 80 tokens per camera |
| Encoder | 4 layers, width 512, 8 heads, over 240 image tokens + 1 state token + 1 latent token |
| Decoder | 4 layers; 100 learned queries, one per future timestep |
| Output | 100 × 14 joint-position targets: 6 joints and a gripper per arm |
| Latent z | 32-dim. In training a 4-layer encoder compresses the demonstrated action chunk into z (a variational autoencoder); at evaluation z is zero |
| Loss | L1 on the actions + KL on z (weight 10) |
| Size | 68M parameters |

Everything except the 4 decoder layers is the fork's ACT default.

## 2. What M3 adds during training

Per training sample:

- **Wrist-camera masking.** With probability 0.3, the 160 tokens from *both* wrist cameras are
  hidden from the encoder's attention and from the decoder's cross-attention. The top camera is
  never hidden.
- **Query masking.** Each of the 100 action queries is hidden from the other queries with
  probability 0.1 (never all of them). The visible queries are scaled up by 1/0.9.

In evaluation mode no mask is sampled and the model runs as plain ACT.

## 3. Why these choices

| Choice | Reason |
|---|---|
| ACT as the base policy | YAMLab had no training code. ACT is in the LeRobot fork the datasets target and is small enough to train quickly. It is also query-based like the paper's models: its decoder queries play the role of the paper's action queries, so the masking carries over directly |
| Top camera always visible | The paper's rule: the egocentric view is the stable spatial anchor. Masking it scored 37.0% in the paper's ablation |
| Both wrist cameras masked together | Hiding only one lets the model lean on spurious matches between the two wrist views. The paper reports 64.0% for both-wrists against 32.7% for one wrist |
| Query masking at 0.1 | No single query should be relied on, so information spreads across them. 0.1 was the paper's best setting |
| 4 decoder layers | The fork's default is 1. With one layer, query masking does nothing: the decoder input is all zeros, so the first layer's self-attention outputs zeros whatever its keys are |
| No language masking | The paper also masks language tokens, but ACT has no language input |

Caveats:

- The 0.3 wrist-masking probability is our choice. The paper's value was not found.
- This is an adaptation to a much smaller model than the paper's vision-language-action models,
  so the paper's results do not transfer automatically.

## 4. Train and evaluate

```bash
conda activate env_yamlab6 && cd ~/yamlab6

# ACT + M3
python scripts/train/train_act_m3.py --dataset-root datasets/yam_put_pot --preload-frames \
    --output_dir=outputs/train/act_m3 --policy.n_decoder_layers=4 \
    --steps=20000 --batch_size=32 --wandb.enable=false

# plain ACT baseline: same command with --no-m3 and another output folder

# evaluate in the simulator (drop --viz kit and add --headless to run without a window)
D=yamlab_datasets/tasks_data/PutPotOnCooktop/objects
python scripts/eval/eval_act.py --checkpoint outputs/train/act_m3/checkpoints/last/pretrained_model \
    --n_action_steps 25 --task PutPotOnCooktop-v0 \
    --asset pot=$D/Pot/pot_000 --asset cooktop=$D/Cooktop/cooktop_000 \
    --num_episodes 5 --enable_gripper_clamp --enable_cameras --viz kit
```

Masking flags: `--wrist-mask-prob`, `--query-mask-prob`, `--no-query-rescale`, `--ego-camera`.
`--preload-frames` decodes all videos into memory once, which removes the per-sample video
decoding that otherwise leaves the GPU idle.

## 5. Results so far (2026-10-03)

Both models were trained for 20,000 steps at batch size 32 on `yam_put_pot` (20 episodes,
7,779 frames), one seed each, and evaluated on PutPotOnCooktop with `pot_000` / `cooktop_000`,
plain scene, 25 actions executed per prediction, 50 episodes.

| Final model | Full task succeeded | Pot lifted |
|---|---|---|
| ACT + M3 | 11 of 50 (22%) | 13 of 50 (26%) |
| Plain ACT baseline | 12 of 50 (24%) | 15 of 50 (30%) |

- M3 shows no measurable benefit on the plain scene.
- The rates are slightly optimistic: the run used ten parallel environments and stopped at the
  first 50 finished episodes, and successes finish sooner than timeouts.
- Five-episode runs are too noisy to rank models: the same checkpoint scored 4 of 5 and then
  0 of 5.
- Not yet tested: randomized or cluttered scenes
  (`--enable_domain_randomization --use_unseen_materials`), which is where the paper claims M3
  helps.
- The dataset is small (two demonstrations per pot and cooktop pair); more data is the most
  likely way to raise the success rate for either model.
