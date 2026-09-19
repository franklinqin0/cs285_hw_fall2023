# HW3 local runs

The existing `cs285_hw3` Conda environment runs the Pendulum bootstrapping check
on this Mac. Run from the `hw3` directory:

```bash
conda activate cs285_hw3
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python cs285/scripts/run_hw3_sac.py \
  -cfg experiments/sac/sanity_pendulum.yaml --no_gpu
```

Use `--no_gpu` with this environment: its PyTorch 1.13.1 build crashes in the
Apple MPS backend during action sampling. CPU execution has been verified.
NumPy 1.26.4 and OpenCV 4.11.0.86 are installed; the previous NumPy 2.x version
was incompatible with PyTorch's NumPy conversion. These versions are also
recorded in `env_mac.yml`.

`sanity_pendulum.yaml` runs all 300,000 environment steps with `train_actor:
false`, matching the homework's initial bootstrapping check. The critic and
target critic train while the actor stays fixed. Q-values should approach a
finite negative scale; a high evaluation return is not expected. The
configuration retains its entropy bonus in the critic targets. Actor-gradient
and double/min-Q exercises remain unfinished. To work on actor training later,
implement the actor-loss TODOs and set `train_actor: true`.

Each run writes TensorBoard events and a final `checkpoint.pt` under `data/`.
The checkpoint contains `agent_state_dict`, the final zero-based `step`, and
the random `seed`. View the metrics with:

```bash
tensorboard --logdir data
```

Run the focused bootstrapping tests from `hw3`:

```bash
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

## Humanoid SAC on CUDA

From `hw3`, with a CUDA-enabled PyTorch environment:

```bash
python cs285/scripts/run_hw3_sac.py \
  -cfg experiments/sac/humanoid.yaml \
  --eval_interval 20000 --num_eval_trajectories 5 \
  --no_distribution_validation
```

The runner defaults to one PyTorch CPU thread (`--torch_num_threads 1`).
Training computes scalar metrics only at `--log_interval` steps. Actor updates
skip unused critic parameter gradients while preserving gradients through the
sampled action, and target networks use in-place interpolation. These changes
retain the network sizes, batch size, and number of training updates.

`--no_distribution_validation` optionally skips PyTorch distribution argument
checks and their GPU synchronization; omit it when debugging invalid values.
The evaluation options above reduce scheduled evaluation episodes eightfold
relative to the defaults, making the evaluation curve less frequent and noisier.
Evaluation and policy sampling share the Torch RNG, so changing evaluation
frequency changes the exact training trajectory even with the same seed.

A short H800 benchmark on 2026-09-13 (one CPU thread, Humanoid 3x256 networks,
batch 256, two critics, 30 warmup updates and 1,500 measured steps) measured
about 129 steps/s before optimization, 141 steps/s after, and 149 steps/s with
distribution validation disabled. This includes environment steps and replay
sampling but excludes evaluation, video, and logging, and uses a small replay
buffer. It is a throughput comparison, not a full 5M-step runtime prediction.

## Stabilizing target values

Bootstrapped Q-learning overestimates values because the target maximizes over
noisy critic estimates. `target_critic_backup_type` in
`cs285/agents/soft_actor_critic.py` (`q_backup_strategy`) selects how the target
value is formed from the ensemble of target critics:

| Strategy | `target_critic_backup_type` | Critics | Hopper config |
|---|---|---:|---|
| Single-Q | (default, single critic) | 1 | `experiments/sac/hopper.yaml` |
| Double-Q | `doubleq` | 2 | `experiments/sac/hopper_doubleq.yaml` |
| Clipped double-Q | `min` | 2 | `experiments/sac/hopper_clipq.yaml` |
| REDQ | `redq` | 10 | `experiments/sac/hopper_redq.yaml` |

- **Single-Q** keeps the per-critic targets as-is; with one critic this is plain
  TD bootstrapping and overestimates the most.
- **Double-Q** rolls the target predictions by one (`torch.roll(next_qs,
  shifts=1, dims=0)`), so each critic bootstraps from the *other* critic's
  target rather than its own, decorrelating the max from the update.
- **Clipped double-Q** (`min`) takes the elementwise minimum across critics,
  the most conservative two-critic target.
- **REDQ** keeps ten critics and, per transition, independently samples two
  distinct target critics and takes their minimum (shared by all critics); the
  actor still uses the ensemble mean.

Run the three-way Hopper comparison (one process per GPU), recording one eval
video per evaluation with headless EGL:

```bash
for cfg in hopper hopper_doubleq hopper_clipq; do
  MUJOCO_GL=egl python cs285/scripts/run_hw3_sac.py \
    -cfg experiments/sac/$cfg.yaml -nvid 1
done
```

Plot `eval_return` and `q_values` together in TensorBoard. The expected story is
that `q_values` increase from clipped double-Q < double-Q < single-Q (less to
more overestimation), while the more conservative targets are at least as stable
in `eval_return`. A single-seed 100,000-step snapshot on 2026-09-13 showed this
ordering — approximate max `q_values` 182 (clip) < 203 (double) < 237 (single),
with clipped double-Q giving the best eval return — but a single seed is noisy,
so treat the ordering, not the exact numbers, as the result.

Pick the most stable backup type (clipped double-Q here) for the longer Humanoid
run; `experiments/sac/humanoid.yaml` already sets `target_critic_backup_type:
min` with two critics. The REDQ configs are covered in the next section.

## CUDA Graph and independent GPU experiments

Add `--cuda_graph` to capture the entire SAC update, including Adam and soft
target updates. This requires CUDA, fixed batch shapes, Adam, soft targets,
and constant learning rates with factor 1. Distribution validation is disabled
in this mode. Capture restores weights, Adam moments/counters, and RNG state
after warmup; each replay uses the incoming batch and fresh random numbers.
The regular eager path remains available by omitting the flag.

On the same H800, a 1,500-step Humanoid short benchmark reached approximately
364 steps/s with CUDA Graph, versus the earlier 149 steps/s eager baseline.
Capture took about 0.34 seconds. This excludes evaluation, logging, and a full
replay buffer. Reproduce a current eager/graph comparison from `hw3` with:

```bash
python cs285/scripts/benchmark_sac.py -cfg experiments/sac/humanoid.yaml
python cs285/scripts/benchmark_sac.py -cfg experiments/sac/humanoid.yaml --cuda_graph
```

The installed PyTorch 2.3.1 compiler could not fully compile the tanh
distribution's `log_prob` path (a weak-reference operation in its inverse
transform). CUDA Graph avoids that compiler limitation. Detached sampling
uses `rsample` inside `no_grad`, avoiding the synchronizing standard-deviation
check in CUDA `Normal.sample` while retaining detached actions for REINFORCE.

`hopper_redq.yaml`, `humanoid_redq.yaml`, and `humanoid_redq_500k.yaml` implement
the homework's ensembled clipped double-Q extension: ten critics, two distinct
target critics chosen independently for each transition, their minimum shared
by all critics. The actor still uses the ensemble mean. These configs keep
one critic update per environment transition; they do not implement the
higher update-to-data ratio often used in the original REDQ experiments.

Launch one independent experiment per physical GPU (no gradient sharing):

```bash
python cs285/scripts/launch_sac_multi_gpu.py --gpus 0 1 2 3 4 5 --configs \
  experiments/sac/hopper.yaml experiments/sac/hopper_doubleq.yaml \
  experiments/sac/hopper_clipq.yaml experiments/sac/hopper_redq.yaml \
  experiments/sac/humanoid_500k.yaml experiments/sac/humanoid_redq_500k.yaml
```

The launcher defaults to CUDA Graph, seed 1, evaluation every 5,000 steps with
ten episodes, and no videos. `-nvid 1` records one full trajectory at every
evaluation; `-nvid 4` records four each time. Each process has its own CUDA
visibility; when videos are enabled, the launcher defaults to headless EGL
rendering and maps numeric physical GPU IDs to `MUJOCO_EGL_DEVICE_ID`.
The rendering environment is recorded in the manifest. Each process has its own
config snapshot and stdout log. The suite `manifest.json` records
commands, physical GPU IDs, and PIDs; it is a launch record, not a live status
file. TensorBoard logs also contain `run_config.json` with arguments and the
PyTorch version. Checkpoints are saved atomically every 50,000 steps (adjust
`--checkpoint_interval`) and at completion. They contain network weights and
step/seed metadata for inspection; replay and optimizer state are not saved,
so they are not exact training-resume checkpoints.

If the NVIDIA EGL driver stalls in `mjr_readPixels`, prefix the launch command
with `MUJOCO_GL=osmesa LP_NUM_THREADS=1` to render videos on the CPU while
retaining CUDA Graph training on the selected GPU. This requires libOSMesa.
