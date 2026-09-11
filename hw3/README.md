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
