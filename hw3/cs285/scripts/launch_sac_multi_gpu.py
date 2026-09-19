"""Launch independent SAC experiments, one process per GPU, with a run manifest."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--gpus", nargs="+", required=True, help="Physical GPU IDs (one per job)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)
    parser.add_argument("--checkpoint_interval", type=int, default=50000)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) != len(args.configs) or len(set(args.gpus)) != len(args.gpus):
        parser.error("Provide exactly one distinct GPU for each configuration")
    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = args.output_dir or root / "data" / ("sac_suite_" + stamp)
    directory = directory.resolve()
    # Validate all configs before starting any processes.
    configs = []
    for filename in args.configs:
        path = Path(filename).resolve()
        with path.open() as source:
            config = yaml.safe_load(source)
        if config.get("base_config") != "sac":
            parser.error(f"Not a SAC configuration: {path}")
        if not args.eager and not config.get("use_soft_target_update", False):
            parser.error(f"CUDA graph mode requires soft targets: {path}")
        configs.append((path, config))
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {"created_utc": stamp, "jobs": []}
    manifest_path = directory / "manifest.json"
    for index, ((source, config), gpu) in enumerate(zip(configs, args.gpus)):
        label = f"{index}_{source.stem}_seed{args.seed}"
        config["exp_name"] = config.get("exp_name", source.stem) + f"_{'eager' if args.eager else 'cudagraph'}_s{args.seed}_{stamp}"
        snapshot = directory / (label + ".yaml")
        snapshot.write_text(yaml.safe_dump(config, sort_keys=False))
        log_path = directory / (label + ".log")
        command = [sys.executable, "-u", str(root / "cs285/scripts/run_hw3_sac.py"),
                   "-cfg", str(snapshot), "--seed", str(args.seed), "--which_gpu", "0",
                   "--torch_num_threads", "1", "--eval_interval", str(args.eval_interval),
                   "--num_eval_trajectories", str(args.num_eval_trajectories),
                   "--num_render_trajectories", str(args.num_render_trajectories),
                   "--checkpoint_interval", str(args.checkpoint_interval)]
        if not args.eager:
            command.append("--cuda_graph")
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        if args.num_render_trajectories > 0:
            # Headless MuJoCo rendering; EGL indexes physical rather than CUDA-visible GPUs.
            env.setdefault("MUJOCO_GL", "egl")
            if env["MUJOCO_GL"] == "egl" and gpu.isdecimal():
                env["MUJOCO_EGL_DEVICE_ID"] = gpu
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        job = {"config": str(source), "snapshot": str(snapshot), "gpu": gpu,
               "stdout": str(log_path), "command": command, "pid": None}
        job["render_env"] = {k: env[k] for k in ("MUJOCO_GL", "MUJOCO_EGL_DEVICE_ID") if k in env}
        if not args.dry_run:
            with log_path.open("w") as output:
                process = subprocess.Popen(command, cwd=root, env=env, stdout=output,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                           start_new_session=True)
            job["pid"] = process.pid
        manifest["jobs"].append(job)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(job), flush=True)
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
