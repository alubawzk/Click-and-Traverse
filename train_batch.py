import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime

python_executable = sys.executable

tasks = [
    # cuda_device, task, exp_name, restore_name, ground, lateral, overhead, obs_path,
    # term_collision_threshold, num_timesteps
    (0, "G1Cat", "debug_", "none", 0.0, 0.0, 0.0, "empty", 0.0, None),

]

# exp_name: debug mode if 'debug' in experiment name
# restore_name: you may need to restore the model from a checkpoint (e.g. our released model) if you want to curriculum learning. 'none' means training from scratch
# ground, lateral, overhead: reward coefficients for ground, lateral, and overhead obstacles; you can set all to 1. for convenience. You can also set them according to the obstacle configuration for more fine-grained control
# obs_path: path of the obstacle configuration, e.g., 'data/assets/TypiObs/bar0', 'data/assets/TypiObs/ceil1'. HumanoidPF 数据和障碍物数据是一一对应的，所以这里的 obs_path 会对应到 HumanoidPF 数据集中的同名数据
# term_collision_threshold: the threshold of collision distance to terminate the episode
# num_timesteps: optional total training timesteps for this task. Use None to
# fall back to train_ppo.py defaults, or override globally with
# `python train_batch.py --num-timesteps ...`

processes = []

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=None,
        help=(
            "Override total training timesteps for all tasks. If omitted, each "
            "task entry uses its own num_timesteps value or the default in train_ppo.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="./output_logs",
        help="Directory used to store stdout/stderr logs.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    process_cmd_map = {}
    for (
        cuda_device,
        task,
        exp_name,
        restore_name,
        ground,
        lateral,
        overhead,
        obs_path,
        term_collision_threshold,
        task_num_timesteps,
    ) in tasks:
        num_timesteps = (
            args.num_timesteps if args.num_timesteps is not None else task_num_timesteps
        )
        cmd = [
            python_executable,
            "-m",
            "train_ppo",
            "--task",
            task,
            "--restore_name",
            restore_name,
            "--exp_name",
            exp_name,
            "--ground",
            str(ground),
            "--lateral",
            str(lateral),
            "--overhead",
            str(overhead),
            "--term_collision_threshold",
            str(term_collision_threshold),
            "--obs_path",
            obs_path,
        ]
        if num_timesteps is not None:
            cmd.extend(["--num-timesteps", str(num_timesteps)])

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        stdout_file = os.path.join(output_dir, f"{timestamp}_{cuda_device}_stdout.log")
        stderr_file = os.path.join(output_dir, f"{timestamp}_{cuda_device}_stderr.log")
        cmd_display = shlex.join(["CUDA_VISIBLE_DEVICES=" + str(cuda_device), *cmd])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        with open(stdout_file, "w") as out_file, open(stderr_file, "w") as err_file:
            print(f"Executing: {cmd_display}")
            out_file.write(f"{cmd_display}\n")
            err_file.write(f"{cmd_display}\n")
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=out_file,
                stderr=err_file,
            )
            processes.append(process)
            process_cmd_map[process] = cmd_display
    while processes:
        for process in processes:
            retcode = process.poll()
            if retcode is not None:
                if retcode != 0:
                    cmd = process_cmd_map[process]
                    print(f"\033[91mReturn code {retcode}.\nCommand: {cmd}\033[0m")
                processes.remove(process)

    print("All tasks completed.")
