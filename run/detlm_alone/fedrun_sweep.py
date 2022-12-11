import os
import sys
import time
import argparse
from loguru import logger
from multiprocessing import Pool

from run_dir import run_dir

from task_config import *
# from globalhost import machine_ip

import torch


def run_process(proc):
    os.system(proc)


def add_args(parser):
    parser.add_argument("--tasks", type=str, default="sst_2,agnews")
    parser.add_argument("--model_name", type=str, default="ditilbert")
    parser.add_argument("--machine_name", type=str, default="v100")
    parser.add_argument("--client_num", type=int, default=100)
    parser.add_argument("--valid_gpu", type=str, default="0,1")
    parser.add_argument("--vocabulary_type", type=str, default="all")
    parser.add_argument("--niid", type=str, default="-1")
    parser.add_argument("--test_mode", type=str, default="after")
    parser.add_argument("--alpha", type=float, default="1.0")
    return parser


parser = argparse.ArgumentParser()
parser = add_args(parser)
args = parser.parse_args()

# "sst_2"
tasks = ["20news", "agnews", "20news"] \
    if args.tasks == "-1" else args.tasks.split(",")
machine_name = args.machine_name
client_num = args.client_num
valid_gpu = args.valid_gpu

run_name = "run/detlm_alone/main"
run_dir = "/workspace/code/fedvocab/"
logger.debug(f"run_name: {run_name}")

valid_gpu = ",".join([str(i) for i in range(torch.cuda.device_count())])  \
    if valid_gpu == "-1" else valid_gpu
valid_gpu = valid_gpu.split(",")
n_gpu = len(valid_gpu)
client_device_dict = {i: valid_gpu[i] for i in range(n_gpu)}
logger.warning(f"valid_gpu: {valid_gpu}")

if args.niid == "-1":
    args.niid = "False"
else:
    args.niid = "True"

model_dict = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-uncased",
    "albert": "albert-base-v2"
}

seed = "42"
epoch = "1"
lr = "5e-5"
model_type = args.model_name
task_config_dir = tlm_task_config_dir
cmds = []
gpu_index = 0
for task in tasks:
    device_index = gpu_index % n_gpu
    task_config_dir[task][model_type]["epochs"] = str(epoch)
    task_config_dir[task][model_type]["lr"] = str(lr)
    round = task_config_dir[task][model_type]["comm_round"]
    logger.warning(
        f"run {task}_model={model_type}_"
        f"lr={lr}_epoch={epoch}_seed={seed}_round={round}_"
        f"niid={args.niid}_client_num_in_total={client_num}_alpha={args.alpha} "
        f"test_mode={args.test_mode} "
        f"on device: {client_device_dict[device_index]}")
    cmd = f'CUDA_VISIBLE_DEVICES={client_device_dict[device_index]} python3 {run_dir}/{run_name}.py '
    options = ["--dataset", task,
               "--model_type", model_type,
               "--model_name", model_dict[model_type],
               "--do_lower_case", "True",
               "--machine_name", machine_name,
               "--vocabulary_type", args.vocabulary_type,
               "--client_num_in_total", str(client_num),
               "--wandb_enable", "False",
               "--niid", args.niid,
               "--seed", str(seed),
               "--gpu", "0",
               "--client_num_in_total", str(args.client_num),
               "--partition_method", f"niid_label_clients={client_num}_alpha={args.alpha}",
               "--alpha", str(args.alpha),
               "--do_train", "True",
               "--test_mode", args.test_mode]
    for key, value in task_config_dir[task][model_type].items():
        options.extend(["--" + key, value])
    cmd += " ".join(options)
    cmds.append(cmd)
    gpu_index += 1

run_process("sleep 3s")
logger.warning(f"run {len(cmds)} tasks")

pool = Pool(processes=n_gpu)
pool.map(run_process, cmds)
