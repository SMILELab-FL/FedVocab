import os
import torch
import random
import numpy as np
from training.utils.register import registry


def init_training_device(gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def set_args_path(args, machine_dict, desc="fednpm_alone"):
    # set some path
    logger = registry.get("logger")
    if not logger:
        raise ValueError("Not set logger")

    logger_file_path = os.path.join(
        machine_dict[args.machine_name]["output_logger_path"],
        f"{desc}_dataset={args.dataset}_seed={args.seed}.log")
    logger.add(open(logger_file_path, "w"))

    args.output_dir = os.path.join(machine_dict[args.machine_name]["output_dir"],
        f"{desc}_{args.dataset}/multi-seed/"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    cached_dir_name = f"{desc}_world_size={args.world_size}"
    args.cache_dir = os.path.join(
        machine_dict[args.machine_name]["cache_dir"],
        cached_dir_name)
    os.makedirs(args.cache_dir, exist_ok=True)

    args.save_dir = os.path.join(args.output_dir, f"model_niid={args.niid}")
    os.makedirs(args.save_dir, exist_ok=True)

    return args


def skip_parameters(args):
    eval_file = os.path.join(args.save_dir, f"{args.model}_sweep_{args.seed}_eval.results")
    patten = f"lr={args.lr}_epoch={args.epochs}"
    if not os.path.exists(eval_file):
        return False, patten
    with open(eval_file) as file:
        for line in file:
            if patten in line:
                return True, line
    return False, patten