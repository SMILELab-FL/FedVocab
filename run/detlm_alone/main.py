import os
import time
import argparse
import json
import random
from copy import deepcopy

import sys
import torch
from torch.nn.parameter import Parameter

from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool

from run_dir import run_dir

import wandb
from loguru import logger
from globalhost import machine_dict
from training.utils.register import registry
from data_manager.data_attributes import tc_data_attributes

from run.detlm_alone.config import add_args
from run.detlm_alone.misc import (init_training_device, setup_seed, DeTLMTranier,
                                  get_parameter_number, report, get_words_index,
                                  random_private_words, skip_parameters, random_all_words,
                                  save_every_model_parameters, get_best_model,
                                  load_every_model_parameters, test_report)
from run.detlm_alone.data_utils import load_and_processing_data
from run.detlm_alone.model_build_utils import add_model_args, create_model
from run.detlm_alone.de_fedutils import FedDeTLMTrainer, evaluation


if __name__ == "__main__":
    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    args.fl_algorithm = "FedDeA"
    args.alter_update = True
    args.local_add = False

    logger.info(f"run script in {run_dir}")
    logger.debug(f"run args: {args}")
    logger.critical(f"Method: {args.fl_algorithm}")
    logger.critical(f"Test Mode: {args.test_mode}")

    registry.register("logger", logger)
    # set some path
    logger_file_path = os.path.join(
        machine_dict[args.machine_name]["output_logger_path"],
        f"detlm_alone_dataset={args.dataset}_seed={args.seed}.log")
    logger.add(open(logger_file_path, "w"))

    args.output_dir = os.path.join(
        machine_dict[args.machine_name]["output_dir"],
        f"{args.dataset}/{args.fl_algorithm.lower()}/tlm/"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    logger.debug(f"output dir in {args.output_dir}")

    args.save_dir = os.path.join(args.output_dir, f"{args.test_mode}")
    os.makedirs(args.save_dir, exist_ok=True)

    # Set the random seed and GPU
    setup_seed(args.seed)
    args.gpu = init_training_device(args.gpu)
    logger.debug(f"running on: {args.gpu}")

    clients_num = args.client_num_in_total
    client_id_list = [
        i for i in range(clients_num)
    ]
    args.client_id_list = client_id_list

    # create model.
    model_args = add_model_args(args)
    config, deEmbeding, model, tokenizer = create_model(
        model_args,
        client_id_list=args.client_id_list,
        formulation="classification"
    )
    local_model = deepcopy(model)
    global_p, local_p = get_parameter_number(model)['Total'], get_parameter_number(deEmbeding[0])['Total']
    model_p = global_p + local_p
    logger.debug(f"all model size: {model_p:.1f}M, "
                 f"local model size: {local_p:.1f}M, "
                 f"global model size: {global_p:.1f}M")

    # create dataset
    dataset = load_and_processing_data(args, model_args, tokenizer, client_id_list)
    train_data_num, train_data_global, \
    test_data_num, test_data_global, dev_data_num, dev_data_global, \
    train_data_local_num_dict, train_data_local_dict, dev_data_local_num_dict, dev_data_local_dict, \
    test_data_local_num_dict, test_data_local_dict = dataset
    logger.debug(f"train_data_num: {train_data_num}, "
                 f"test_data_num: {test_data_num}, "
                 f"dev_data_num: {dev_data_num}")

    args.personalized = True if args.fl_algorithm == "FedRecon" else False
    if not args.share_private_words_index_path:
        logger.warning(f"generating share-private words index to {model_args.model_name}")
        get_words_index(model_args)
    else:
        logger.warning(f"loading share-private words index to {args.share_private_words_index_path}")
        with open(args.share_private_words_index_path) as file:
            share_private_words_index = json.load(file)
            registry.register("share_words_index", share_private_words_index["share_words_index"])
            registry.register("private_words_index", share_private_words_index["private_words_index"])
    deEmbeding = random_all_words(deEmbeding, ratio=args.random_ratio)
    share_embedding_weights = None

    # federated setting
    if args.ci:
        num_per_round = int(args.client_num_in_total * args.ci)
    else:
        num_per_round = args.client_num_per_in_round
        args.ci = num_per_round / args.client_num_in_total
    logger.critical(f"CI: {args.ci}, Client num per in round: {num_per_round}")

    total_client_num = args.client_num_in_total
    aggregator = Aggregators.fedavg_aggregate

    args.global_eval_batch_size = args.eval_batch_size * 4
    args.local_eval_batch_size = args.eval_batch_size

    Trainer = DeTLMTranier(
        args=args, device=args.gpu
    )
    args.num_labels = Trainer.num_labels

    trainer = FedDeTLMTrainer(
        args=args,
        model=local_model,
        train_dataset=train_data_local_dict,
        test_dataset=None,
        data_slices=client_id_list,
        aggregator=aggregator,
        logger=logger,
        embedder=deEmbeding,
        trainer=Trainer
    )
    # args.do_train = False
    if args.do_train:
        logger.debug("The procedure is training")
        transmit_p = round(global_p, 1)
        logger.warning(f"{args.fl_algorithm} transmit {transmit_p:.3f}M")

        results = {
            "test_mode": args.test_mode, "logs": {},
            "transmit_p": transmit_p,
        }

        # train procedure
        to_select = [i for i in range(total_client_num)]

        for rd in range(args.comm_round):
            model_parameters = SerializationTool.serialize_model(model)
            selection = random.sample(to_select, num_per_round)
            logger.debug(f"selection client({num_per_round}): {selection}")

            aggregated_parameters, aggregated_share_parameters = trainer.train(
                model_parameters=model_parameters,
                id_list=selection,
                aggregate=True,
                share_embedding_weights=share_embedding_weights,
            )
            SerializationTool.deserialize_model(model, aggregated_parameters)

            share_embedding_weights = aggregated_share_parameters
            if share_embedding_weights is not None:
                logger.warning("update share_embedding_weights")
                shape = trainer.embedder[0].word_embeddings.weight.shape
                temp = torch.randn((shape[0], shape[1]))
                for idx in selection:
                    temp[trainer.share_words_index] = share_embedding_weights
                    temp[trainer.private_words_index] = trainer.embedder[idx].word_embeddings.weight[
                        trainer.private_words_index]
                    trainer.embedder[idx].word_embeddings.weight = Parameter(temp)

            if args.test_mode == "after":
                for idx in selection:
                    global_acc, _ = trainer.test(
                        model, test_data_global,
                        args.global_eval_batch_size, idx
                    )
                    if global_acc > trainer.global_acc[idx]:
                        trainer.global_acc[idx] = global_acc
                    logger.debug(f"Client: {idx}, GlobalAcc: {global_acc:.3f}, "
                                 f"BestGlobalAcc: {trainer.global_acc[idx]:.3f}")

                    local_acc, _ = trainer.test(
                        model, dev_data_local_dict[idx],
                        args.local_eval_batch_size, idx
                    )
                    if local_acc > trainer.local_acc[idx]:
                        trainer.local_acc[idx] = local_acc
                        trainer.every_model_parameters[idx] = \
                            [trainer.embedder[idx].state_dict(), trainer.model.state_dict()]
                    logger.debug(f"Client: {idx}, LocalAcc: {local_acc:.3f}, "
                                 f"BestLocalAcc: {trainer.local_acc[idx]:.3f}")

            global_acc = trainer.global_acc
            local_acc = trainer.local_acc
            global_avg_acc = sum(global_acc) / len(global_acc)
            local_avg_acc = sum(local_acc) / len(local_acc)

            results["logs"][f"round_{rd}"] = {
                "global_avg_acc": global_avg_acc,
                "local_avg_acc": local_avg_acc}

            logger.warning(
                f"{args.dataset} and {args.model_type} using "
                f"{args.fl_algorithm} test with {args.test_mode}"
            )
            logger.critical(
                f"Round: {rd} "
                f"GlobalAvgAcc: {global_avg_acc:.3f} "
                f"LocalAvgAcc: {local_avg_acc:.3f}"
            )

        times = time.strftime("%Y%m%d%H%M", time.localtime())
        save_every_model_parameters(args, trainer.every_model_parameters, times)
        global_acc = trainer.global_acc
        local_acc = trainer.local_acc
        global_avg_acc = sum(global_acc) / len(global_acc)
        local_avg_acc = sum(local_acc) / len(local_acc)
        results["global_avg_acc"] = round(global_avg_acc, 3)
        results["local_avg_acc"] = round(local_avg_acc, 3)
        results["global_acc"] = global_acc
        results["local_acc"] = local_acc
        results["times"] = times
        results["args"] = args

        logger.critical(f"{args.fl_algorithm}-{args.dataset} is Done "
                        f"and {args.model_type} with {args.test_mode} test mode is "
                        f"GlobalAvgAcc: {global_avg_acc:.3f} "
                        f"LocalAvgAcc: {local_avg_acc:.3f}")
        report(args, results, times=times)
