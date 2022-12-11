import os
import random
import numpy as np

from globalhost import machine_dict
from training.utils.register import registry
from model.transformer.model_args import ClassificationArgs
from data_manager.data_attributes import tc_data_attributes
from run.detlm_alone.misc import embedding_weights_load, embedding_pattern
from run.detlm_alone.demodel import (
    DistilBertForSequenceClassificationWithoutEmbedding,
    BertForSequenceClassificationWithoutEmbedding,
)

import torch
from transformers import (
    BertConfig,
    BertTokenizer,
    DistilBertConfig,
    DistilBertTokenizer,
)

from transformers.models.distilbert.modeling_distilbert import Embeddings
from transformers.models.bert.modeling_bert import BertEmbeddings


MODEL_CLASSES = {
    "classification": {
        "bert": (BertConfig, BertForSequenceClassificationWithoutEmbedding, BertTokenizer),
        "distilbert": (DistilBertConfig, DistilBertForSequenceClassificationWithoutEmbedding, DistilBertTokenizer),
    },
}

Embeddings_CLASS = {
    "bert": BertEmbeddings,
    "distilbert": Embeddings,
}


def add_model_args(args):
    model_args = ClassificationArgs()
    model_args.model_name = os.path.join(machine_dict[args.machine_name]["pretrained_model_path"],
                                         args.model_name)
    cached_dir_name = args.model_name + f"-world_size={args.world_size}"
    model_args.cache_dir = os.path.join(machine_dict[args.machine_name]["cache_dir"],
                                        cached_dir_name)
    os.makedirs(model_args.cache_dir, exist_ok=True)

    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    num_labels = tc_data_attributes[args.dataset]
    model_args.num_labels = num_labels
    model_args.update_from_dict({"fl_algorithm": args.fl_algorithm,
                                 "freeze_layers": args.freeze_layers,
                                 "epochs": args.epochs,
                                 "learning_rate": args.lr,
                                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                 "do_lower_case": args.do_lower_case,
                                 "manual_seed": args.seed,
                                 "local_add": args.local_add,
                                 # for ignoring the cache features.
                                 "reprocess_input_data": args.reprocess_input_data,
                                 "overwrite_output_dir": True,
                                 "max_seq_length": args.max_seq_length,
                                 "train_batch_size": args.train_batch_size,
                                 "eval_batch_size": args.eval_batch_size,
                                 "evaluate_during_training": False,  # Disabled for FedAvg.
                                 "evaluate_during_training_steps": args.evaluate_during_training_steps,
                                 "fp16": args.fp16,
                                 "data_file_path": args.data_file_path,
                                 "partition_file_path": args.partition_file_path,
                                 "partition_method": args.partition_method,
                                 "random_p": np.array(args.random_p),
                                 "dataset": args.dataset,
                                 "output_dir": args.output_dir,
                                 "fedprox_mu": args.fedprox_mu,
                                 "randomEmbedder": False,
                                 })
    model_args.config["num_labels"] = num_labels
    model_args.do_train = args.do_train
    registry.register("randomEmbedder", model_args.randomEmbedder)

    return model_args


def create_model(args, client_id_list, formulation="classification"):
    # create model, tokenizer, and model config (HuggingFace style)
    logger = registry.get("logger")
    logger.debug(f"create model ...")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][args.model_type]

    config = config_class.from_pretrained(args.model_name, **args.config)
    model = model_class.from_pretrained(args.model_name, config=config)

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name, do_lower_case=args.do_lower_case
    )
    if args.randomEmbedder:
        config_other_class, _, tokenizer_other_class = \
            MODEL_CLASSES[formulation]["bert"]
        # TODO hard code
        other_model_path = "/workspace/pretrain/nlp/bert-base-uncased//"
        config_other = config_other_class.from_pretrained(
            other_model_path, **args.config
        )
        tokenizer_other = tokenizer_other_class.from_pretrained(
            other_model_path, do_lower_case=args.do_lower_case
        )
        registry.register("other_tokenizer", tokenizer_other)
        other_weights_path = os.path.join(other_model_path, "pytorch_model.bin")
        other_all_weights = torch.load(other_weights_path)
    else:
        other_all_weights = None

    weights_path = os.path.join(args.model_name, "pytorch_model.bin")
    all_weights = torch.load(weights_path)

    deEmbeding = {}
    other_embedder_ids = []

    for client_id in client_id_list:

        if args.randomEmbedder:
            # embedder_type = random.sample([args.model_type, "bert"], 1)[0]
            embedder_type = np.random.choice([args.model_type, "bert"], p=args.random_p.ravel())
        else:
            embedder_type = args.model_type

        if embedder_type == args.model_type:
            embedder = Embeddings_CLASS[args.model_type](config)
            if args.do_train:
                deEmbeding[client_id] = embedding_weights_load(
                    args, embedder,
                    pattern=embedding_pattern[args.model_type],
                    all_weights=all_weights
                )
            else:
                deEmbeding[client_id] = embedder
        else:
            embedder = Embeddings_CLASS["bert"](config_other)
            deEmbeding[client_id] = embedding_weights_load(
                args, embedder,
                pattern=embedding_pattern["bert"],
                all_weights=other_all_weights
            )
            other_embedder_ids.append(client_id)

    registry.register("other_embedder_ids", other_embedder_ids)
    logger.critical(f"other_embedder_ids: {len(other_embedder_ids)}")
    return config, deEmbeding, model, tokenizer
