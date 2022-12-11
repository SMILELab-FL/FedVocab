import os
import sys
import argparse
from loguru import logger

import torch
from transformers import DistilBertConfig, BertConfig
from transformers import DistilBertTokenizer, BertTokenizer
from transformers.models.distilbert.modeling_distilbert import Embeddings
from transformers.models.bert.modeling_bert import BertEmbeddings

run_dir = "/".join(os.path.abspath(sys.argv[0]).split("/")[0:-3])
sys.path.append(run_dir)

from run.detlm_alone.demodel import DistilBertForSequenceClassificationWithoutEmbedding
from run.detlm_alone.demodel import BertForSequenceClassificationWithoutEmbedding
from run.attack.utilis import (
    load_every_model_parameters, get_batch_size,
    get_sequences_list, gradient_inversion, get_bert
)


MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassificationWithoutEmbedding, BertTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForSequenceClassificationWithoutEmbedding, DistilBertTokenizer),
}


EMBEDDER_CLASS = {
    "distilbert": Embeddings,
    "bert": BertEmbeddings
}


def add_args(parser):
    parser.add_argument('--mode', type=str, default='fedde', metavar='N',
        help="fl algorithm")
    parser.add_argument('--index', type=str, default='0', metavar='N',
        help="gpu index")
    parser.add_argument('--bs', type=int, default='0', metavar='N',
        help="attack batch size")
    parser.add_argument('--output_path', type=str,
        default='/workspace/output/fednlp/attack/inversion_output/', metavar='N',
        help="attack results saved dirs")
    parser.add_argument('--raw_path', type=str,
        default='/workspace/output/fednlp/attack/agnews_raw.text', metavar='N',
        help="attack dataset")
    parser.add_argument('--opp',  type=str,
        default="/workspace/pretrain/nlp/distilbert-base-uncased/", metavar='N',
        help="original pretrained tokenizer/config/model path")
    parser.add_argument('--omp', type=str,
        default="/workspace/pretrain/nlp/distilbert-base-uncased/", metavar='N',
        help="original pretrained model path")
    parser.add_argument('--model_type', type=str,
        default="distilbert", metavar='N',
        help="model type")
    parser.add_argument('--trained_model_path', type=str,
        default="/workspace/output/fednlp/agnews/feddea/tlm/after/", metavar='N',
        help="original pretrained model path")
    parser.add_argument('--times', type=str,
        default="202206062008", metavar='N',
        help="original pretrained model path")

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    # parse python script input parameters
    parser = argparse.ArgumentParser()
    args = add_args(parser)

    mode = args.mode
    index = args.index
    bs = args.bs

    logger.debug(f"Start Attack {mode} FL Algorithm Inversion")
    logger.info("Loading config, model, tokenizer...")

    output_path = os.path.join(args.output_path, mode)
    os.makedirs(output_path, exist_ok=True)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.opp)

    model_dir = args.trained_model_path
    config = config_class.from_pretrained(args.opp, num_labels=4)
    model = model_class.from_pretrained(
        args.opp,
        config=config
    )
    deEmbeding = {}
    embeddings = EMBEDDER_CLASS[args.model_type]
    embedder = embeddings(config)
    deEmbeding[0] = embedder
    model, embedder = load_every_model_parameters(model_dir, "FedDeA", args.times, deEmbeding, model)
    bert = get_bert(model, args.model_type)

    raw_path = args.raw_path
    all_sequences = get_batch_size(raw_path, None, 30)
    seqs_list = get_sequences_list(all_sequences, bs)

    if torch.cuda.is_available():
        device = f"cuda:{index}"
    else:
        device = "cpu"

    closer_embeds_list = []
    for i, seq_list in enumerate(seqs_list):
        closer_embeds = gradient_inversion(
            seq_list, tokenizer, bert, embedder, device,
            bs=bs, idx=i, mode=mode, model_type=args.model_type)
        closer_embeds_list.append(closer_embeds)

    save_opt = {
        "original_seqs": all_sequences,
        "closer_embeds_list": closer_embeds_list
    }
    save_file = os.path.join(output_path, f"{args.model_type}_{mode}_agnews_bs={bs}.pth")
    torch.save(save_opt, save_file)
