import pandas as pd
from loguru import logger
from collections import Counter

import torch
from transformers import (
    DistilBertForMaskedLM, DistilBertTokenizer,
    GPT2Model, GPT2TokenizerFast, BertTokenizer
)

from utilis import get_batch_size


def inversion(dummy_embeds, original_embedds, inputs, i, device):
    seq_length = inputs["input_ids"][i].view(-1).size(0)
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.expand_as(inputs["input_ids"][i]).to(device)
    position_ids = position_ids.unsqueeze(0)
    position_embeds = embedder.position_embeddings(position_ids)
    word_embeddings = dummy_embeds[i] - position_embeds

    # reverse query the token IDs
    token_ids = []
    for idx in range(seq_length):
        temp = word_embeddings[:, idx:idx + 1].view(-1)
        distance = ((original_embedds.weight - temp) ** 2).sum(dim=1)
        token_id = distance.argmin().item()
        token_ids.append(token_id)

    inversion_text = tokenizer.decode(token_ids)
    return inversion_text


def format_token(tokenizer_gpt, sequence):
    token_list = []
    for token in tokenizer_gpt.tokenize(sequence):
        token = token.replace("Ġ", "")
        token_list.append(token)
    return token_list


def get_prf_score(sequences, inversion_seqs):
    inversion_results = {"orig": [], "inver": [], "p": [], "r": []}
    for i in range(len(sequences)):
        original = tokenizer.tokenize(sequences[i])
        inversion_seqs_tokens = format_token(tokenizer_gpt, inversion_seqs[i])
        inversion_tokens_cnt = Counter(inversion_seqs_tokens)

        inversion_tokens = []
        cnt = 0
        for token in original:
            if inversion_tokens_cnt[token]:
                cnt += 1
                inversion_tokens.append(token)
            else:
                inversion_tokens.append("<miss>")
        inversion_results["orig"].append(" ".join(original))
        inversion_results["inver"].append(" ".join(inversion_tokens))
        inversion_results["r"].append(cnt / len(original))
        inversion_results["p"].append(cnt / len(inversion_seqs_tokens))
    return inversion_results


def compute_pr(bs=1, device=7, mode="fedavg"):
    in_data = torch.load(f"/workspace/output/fednlp/attack/inversion_output/{mode}/"
                         f"distilbert_{mode}_agnews_bs={bs}.pth")
    device = f"cuda:{device}"
    embedder.to(device)
    gpt_origin_embedder.to(device)

    def get_sequences_list(all_sequences, batch_size):
        for i in range(0, len(all_sequences), batch_size):
            yield all_sequences[i:i + batch_size]

    sequences_list = get_sequences_list(all_sequences, bs)

    inversion_seqs = []
    sequences = in_data["original_seqs"]
    closer_embeds_list = in_data["closer_embeds_list"]
    for i, sequence_list in enumerate(sequences_list):
        inputs = tokenizer.batch_encode_plus(
            sequence_list, return_tensors="pt", padding=True, max_length=30)
        sample_num, _ = list(inputs["input_ids"].size())
        close_embedds = closer_embeds_list[i].to(device)
        for j in range(sample_num):
            inversion_seq = inversion(close_embedds, gpt_origin_embedder, inputs, j, device)
            inversion_seqs.append(inversion_seq)
    inversion_results = get_prf_score(sequences, inversion_seqs)
    pd.DataFrame(inversion_results).to_csv(f"/workspace/output/fednlp/attack/inversion_output/"
                                           f"{mode}_bs={bs}.tsv", sep="\t", index=False)
    avg_pre = sum(inversion_results["p"]) / len(inversion_results["p"])
    avg_rec = sum(inversion_results["r"]) / len(inversion_results["r"])
    avg_f1 = 2 * (avg_pre * avg_rec) / (avg_pre + avg_rec)
    print(f"bas:{bs}, pre:{avg_pre:.3f}, rec:{avg_rec:.3f}, f1:{avg_f1:.3f}")
    return avg_pre, avg_rec, avg_f1


model_gpt = GPT2Model.from_pretrained("/workspace/pretrain/nlp/gpt-2/")
tokenizer_gpt = GPT2TokenizerFast.from_pretrained("/workspace/pretrain/nlp/gpt-2/")
gpt_origin_embedder = model_gpt.wte

tokenizer = DistilBertTokenizer.from_pretrained("/workspace/pretrain/nlp/distilbert-base-uncased/")
model = DistilBertForMaskedLM.from_pretrained("/workspace/pretrain/nlp/distilbert-base-uncased/")
embedder = model.distilbert.embeddings

path = "/workspace/output/fednlp/attack/agnews_raw.text"
all_sequences = get_batch_size(path, None, 30, 128)

for bs in [1, 2, 4, 8, 16, 32, 64]:
    compute_pr(bs, 0, "fedde")
