import os
import random
from tqdm import tqdm
from loguru import logger
from collections import Counter

import torch
from torch.autograd import grad


def get_bert(model, model_type):
    if model_type == "distilbert":
        return model.distilbert
    else:
        return model.bert


def get_transformer(bert, model_type):
    if model_type == "distilbert":
        return bert.transformer.layer
    else:
        return bert.encoder.layer


def load_every_model_parameters(output_dir, fl_algorithm, times, deEmbeding, model):
    save_path = os.path.join(output_dir, f"{fl_algorithm}_{times}_models.pth")
    logger.info(f"Loading model from {save_path} ... ")
    all_client_model_parameters = torch.load(save_path)
    embedder_parameters, model_parameters = all_client_model_parameters[0]
    deEmbeding[0].load_state_dict(embedder_parameters)
    model.load_state_dict(model_parameters)
    return model, deEmbeding[0]


def get_batch_size(path, batch_size=None, seq_len=30, sample_num=128):
    target_seqs = []
    with open(path) as file:
        for line in file:
            tokens = line.strip().split()
            number = [char.isdigit() for char in tokens]
            number_cnt = Counter(number)
            if number_cnt[True] >= 3 and len(tokens) <= seq_len:
                target_seqs.append(" ".join(tokens))
    logger.info(f"target_seqs len: {len(target_seqs)}")
    if batch_size:
        return random.sample(target_seqs, batch_size)
    else:
        return sorted(target_seqs, key=lambda seq: len(seq))[0:sample_num]


def gradient_inversion(sequences, tokenizer, bert, embedder, device,
                       bs=1, max_length=30, thres=4000,
                       idx=None, mode=None, model_type="distilbert"):
    inputs = tokenizer.batch_encode_plus(
        sequences, return_tensors="pt",
        padding=True, max_length=max_length
    )

    # embedder = bert.embeddings
    embedder.to(device)
    inputs = inputs.to(device)
    input_embeded = embedder(inputs["input_ids"])
    logger.info(f"input_embeded size: {input_embeded.size()}")

    inputs_embeds = embedder(inputs["input_ids"])
    attention_mask = inputs["attention_mask"]
    output_attentions = bert.config.output_attentions

    trans = get_transformer(bert, model_type)
    trans.to(device)
    dlbrt_output = trans(
        inputs_embeds,
        # attention_mask,
        output_attentions=output_attentions,
    )
    hidden_states = dlbrt_output[0]

    # True clients grads
    dy_dx = grad(hidden_states.sum(), trans.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))
    trans.zero_grad()

    dummy_embeds = torch.randn(inputs_embeds.size(), device=device).requires_grad_(True)
    optimizer = torch.optim.Adam([dummy_embeds, ], lr=1e-2)
    closer_embeds = None
    min_loss = 1e11

    # for i in tqdm(range(300000), "mimic input embedding"):
    for i in range(300000):
        optimizer.zero_grad()
        embedder.zero_grad()
        dlbrt_output = trans(
            dummy_embeds,
            # attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = dlbrt_output[0]
        dummy_dy_dx = grad(hidden_states.sum(), trans.parameters(), create_graph=True)

        grad_diff = 0
        grad_count = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
            grad_count += gx.nelement()
        grad_diff.backward()
        if grad_diff.item() < min_loss:
            closer_embeds = dummy_embeds
            min_loss = grad_diff.item()
        if i > 0 and i % 1000 == 0:
            logger.info(f"mode:{mode}, bs: {bs}, idx: {idx}, step: {i}, loss: {grad_diff.item():.3f}, min loss: {min_loss:.3f}")
        optimizer.step()
        if min_loss <= thres * bs:
            logger.debug(f"{idx} done and current min_loss {min_loss} <= {thres} * {bs}")
            break
    return closer_embeds


def get_sequences_list(all_sequences, batch_size):
    for i in range(0, len(all_sequences), batch_size):
        yield all_sequences[i:i + batch_size]
