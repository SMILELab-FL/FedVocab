import os
import pickle
import time
import math
import copy
import json
import random
import numpy as np
import sklearn
from collections import OrderedDict
from sklearn.metrics import matthews_corrcoef, confusion_matrix

import torch
from torch.optim import *
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup

from torch.nn.parameter import Parameter
from training.utils.register import registry
from data_manager.data_attributes import tc_data_attributes


embedding_pattern = {
    "distilbert": "distilbert.embeddings.",
    "bert": "bert.embeddings.",
    "albert": "albert.embeddings."
}


class DeTLMTranier(object):
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.num_labels = tc_data_attributes[args.dataset]
        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []
        self.results = {}
        self.logger = registry.get("logger")
        self.best_accuracy = 0.0

    def alter_train_model(self, model, train_dl, embedder, idx):
        model.to(self.device)
        embedder.to(self.device)
        optimizer = AdamW(embedder.parameters(), lr=self.args.lr, eps=self.args.adam_epsilon)
        loss_fct = CrossEntropyLoss()

        epoch_loss, epoch_acc = [], []
        for epoch in range(0, self.args.epochs):
            model.train()
            embedder.train()
            batch_loss, batch_acc = [], []
            for batch_idx, batch in enumerate(train_dl):
                batch = tuple(t for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                x = batch[1].to(self.device)
                labels = batch[4].to(self.device)

                # (loss), logits, (hidden_states), (attentions)
                x = embedder(x)
                output = model(inputs_embeds=x)
                logits = output[0]

                logits = logits.view(-1, self.num_labels)
                labels = labels.view(-1)
                loss = loss_fct(logits, labels)
                num_corrects = torch.sum(torch.argmax(logits, 1).eq(labels))
                acc = num_corrects / x.size()[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(embedder.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    embedder.zero_grad()

                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    epoch_acc.append(sum(batch_acc) / len(batch_acc))

            if len(epoch_loss) > 0:
                self.logger.info(
                    f'Client id: {idx}, AlterUpdate Local Training Epoch: {epoch}, '
                    f'Training Loss: {sum(epoch_loss)/len(epoch_loss):.3f}, '
                    f'Training Accuracy: {sum(epoch_acc)/len(epoch_acc):.3f}'
                )
            else:
                self.logger.critical(
                    'Client id {} has {} epoch_loss'.format(idx, epoch, len(epoch_loss))
                )

    def train_model(self, model, train_dl, embedder, idx):
        model.to(self.device)
        embedder.to(self.device)

        # build optimizer and scheduler
        iteration_in_total = len(train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer, scheduler = self.build_optimizer(model, embedder, iteration_in_total)
        loss_fct = CrossEntropyLoss()

        # training result
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        # tr_loss = []
        if self.args.fl_algorithm == "FedDeP":
            self.logger.warning("using FedDep")
            global_model = copy.deepcopy(model)

        epoch_loss, epoch_acc = [], []
        for epoch in range(0, self.args.epochs):
            batch_loss, batch_acc = [], []
            for batch_idx, batch in enumerate(train_dl):
                model.train()
                embedder.train()

                batch = tuple(t for t in batch)
                # dataset = TensorDataset(all_guid, all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
                x = batch[1].to(self.device)
                labels = batch[4].to(self.device)

                # (loss), logits, (hidden_states), (attentions)
                x = embedder(x)
                output = model(inputs_embeds=x)
                logits = output[0]

                logits = logits.view(-1, self.num_labels)
                labels = labels.view(-1)
                loss = loss_fct(logits, labels)
                num_corrects = torch.sum(torch.argmax(logits, 1).eq(labels))
                acc = num_corrects / x.size()[0]

                if self.args.fl_algorithm == "FedDeP":
                    fed_prox_reg = 0.0
                    mu = self.args.fedprox_mu
                    for (p, g_p) in zip(model.parameters(), global_model.parameters()):
                        fed_prox_reg += ((mu / 2) * torch.norm((p - g_p.data)) ** 2)
                    loss += fed_prox_reg

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(embedder.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    embedder.zero_grad()
                    global_step += 1

                batch_loss.append(loss.item())
                batch_acc.append(acc.item())
                if len(batch_loss) > 0:
                    epoch_loss.append(sum(batch_loss) / len(batch_loss))
                    epoch_acc.append(sum(batch_acc) / len(batch_acc))

            if len(epoch_loss) > 0:
                self.logger.info(
                    f'Client id: {idx}, NormalUpdate Local Training Epoch: {epoch}, '
                    f'Training Loss: {sum(epoch_loss) / len(epoch_loss):.3f}, '
                    f'Training Accuracy: {sum(epoch_acc) / len(epoch_acc):.3f}'
                )
            else:
                self.logger.critical(
                    'Client id {} has {} epoch_loss'.format(idx, epoch, len(epoch_loss))
                )

        self.logger.info(f"Client id {idx} train procedure is finished")

    def build_optimizer(self, model, embedder, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        self.logger.warning("warmup steps = %d" % self.args.warmup_steps)

        self.freeze_model_parameters(model)

        update_params = [
            {"params": filter(lambda x: x.requires_grad, model.parameters())},
            {"params": filter(lambda x: x.requires_grad, embedder.parameters()), }
        ]

        if self.args.optimizer == "adamw":
            self.logger.info(f"{self.args.fl_algorithm} Using AdamW as Optimizer")
            optimizer = AdamW(update_params, lr=self.args.lr, eps=self.args.adam_epsilon)
        else:
            self.logger.info(f"{self.args.fl_algorithm} Using SGD as Optimizer")
            optimizer = SGD(update_params, lr=self.args.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=iteration_in_total
        )
        return optimizer, scheduler

    def freeze_model_parameters(self, model):
        modules = list()
        self.logger.info("freeze layers: %s" % str(self.freeze_layers))
        for layer_idx in self.freeze_layers:
            if layer_idx == "e":
                modules.append(model.distilbert.embeddings)
            else:
                modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        self.logger.info(get_parameter_number(model))


def compute_metrics(preds, labels, eval_examples=None):
    assert len(preds) == len(labels)

    extra_metrics = {}
    extra_metrics["acc"] = sklearn.metrics.accuracy_score(labels, preds)
    mismatched = labels != preds

    if eval_examples:
        wrong = [i for (i, v) in zip(eval_examples, mismatched) if v.any()]
    else:
        wrong = ["NA"]

    # mcc = matthews_corrcoef(labels, preds)
    #
    # tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    # return (
    #     {**{"mcc": mcc, "tp": tp, "tn": tn, "fp": fp, "fn": fn}, **extra_metrics},
    #     wrong,
    # )
    return (extra_metrics, wrong)


def evaluation(args, eval_batch_size, test_dl, model, embedder):
    logger = registry.get("logger", None)
    device = args.gpu
    num_labels = args.num_labels

    results = {}

    eval_loss = 0.0
    nb_eval_steps = 0
    n_batches = len(test_dl)
    test_sample_len = len(test_dl.dataset)
    preds = np.empty((test_sample_len, num_labels))

    out_label_ids = np.empty(test_sample_len)
    model.to(device)
    embedder.to(device)
    model.eval()
    embedder.eval()

    if logger:
        logger.info("test_sample_len = %d, n_batches = %d" % (test_sample_len, n_batches))

    for i, batch in enumerate(test_dl):
        with torch.no_grad():
            batch = tuple(t.to(device) for t in batch)

            x = batch[1]
            labels = batch[4]

            x = embedder(x)
            output = model(inputs_embeds=x)
            logits = output[0]

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            eval_loss += loss.item()

        nb_eval_steps += 1
        start_index = eval_batch_size * i

        end_index = start_index + eval_batch_size if i != (n_batches - 1) else test_sample_len
        # logging.info("batch index = %d, start_index = %d, end_index = %d" % (i, start_index, end_index))
        preds[start_index:end_index] = logits.detach().cpu().numpy()
        out_label_ids[start_index:end_index] = labels.detach().cpu().numpy()

    eval_loss = eval_loss / nb_eval_steps

    model_outputs = preds
    preds = np.argmax(preds, axis=1)

    result, wrong = compute_metrics(preds, out_label_ids, test_dl.examples)
    result["eval_loss"] = eval_loss
    results.update(result)

    # if result["acc"] > self.best_accuracy:
    #     self.best_accuracy = result["acc"]
    # self.logger.debug("best_accuracy = %f" % self.best_accuracy)
    #
    # self.results.update(result)
    # self.logger.critical(self.results)
    return result, model_outputs, wrong


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num / 1e6, 'Trainable': trainable_num / 1e6}


def post_complete_message(tc_args):
    pipe_path = "/tmp/fednlp_tc"
    if not os.path.exists(pipe_path):
        os.mkfifo(pipe_path)
    pipe_fd = os.open(pipe_path, os.O_WRONLY)

    with os.fdopen(pipe_fd, 'w') as pipe:
        pipe.write("training is finished! \n%s" % (str(tc_args)))


def setup_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def init_training_device(gpu):
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    return device


# def report(args, results, times):
#     save_path = os.path.join(args.output_dir, f"model_niid={args.niid}/")
#     os.makedirs(save_path, exist_ok=True)
#     file = os.path.join(save_path, f"{args.model_type}_sweep_{args.seed}_eval.results")
#
#     max_acc = sum(results['EachMaxAcc']) / len(results['EachMaxAcc'])
#     with open(file, "a+") as f:
#         line = f"time={times}_{args.dataset}_lr={args.lr}_epoch={args.epochs}_" \
#                f"round={args.comm_round}_" \
#                f"optimizer={args.optimizer}_niid={args.niid}_alpha={args.alpha}_share={args.personalized}_acc={max_acc}"
#         f.write(line + "\n")
#     file_json = os.path.join(save_path, f"{args.model_type}_sweep_{args.seed}_{times}_eval.json")
#     with open(file_json, "w") as file:
#         json.dump(results, file)

def report(args, results, times):
    file = os.path.join(args.save_dir, f"{args.model_type}_sweep_{args.seed}_eval.results")
    global_avg_acc = results["global_avg_acc"]
    local_avg_acc = results["local_avg_acc"]

    with open(file, "a+") as f:
        line = f"time={times}_{args.dataset}_lr={args.lr}_epoch={args.epochs}_" \
               f"optimizer={args.optimizer}_niid={args.niid}_alpha={args.alpha}_" \
               f"num={args.client_num_in_total}_ci={args.ci}_ratio={args.random_ratio}_" \
               f"globalacc={global_avg_acc}_localacc={local_avg_acc}"
        f.write(line+"\n")

    result_json = os.path.join(args.save_dir, f"{args.model_type}_sweep_{args.seed}_{times}_eval.pkl")
    with open(result_json, "wb") as f:
        pickle.dump(results, f)


def test_report(args, results, line):
    file_json = os.path.join(args.save_dir, f"{args.model_type}_sweep_{args.seed}_test.json")
    file_test = os.path.join(args.save_dir, f"{args.model_type}_sweep_{args.seed}_test.results")

    with open(file_json, "w") as file:
        json.dump(results, file)

    acc_list = list(results.values())
    report_acc = sum(acc_list) / len(acc_list)
    report_line = line + f"_test-acc={report_acc}"
    with open(file_test, "a+") as file:
        file.write(report_line + "\n")


def save_every_model_parameters(args, every_parameters, times):
    # os.makedirs(args.output_dir, exist_ok=True)
    file = os.path.join(args.save_dir, f"{args.fl_algorithm}_{times}_models.pth")
    torch.save(every_parameters, file)


def get_best_model(args):
    save_path = os.path.join(args.output_dir, f"model_niid={args.niid}/")
    file = os.path.join(save_path, f"{args.model_type}_sweep_{args.seed}_eval.results")
    max_acc = 0.0
    best_parameters = None
    with open(file) as f:
        for line in f:
            parameters_info = line.strip().split("_")
            for parameter in parameters_info:
                if parameter.startswith("acc"):
                    _, acc = parameter.split("=")
                    if float(acc) > max_acc:
                        max_acc = float(acc)
                        best_parameters = line.strip()
    logger = registry.get("logger")
    logger.warning(f"The best parameters is {best_parameters}")
    times = best_parameters.split("_")[0].split("=")[1]
    return times, best_parameters


def load_every_model_parameters(args, times, deEmbeding, model):
    save_path = os.path.join(args.output_dir, f"model_niid={args.niid}/{args.model_type}_{times}_models.pth")
    all_client_model = {}
    all_client_model_parameters = torch.load(save_path)
    for i in all_client_model_parameters:
        embedder_parameters, model_parameters = all_client_model_parameters[i]
        deEmbeding[i].load_state_dict(embedder_parameters)
        model.load_state_dict(model_parameters)
        all_client_model[i] = [deEmbeding[i], model]
    # del deEmbeding, model
    return all_client_model


def build_results(args):
    results = {}
    results["CurMaxAccAvg"] = 0.0
    results["MaxAccLocal"] = 0.0
    results["CurAccLog"] = []
    results["EachMaxAcc"] = [0.0] * args.client_num_in_total
    return results


def update_results(results, round_id, client_list, result_list):
    result = dict()
    result["client_list"] = client_list
    result["result_list"] = result_list
    results[f"round_{round_id}"] = result
    cur_acc_avg = sum(result_list) / len(result_list)
    max_acc = max(result_list)
    if max_acc > results["MaxAccLocal"]:
        results["MaxAccLocal"] = max_acc
    if cur_acc_avg > results["CurMaxAccAvg"]:
        results["CurMaxAccAvg"] = max_acc
    results["CurAccLog"].append(cur_acc_avg)
    for i, client_id in enumerate(client_list):
        if result_list[i] > results["EachMaxAcc"][client_id]:
            results["EachMaxAcc"][client_id] = result_list[i]
    return results


def embedding_weights_load(args, embedder, pattern="distilbert.embeddings.", all_weights=None):
    embedding_weight = OrderedDict()
    weights_path = os.path.join(args.model_name, "pytorch_model.bin")
    # all_weights = torch.load(weights_path, map_location=torch.device('cpu'))
    if all_weights is None:
        all_weights = torch.load(weights_path)
    for key, value in all_weights.items():
        if "embedding" in key:
            new_key = key.replace(pattern, "")
            if pattern.startswith("bert") and new_key == "LayerNorm.gamma":
                new_key = "LayerNorm.weight"
            elif pattern.startswith("bert") and new_key == "LayerNorm.beta":
                new_key = "LayerNorm.bias"
            embedding_weight[new_key] = value
    embedder.load_state_dict(embedding_weight, strict=False)
    return embedder


def get_words_index(model_args):
    output_dir = os.path.join(model_args.model_name, f"{model_args.model_type}_share_private_words_index.json")
    vocab_file = os.path.join(model_args.model_name, "vocab.txt")

    bert_words = dict()
    with open(vocab_file) as file:
        for line in file:
            word = line.strip()
            bert_words[word] = len(bert_words)

    private_words_index = []
    share_words_index = []
    for word in bert_words:
        if not word.startswith("[") and any([char.isdigit() for char in word]):
            private_words_index.append(bert_words[word])
        else:
            share_words_index.append(bert_words[word])

    with open(output_dir, "w") as file:
        json.dump({"share_words_index": share_words_index, "private_words_index": private_words_index}, file)

    registry.register("share_words_index", share_words_index)
    registry.register("private_words_index", private_words_index)


def random_private_words(deEmbeding):
    share_words_index = registry.get("share_words_index")
    private_words_index = registry.get("private_words_index")
    logger = registry.get("logger")
    if private_words_index:
        logger.warning("random private words embedding")
        shape = deEmbeding[0].word_embeddings.weight.shape
        temp = torch.randn((len(deEmbeding), shape[0], shape[1]))
        for client_idx, client_embedder in deEmbeding.items():
            temp[client_idx][share_words_index] = deEmbeding[client_idx].word_embeddings.weight[share_words_index]
            deEmbeding[client_idx].word_embeddings.weight = Parameter(temp[client_idx])
        return deEmbeding
    else:
        return deEmbeding


def random_all_words(deEmbeding, ratio=0.5):
    logger = registry.get("logger")
    shape = deEmbeding[0].word_embeddings.weight.shape
    random_num = int(shape[0] * ratio)
    private_words_index = registry.get("private_words_index")
    random_idx = random.sample([i for i in range(shape[0])], random_num)
    logger.warning(f"random words embedding ratio={ratio} and "
                   f"random_num={random_num} + privacy_num={len(private_words_index)}")
    share_words_index = []
    for i in range(shape[0]):
        if i in random_idx or i in private_words_index:
            continue
        else:
            share_words_index.append(i)
    temp = torch.randn((len(deEmbeding), shape[0], shape[1]))
    for client_idx, client_embedder in deEmbeding.items():
        if len(share_words_index) > 0:
            temp[client_idx][share_words_index] = deEmbeding[client_idx].word_embeddings.weight[share_words_index]
        deEmbeding[client_idx].word_embeddings.weight = Parameter(temp[client_idx], requires_grad=True)
    return deEmbeding


def save_embedding_weights(args, embedder, time=None):
    save_path = os.path.join(args.output_dir, f"{args.model_type}_{time}_embedder.h5")
    embedding_parameters_list = []
    for client_indx in range(args.client_num_in_total):
        embedding_parameters_list.append(embedder[client_indx].state_dict())
    torch.save(embedding_parameters_list, save_path)


def skip_parameters(args):
    eval_file = os.path.join(args.save_dir, f"{args.model_type}_sweep_{args.seed}_eval.results")
    patten = f"lr={args.lr}_epoch={args.epochs}"

    if not os.path.exists(eval_file):
        return False, patten
    with open(eval_file) as file:
        for line in file:
            if patten in line:
                return True, line
    return False, patten
