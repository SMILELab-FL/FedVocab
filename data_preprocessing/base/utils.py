# Variables
import os
import spacy
from spacy.lang.cs import Czech
from spacy.lang.ru import Russian
from spacy.lang.en import English
from spacy.lang.zh import Chinese
from spacy.lang.de import German

from collections import defaultdict, Counter

from data_preprocessing.base.globals import *

import random
import gensim
import h5py
import json
import numpy as np

from training.utils.register import registry

FLOAT_SIZE = 4


class SpacyTokenizer:
    def __init__(self):
        self.__zh_tokenizer = None
        self.__en_tokenizer = None
        self.__cs_tokenizer = None
        self.__de_tokenizer = None
        self.__ru_tokenizer = None

    @staticmethod
    def get_tokenizer(lang):
        if lang == "zh":
            # nlp = spacy.load("zh_core_web_sm")
            nlp = Chinese()
        elif lang == "en":
            # nlp = spacy.load("en_core_web_sm")
            nlp = English()
        elif lang == "cs":
            nlp = Czech()
        elif lang == "de":
            # nlp = spacy.load("de_core_web_sm")
            nlp = German()
        elif lang == "ru":
            nlp = Russian()
        else:
            raise Exception("Unacceptable language.")
        return nlp

    @property
    def zh_tokenizer(self):
        if self.__zh_tokenizer is None:
            self.__zh_tokenizer = self.get_tokenizer("zh")
        return self.__zh_tokenizer

    @property
    def en_tokenizer(self):
        if self.__en_tokenizer is None:
            self.__en_tokenizer = self.get_tokenizer("en")
        return self.__en_tokenizer

    @property
    def cs_tokenizer(self):
        if self.__cs_tokenizer is None:
            self.__cs_tokenizer = self.get_tokenizer("cs")
        return self.__cs_tokenizer

    @property
    def de_tokenizer(self):
        if self.__de_tokenizer is None:
            self.__de_tokenizer = self.get_tokenizer("de")
        return self.__de_tokenizer

    @property
    def ru_tokenizer(self):
        if self.__ru_tokenizer is None:
            self.__ru_tokenizer = self.get_tokenizer("ru")
        return self.__ru_tokenizer


def build_vocab(x):
    # x -> [num_seqs, num_tokens]
    vocab = dict()
    for single_x in x:
        for token in single_x:
            if token not in vocab:
                vocab[token] = len(vocab)
    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    return vocab


def build_freq_vocab(x):
    freq_vocab = dict()
    for single_x in x:
        for token in single_x:
            if token not in freq_vocab:
                freq_vocab[token] = 1
            else:
                freq_vocab[token] += 1
    return freq_vocab


def padding_data(x, max_sequence_length):
    padding_x = []
    seq_lens = []
    for single_x in x:
        # print(single_x)
        new_single_x = single_x.copy()
        if len(new_single_x) <= max_sequence_length:
            seq_lens.append(len(new_single_x))
            for _ in range(len(new_single_x), max_sequence_length):
                new_single_x.append(PAD_TOKEN)
        else:
            seq_lens.append(max_sequence_length)
            new_single_x = new_single_x[:max_sequence_length]
        padding_x.append(new_single_x)
    return padding_x, seq_lens


def padding_char_data(x, max_sequence_length, max_word_length):
    padding_x = []
    word_lens = []
    for sent in x:
        new_sent = []
        temp_word_lens = []
        for chars in sent:
            new_chars = chars.copy()
            if len(new_chars) <= max_word_length:
                temp_word_lens.append(len(new_chars))
                for _ in range(len(new_chars), max_word_length):
                    new_chars.append(PAD_TOKEN)
            else:
                temp_word_lens.append(max_word_length)
                new_chars = new_chars[:max_word_length]
            new_sent.append(new_chars)
        if len(new_sent) <= max_sequence_length:
            for _ in range(len(new_sent), max_sequence_length):
                new_sent.append([PAD_TOKEN for _ in range(max_word_length)])
        else:
            new_sent = new_sent[:max_sequence_length]
            temp_word_lens = temp_word_lens[:max_sequence_length]
        word_lens.append(temp_word_lens)
        padding_x.append(new_sent)
    return padding_x, word_lens


def token_to_idx(x, vocab):
    idx_x = []
    for single_x in x:
        new_single_x = []
        for token in single_x:
            if token in vocab:
                new_single_x.append(vocab[token])
            else:
                new_single_x.append(vocab[UNK_TOKEN])
        idx_x.append(new_single_x)
    return idx_x

def char_to_idx(x, vocab):
    idx_x = []
    for sent in x:
        new_sent = []
        for token in sent:
            new_token = []
            for ch in token:
                if ch in vocab:
                    new_token.append(vocab[ch])
                else:
                    new_token.append(vocab[UNK_TOKEN])
            new_sent.append(new_token)
        idx_x.append(new_sent)
    return idx_x


def label_to_idx(y, vocab):
    idx_y = []
    for label in y:
        idx_y.append(vocab[label])
    return idx_y


def remove_words(x, removed_words):
    remove_x = []
    for single_x in x:
        new_single_x = []
        for token in single_x:
            if token not in removed_words:
                new_single_x.append(token)
        remove_x.append(new_single_x)
    return remove_x


def load_word2vec_embedding(path, source_vocab):
    vocab = dict()
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    weights = []
    for key, value in model.vocab.items():
        if source_vocab is not None and key in source_vocab:
            vocab[key] = len(vocab)
            weights.append(model.vectors[value.index])
        else:
            vocab[key] = len(vocab)
            weights.append(model.vectors[value.index])
    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    weights.append(np.zeros(model.vector_size))
    weights.append(np.zeros(model.vector_size))
    weights = np.array(weights)
    return vocab, weights


def load_pretrained_embedding(path, dimension):
    pretrained_dict = dict()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            temp = line.split(" ")
            word = " ".join(temp[:-dimension])
            pretrained_dict[word] = np.array([float(num) for num in temp[-dimension:]])
    return pretrained_dict


def load_glove_embedding(path, source_vocab, dimension, pretrained_dict=None):
    vocab = dict()
    weights = []
    pre_cnt, us_cnt = 0, 0
    logger = registry.get("logger")

    if pretrained_dict:
        for word in pretrained_dict:
            if source_vocab is not None:
                if word in source_vocab:
                    vocab[word] = len(vocab)
                    weights.append(pretrained_dict[word])
                    us_cnt += 1
            else:
                vocab[word] = len(vocab)
                weights.append(pretrained_dict[word])
            pre_cnt += 1
    else:
        logger.info(f"load glove pretrained embedding from {path}")
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                temp = line.split(" ")
                word = " ".join(temp[:-dimension])
                if source_vocab is not None:
                    if word in source_vocab:
                        vocab[word] = len(vocab)
                        weights.append(np.array([float(num) for num in temp[-dimension:]]))
                        us_cnt += 1
                else:
                    vocab[word] = len(vocab)
                    weights.append(np.array([float(num) for num in temp[-dimension:]]))
                pre_cnt += 1

    vocab[PAD_TOKEN] = len(vocab)
    vocab[UNK_TOKEN] = len(vocab)
    weights.append(np.zeros(dimension))
    weights.append(np.zeros(dimension))
    weights = np.array(weights)

    registry.register("pretrained_vocb_size", pre_cnt)
    registry.register("using_pre_size", us_cnt if us_cnt != 0 else pre_cnt)
    logger.info(f"pretrained vocabulary size is {pre_cnt}*{dimension}")
    logger.info(f"using pretrained vocabulary size is {us_cnt}*{dimension}")
    return vocab, weights


def personal_client_load_glove_embedding(path, source_vocab, dimension):
    clients_vocab = dict()
    clients_weights = []
    pre_cnt = 0
    us_cnt = 0
    pretrained_words = dict()

    logger = registry.get("logger")
    logger.debug(f"client load glove embedding from {path}")
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            temp = line.split(" ")
            word = " ".join(temp[:-dimension])
            pretrained_words[word] = [float(num) for num in temp[-dimension:]]
            pre_cnt += 1

    # for client_index, client_source_vocab in source_vocab.items():
    uninitial_words = []
    for word in source_vocab:
        if pretrained_words.get(word, None):
            clients_vocab[word] = len(clients_vocab)
            clients_weights.append(np.array([float(num) for num in temp[-dimension:]]))
            us_cnt += 1
        else:
            # clients_vocab[word] = len(clients_vocab)
            # clients_weights.append(np.random.randn(dimension))
            uninitial_words.append(word)
    mask_id = len(clients_vocab)
    for i, word in enumerate(uninitial_words):
        clients_vocab[word] = len(clients_vocab)
        clients_weights.append(np.random.randn(dimension))

    clients_vocab[PAD_TOKEN] = len(clients_vocab)
    clients_vocab[UNK_TOKEN] = len(clients_vocab)
    clients_weights.append(np.zeros(dimension))
    clients_weights.append(np.zeros(dimension))
    clients_weights = np.array(clients_weights)

    registry.register("pretrained_vocb_size", pre_cnt)
    registry.register("using_pre_size", us_cnt)
    logger.info(f"pretrained vocabulary size is {pre_cnt}*{dimension}")
    logger.debug(
        "using pretrained vocabulary size is {}".format(us_cnt))
    return clients_vocab, clients_weights


def client_load_glove_embedding(path, source_vocab, dimension):
    clients_vocab = defaultdict(dict)
    clients_weights = defaultdict(list)
    clients_num = len(source_vocab)
    pre_cnt = 0
    us_cnt = dict()
    for client_idx in range(clients_num):
        us_cnt[client_idx] = 0

    logger = registry.get("logger")
    logger.debug(f"client load glove embedding from {path}")
    pretrained_dict = load_pretrained_embedding(path, dimension)
    pre_cnt = len(pretrained_dict)

    for client_index, client_source_vocab in source_vocab.items():
        for word in client_source_vocab:
            if word in pretrained_dict:
                clients_vocab[client_index][word] = len(clients_vocab[client_index])
                clients_weights[client_index].append(pretrained_dict[word])
                us_cnt[client_index] += 1

    for client_index in range(clients_num):
        clients_vocab[client_index][PAD_TOKEN] = len(clients_vocab[client_index])
        clients_vocab[client_index][UNK_TOKEN] = len(clients_vocab[client_index])
        clients_weights[client_index].append(np.zeros(dimension))
        clients_weights[client_index].append(np.zeros(dimension))
        clients_weights[client_index] = np.array(clients_weights[client_index])

    registry.register("pretrained_vocb_size", pre_cnt)
    registry.register("using_pre_size", us_cnt)
    logger.info(f"pretrained vocabulary size is {pre_cnt}*{dimension}")
    logger.debug("using pretrained vocabulary size is {}".format({us_cnt[client_index] for client_index in range(clients_num)}))
    return clients_vocab, clients_weights


def share_private_load_glove_embedding(pretrain_path, source_vocab, dimension):
    pretrain_weights = load_pretrained_embedding(pretrain_path, dimension)

    private_words_path = registry.get("private_words_path")
    if private_words_path:
        with open(private_words_path) as f:
            pop_words = json.load(f)
            pop_exist_flag = True
    else:
        pop_words = []
        pop_exist_flag = False

    private_vocab = defaultdict(list)
    private_weights = defaultdict(list)
    for client_idx, client_vocab in source_vocab.items():
        for word in client_vocab:
            if word in pretrain_weights:
                if pop_exist_flag and word in pop_words:
                    private_vocab[client_idx].append(word)
                    private_weights[client_idx].append(np.random.randn(dimension))
                elif not pop_exist_flag and any([char.isdigit() for char in word]):
                    private_vocab[client_idx].append(word)
                    private_weights[client_idx].append(np.random.randn(dimension))
                    pop_words.append(word)
                    # 可能有多个重复词
    pop_words = list(set(pop_words))

    share_vocab = dict()
    share_weights = []
    for word in pretrain_weights:
        if word not in pop_words:
            share_vocab[word] = len(share_vocab)
            share_weights.append(pretrain_weights[word])

    registry.register("share_index", len(share_vocab))

    clients_vocab = defaultdict(dict)
    clients_weights = defaultdict(list)
    for client_idx, client_vocab in private_vocab.items():
        clients_vocab[client_idx] = share_vocab.copy()
        clients_weights[client_idx] = share_weights.copy()
        for word in private_vocab[client_idx]:
            clients_vocab[client_idx][word] = len(clients_vocab[client_idx])
        clients_weights[client_idx].extend(private_weights[client_idx])

        clients_vocab[client_idx][PAD_TOKEN] = len(clients_vocab[client_idx])
        clients_vocab[client_idx][UNK_TOKEN] = len(clients_vocab[client_idx])
        clients_weights[client_idx].append(np.zeros(dimension))
        clients_weights[client_idx].append(np.zeros(dimension))

        assert len(clients_weights[client_idx]) == len(clients_vocab[client_idx])

    return clients_vocab, clients_weights


def client_random_load_glove_embedding(pretrain_path, source_vocab, dimension):
    logger = registry.get("logger")
    glove_sizes = ["6", "42"]  # 6B maybe use
    clients_vocab = defaultdict(dict)
    clients_weights = defaultdict(list)

    client_id_list = registry.get("client_id_list")
    pretrained_embedding_dict = dict()
    for size in glove_sizes:
        embedding_path = os.path.join(pretrain_path, f"glove.{size}B.{dimension}d.txt")
        logger.info(f"loading pretrained embedding from {embedding_path}")
        pretrained_embedding_dict[size] = load_pretrained_embedding(embedding_path, dimension)
    p = np.array([0.9, 0.1])
    client_sizes = []
    for client_idx in client_id_list:
        size = np.random.choice(glove_sizes, p=p.ravel())
        client_sizes.append(size)
        client_source = source_vocab[client_idx]

        clients_vocab[client_idx], clients_weights[client_idx] = \
            load_glove_embedding(
                embedding_path, client_source, dimension,
            pretrained_dict=pretrained_embedding_dict[size]
        )
    return clients_vocab, clients_weights


def NER_data_formatter(ner_data):
    formatted_data = []
    if len(ner_data["X"]) != len(ner_data["Y"]):
        print(ner_data["X"])
        print(ner_data["Y"])
        print(len(ner_data["X"]), len(ner_data["Y"]))
    assert len(ner_data["X"]) == len(ner_data["Y"])
    sent_id = 0
    for x, y in zip(ner_data["X"], ner_data["Y"]):
        assert len(x) == len(y)
        for token, tag in zip(x, y):
            formatted_data.append([sent_id, token, tag])
        sent_id += 1
    return formatted_data


def generate_h5_from_dict(file_name, data_dict):
    def dict_to_h5_recursive(h5_file, path, dic):
        for key, value in dic.items():
            if isinstance(value, dict):
                if key == "attributes":
                    h5_file[path + str(key)] = json.dumps(value)
                else:
                    dict_to_h5_recursive(h5_file, path + str(key) + "/", value)
            else:
                if isinstance(value, list) and (len(value) > 0 and isinstance(value[0], str)):
                    h5_file[path + str(key)] = np.array([v.encode('utf8') for v in value], dtype="S")
                else:
                    h5_file[path + str(key)] = value
    f = h5py.File(file_name, "w")
    dict_to_h5_recursive(f, "/", data_dict)
    f.close()


def decode_data_from_h5(data):
    if isinstance(data, bytes):
        return data.decode("utf8")
    return data