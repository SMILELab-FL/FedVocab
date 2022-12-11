from abc import ABC, abstractmethod
import h5py
import json

from training.utils.register import registry
from data_preprocessing.base.base_data_loader import BaseDataLoader
from tqdm import tqdm
import h5py
import json
import numpy as np
import pickle
import os


class BaseDataManager(ABC):
    @abstractmethod
    def __init__(self, args, model_args, rank, client_index_list=None):
        self.model_args = model_args
        self.args = args
        self.train_batch_size = model_args.train_batch_size
        self.eval_batch_size = model_args.eval_batch_size
        self.rank = rank
        # self.num_workers = num_workers
        self.client_index_list = client_index_list

        # TODO: add type comments for the below vars.
        self.train_dataset = None
        self.test_dataset = None
        self.train_examples = None
        self.test_examples = None
        self.train_loader = None
        self.test_loader = None
        self.client_index_pointer = 0
        self.attributes = None

        self.num_clients = self.load_num_clients(
            self.args.partition_file_path, self.args.partition_method)
        # TODO: sync to the same logic to sample index
        # self.client_index_list = self.sample_client_index(rank, num_workers)
        # self.client_index_list = self.get_all_clients()
        # self.client_index_list = self.sample_client_index(rank, num_workers)

        self.logger = registry.get("logger")

    @staticmethod
    def load_attributes(data_path):
        with open(data_path, "rb") as file:
            data_file = pickle.load(file)
        attributes = data_file["attributes"]
        return attributes

    @staticmethod
    def load_num_clients(partition_file_path, partition_name):
        with open(partition_file_path, "rb") as file:
            partition_data = pickle.load(file)
        num_clients = partition_data[partition_name]["n_clients"]
        return num_clients

    @abstractmethod
    def read_instance_from_data_file(self, data_file, index_list, desc):
        pass

    def get_all_clients(self):
        return list(range(0, self.num_clients))

    def load_centralized_data(self, cut_off=None):
        state, res = self._load_data_loader_from_cache(-1)
        if state:
            train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = res
        else:
            data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
            partition_file = h5py.File(
                self.args.partition_file_path, "r", swmr=True)
            partition_method = self.args.partition_method
            train_index_list = []
            test_index_list = []
            for client_idx in tqdm(
                    partition_file[partition_method]["partition_data"].keys(),
                    desc="Loading index from h5 file."):
                train_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["train"][()][:cut_off])
                test_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["test"][()][:cut_off])
            train_data = self.read_instance_from_data_file(data_file, train_index_list)
            test_data = self.read_instance_from_data_file(data_file, test_index_list)

            train_examples, train_features, train_dataset = self.preprocessor.transform(
                **train_data, index_list=train_index_list)
            test_examples, test_features, test_dataset = self.preprocessor.transform(
                **test_data, index_list=test_index_list, evaluate=True)

            with open(res, "wb") as handle:
                pickle.dump((train_examples, train_features, train_dataset,
                             test_examples, test_features, test_dataset),
                    handle)

        train_dl = BaseDataLoader(train_examples, train_features, train_dataset,
            batch_size=self.train_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=False)

        test_dl = BaseDataLoader(test_examples, test_features, test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=False)

        return train_dl, test_dl

    def load_federated_data(self, rank, test_cut_off=None, decompose=None, stand_alone=False):
        if rank == 0:
            return self._load_federated_data_server(test_cut_off=test_cut_off, decompose=decompose)
        else:
            return self._load_federated_data_local(decompose, stand_alone=stand_alone)

    def _load_federated_data_server(self, test_only=True, test_cut_off=None, decompose=False):
        state, res = self._load_data_loader_from_cache(0, decompose=decompose)
        train_data_local_dict = None
        train_data_local_num_dict = None
        test_data_local_num_dict = None
        test_data_local_dict = None
        state = False  # test different initialization
        if state:
            train_examples, train_features, train_dataset, \
            test_examples, test_features, test_dataset = res
            self.logger.info("test data size " + str(len(test_examples)))
            if train_dataset is None:
                train_data_num = 0
            else:
                train_data_num = len(train_dataset)
        else:
            data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
            partition_file = h5py.File(
                self.args.partition_file_path, "r", swmr=True)
            partition_method = self.args.partition_method
            train_index_list = []
            test_index_list = []
            for client_idx in tqdm(
                    partition_file[partition_method]
                    ["partition_data"].keys(),
                    desc="Loading index from h5 file."):
                train_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["train"][()])
                local_test_index_list = partition_file[partition_method][
                    "partition_data"][client_idx]["test"][()]
                test_index_list.extend(local_test_index_list)

            if not test_only:
                train_data = self.read_instance_from_h5(
                    data_file, train_index_list)
            if test_cut_off:
                test_index_list.sort()
            test_index_list = test_index_list[:test_cut_off]
            self.logger.info(
                "caching test index size " + str(len(test_index_list)) + "test cut off " + str(test_cut_off))

            test_data = self.read_instance_from_h5(data_file, test_index_list)

            data_file.close()
            partition_file.close()

            train_examples, train_features, train_dataset = None, None, None
            if not test_only:
                train_examples, train_features, train_dataset = self.preprocessor.transform(
                    **train_data, index_list=train_index_list)
            test_examples, test_features, test_dataset = self.preprocessor.transform(
                **test_data, index_list=test_index_list)
            self.logger.info("caching test data size " + str(len(test_examples)))

            with open(res, "wb") as handle:
                pickle.dump((train_examples, train_features, train_dataset, test_examples, test_features, test_dataset),
                    handle)

        if test_only or train_dataset is None:
            train_data_num = 0
            train_data_global = None
        else:
            train_data_global = BaseDataLoader(
                train_examples, train_features, train_dataset,
                batch_size=self.train_batch_size,
                num_workers=0,
                pin_memory=True,
                drop_last=False)
            train_data_num = len(train_examples) * self.train_batch_size
            self.logger.info("train_dl_global number = " + str(len(train_data_global)))

        test_data_global = BaseDataLoader(
            test_examples, test_features, test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=0,
            pin_memory=True,
            drop_last=False)
        test_data_num = len(test_data_global) * self.eval_batch_size
        self.logger.info("test_dl_global number = " + str(test_data_num))

        return (train_data_num, train_data_global,
                test_data_num, test_data_global,
                train_data_local_num_dict, train_data_local_dict,
                test_data_local_num_dict, test_data_local_dict)

    def _load_federated_data_local(self, decompose=None, stand_alone=False):

        with open(self.args.data_file_path, "rb") as file:
            data_file = pickle.load(file)
        with open(self.args.partition_file_path, "rb") as file:
            partition_file = pickle.load(file)

        partition_method = self.args.partition_method

        train_data_local_dict = {}
        dev_data_local_dict = {}
        test_data_local_dict = {}

        train_data_local_num_dict = {}
        test_data_local_num_dict = {}
        dev_data_local_num_dict = {}

        self.client_index_list = list(set(self.client_index_list))

        n_clients = partition_file[partition_method]["n_clients"]
        if n_clients < self.args.client_num_in_total:
            raise ValueError(f"partition data have {n_clients} clients "
                             f"that mismatch you input {self.args.clients_num} clients")
        steps = int(n_clients / self.args.client_num_in_total)

        state, res = self._load_data_loader_from_cache(self.rank, decompose=decompose)
        state = False
        if state:
            train_examples, train_features, train_dataset, \
            dev_examples, dev_features, dev_dataset,\
            test_examples, test_features, test_dataset = res
        else:
            train_examples, train_features, train_dataset = dict(), dict(), dict()
            test_examples, test_features, test_dataset = dict(), dict(), dict()
            dev_examples, dev_features, dev_dataset = dict(), dict(), dict()
            for i, client_idx in enumerate(self.client_index_list):
                train_index_list, test_index_list, valid_index_list = [], [], []
                start = max(client_idx * steps, 0)
                end = min((client_idx + 1) * steps, n_clients)
                if start > end:
                    raise ValueError(f"with {self.args.partition_method}, client number is more than")

                for idx in range(start, end):
                    idx = str(idx)
                    train_list = partition_file[partition_method]["partition_data"][idx]["train"]
                    train_index_list.extend(train_list)
                    valid_list = partition_file[partition_method]["partition_data"][idx]["valid"]
                    valid_index_list.extend(valid_list)
                    test_list = partition_file[partition_method]["partition_data"][idx]["test"]
                    test_index_list.extend(test_list)

                train_data = self.read_instance_from_data_file(
                    data_file, train_index_list,
                    desc=" train data of client_id=%d " % client_idx)

                valid_data = self.read_instance_from_data_file(
                    data_file, valid_index_list,
                    desc=" valid data of client_id=%d " % client_idx)

                test_data = self.read_instance_from_data_file(
                    data_file, test_index_list,
                    desc=" test data of client_id=%d " % client_idx)

                train_examples[client_idx], train_features[client_idx], train_dataset[client_idx] = \
                    self.preprocessor.transform(**train_data, index_list=train_index_list, idx=client_idx)

                test_examples[client_idx], test_features[client_idx], test_dataset[client_idx] = \
                    self.preprocessor.transform(**test_data, index_list=test_index_list, evaluate=True, idx=client_idx)

                dev_examples[client_idx], dev_features[client_idx], dev_dataset[client_idx] = \
                    self.preprocessor.transform(**valid_data, index_list=valid_index_list, evaluate=True)

            # with open(res, "wb") as handle:
            #     pickle.dump(
            #         (train_examples, train_features, train_dataset,
            #          dev_examples, dev_features, dev_dataset,
            #          test_examples, test_features, test_dataset),
            #         handle)

        for i, client_idx in enumerate(self.client_index_list):
            train_loader = BaseDataLoader(
                train_examples[client_idx], train_features[client_idx], train_dataset[client_idx],
                batch_size=self.train_batch_size, num_workers=0, pin_memory=True, drop_last=False)
            test_loader = BaseDataLoader(
                test_examples[client_idx], test_features[client_idx], test_dataset[client_idx],
                batch_size=self.eval_batch_size, num_workers=0, pin_memory=True, drop_last=False)
            dev_loader = BaseDataLoader(
                dev_examples[client_idx], dev_features[client_idx], dev_dataset[client_idx],
                batch_size=self.eval_batch_size, num_workers=0, pin_memory=True, drop_last=False)

            train_data_local_dict[client_idx] = train_loader
            test_data_local_dict[client_idx] = test_loader
            dev_data_local_dict[client_idx] = dev_loader

            train_data_local_num_dict[client_idx] = len(train_loader) * self.train_batch_size
            test_data_local_num_dict[client_idx] = len(test_loader) * self.eval_batch_size
            dev_data_local_num_dict[client_idx] = len(dev_loader) * self.eval_batch_size

        if stand_alone:
            test_examples_all, test_features_all, test_dataset_all = [], [], []
            for client_idx in self.client_index_list:
                test_examples_all += test_examples[client_idx]
                test_features_all += test_features[client_idx]
                test_dataset_all += test_dataset[client_idx]
            test_data_global = BaseDataLoader(
                test_examples_all, test_features_all, test_dataset_all,
                batch_size=self.eval_batch_size*4,
                num_workers=0,
                pin_memory=True,
                drop_last=False)
            test_data_num = len(test_examples_all)

            all_test_index_list = []
            for idx in range(n_clients):
                idx = str(idx)
                test_list = partition_file[partition_method]["partition_data"][idx]["test"]
                all_test_index_list.extend(test_list)
            assert len(all_test_index_list) == test_data_num

            train_data_global, train_data_num = None, 0
            for key in train_examples:
                train_data_num += len(train_examples[key])

            dev_data_num, dev_data_global = 0, None
        else:
            train_data_global, test_data_global, dev_data_global, train_data_num, test_data_num, dev_data_num \
                = None, None, None, 0, 0, 0

        # data_file.close()
        # partition_file.close()

        return (train_data_num, train_data_global,
                test_data_num, test_data_global,
                dev_data_num, dev_data_global,
                train_data_local_num_dict, train_data_local_dict,
                dev_data_local_num_dict, dev_data_local_dict,
                test_data_local_num_dict, test_data_local_dict)

    def _load_data_loader_from_cache(self, rank, decompose=False):
        """
        Different clients has different cache file.
        client_id = -1 means loading the cached file on server end.
        """
        args = self.args
        model_args = self.model_args

        os.makedirs(model_args.cache_dir, exist_ok=True)
        if rank != -1:
            cached_file_name = f"{args.dataset}_niid={args.niid}_alpha={args.alpha}_rank={str(rank)}_" \
                               f"model_type={args.model_type}_seq={args.max_seq_length}_decompose={decompose}"
        else:
            if args.partition_method != "uniform":
                cached_file_name = f"{args.dataset}_niid={args.niid}_alpha={args.alpha}_num={args.client_num_in_total}_" \
                                   f"model_type={args.model_type}_seq={args.max_seq_length}_decompose={decompose}"
            else:
                cached_file_name = f"{args.dataset}_model_type={args.model_type}_seq={args.max_seq_length}"

        cached_features_file = os.path.join(
            model_args.cache_dir, cached_file_name
        )

        if os.path.exists(cached_features_file) and (
                (not model_args.reprocess_input_data and not model_args.no_cache)
                or (model_args.use_cached_eval_features and not model_args.no_cache)
        ):
            self.logger.info(f"Loading features from cached file {cached_features_file}")
            # train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = None, None, None, None, None, None
            with open(cached_features_file, "rb") as handle:
                train_examples, train_features, train_dataset, \
                dev_examples, dev_features, dev_dataset, \
                test_examples, test_features, test_dataset = pickle.load(
                    handle)
            return True, (train_examples, train_features, train_dataset,
                          dev_examples, dev_features, dev_dataset,
                          test_examples, test_features, test_dataset)
        return False, cached_features_file
