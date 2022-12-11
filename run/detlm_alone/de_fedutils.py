import os

import torch
from torch.optim import *
from torch.nn.parameter import Parameter

from fedlab.core.client.trainer import ClientTrainer
from fedlab.utils.serialization import SerializationTool
from fedlab.core.client import SERIAL_TRAINER

from training.utils.register import registry
from run.detlm_alone.misc import evaluation, build_results, update_results


class DeTLMTrainer(ClientTrainer):
    def __init__(self,
                 model,
                 client_num,
                 aggregator=None,
                 cuda=True,
                 logger=None):
        super().__init__(model, cuda)
        self.client_num = client_num
        self.type = SERIAL_TRAINER  # represent serial trainer
        self.aggregator = aggregator
        self.logger = logger
        self.train_round = 0

    def _get_dataloader(self, dataset, client_id: int):
        raise NotImplementedError()

    def _train_alone(self, model_parameters, train_loader, idx):
        raise NotImplementedError()

    def _test_alone(self, model, test_dl, eval_batch_size, idx):
        raise NotImplementedError()

    def train(self, model_parameters, id_list, aggregate=False, share_embedding_weights=None):
        raise NotImplementedError()

    def test(self, model, test_dl, eval_batch_size, idx):
        raise NotImplementedError()


class FedDeTLMTrainer(DeTLMTrainer):
    def __init__(self,
                 args,
                 model,
                 train_dataset,
                 test_dataset,
                 data_slices,
                 trainer=None,
                 embedder=None,
                 aggregator=None,
                 logger=None,
                 cuda=True) -> None:

        super(FedDeTLMTrainer, self).__init__(
            model=model,
            client_num=len(data_slices),
            cuda=cuda,
            aggregator=aggregator,
            logger=logger
        )
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.data_slices = data_slices  # [0, client_num)

        self.trainer = trainer
        self.embedder = embedder

        # self.results = build_results(args)
        self.gpu = self.args.gpu
        self.wandb = registry.get("wandb", None)

        self.share_words_index = registry.get("share_words_index")
        self.private_words_index = registry.get("private_words_index")

        self.every_model_parameters = {}

        self.global_acc = [0.0] * self.client_num
        self.local_acc = [0.0] * self.client_num

    def _get_dataloader(self, dataset, client_id: int):
        if isinstance(dataset, dict):
            train_loader = dataset[client_id]
        else:
            train_loader = dataset
        return train_loader

    def _train_alone(self, model_parameters, train_loader, idx,
                     share_embedding_weights=None, global_test_dl=None,
                     local_test_dl=None):
        SerializationTool.deserialize_model(self._model, model_parameters)

        if self.args.personalized and share_embedding_weights is not None:
            shape = self.embedder[idx].word_embeddings.weight.shape
            temp = torch.randn((shape[0], shape[1]))
            temp[self.share_words_index] = share_embedding_weights
            temp[self.private_words_index] = self.embedder[idx].word_embeddings.weight[self.private_words_index]
            self.embedder[idx].word_embeddings.weight = Parameter(temp)

        if self.args.alter_update:
            self.logger.warning("using alter update")
            self.trainer.alter_train_model(self._model, train_loader, self.embedder[idx], idx)

        self.trainer.train_model(
            model=self._model,
            train_dl=train_loader,
            embedder=self.embedder[idx],
            idx=idx
        )
        self.logger.info(f"Client id {idx} train procedure is finished")
        return self.model_parameters

    def _test_alone(self, model, test_dl, eval_batch_size, idx):
        result, _, _ = evaluation(
            self.args, eval_batch_size, test_dl, model,
            embedder=self.embedder[idx])
        test_acc, test_loss = result["acc"], result["eval_loss"]
        return test_acc, test_loss

    def train(self, model_parameters, id_list, aggregate=False,
              share_embedding_weights=None, global_test_dl=None,
              local_test_dl=None):
        param_list = []
        share_embedding_param_list = []

        for idx in id_list:
            self.logger.info("Starting training procedure "
                             "of client [{}]".format(idx))
            self.logger.info(
                f"{self.args.dataset} {self.args.model_type} "
                f"train with niid={self.args.niid}_lr={self.args.lr}_"
                f"epoch={self.args.epochs}_seed={self.args.seed}_"
                f"comm_round={self.args.comm_round}_"
                f"method={self.args.fl_algorithm}"
            )

            train_data_loader = self._get_dataloader(dataset=self.train_dataset, client_id=idx)
            self._train_alone(
                model_parameters=model_parameters,
                train_loader=train_data_loader, idx=idx,
                share_embedding_weights=share_embedding_weights,
            )

            if self.args.test_mode == "before":
                global_acc, _ = self._test_alone(
                    self._model, global_test_dl,
                    self.args.global_eval_batch_size, idx)
                if global_acc > self.global_acc[idx]:
                    self.global_acc[idx] = global_acc
                self.logger.debug(f"Client: {idx}, GlobalAcc: {global_acc:.3f}, "
                                  f"BestGlobalAcc: {self.global_acc[idx]:.3f}")

                local_acc, _ = self._test_alone(
                    self._model, local_test_dl[idx],
                    self.args.local_eval_batch_size, idx)
                if local_acc > self.local_acc[idx]:
                    self.local_acc[idx] = local_acc
                self.logger.debug(f"Client: {idx}, LocalAcc: {local_acc:.3f}, "
                                  f"BestLocalAcc: {self.local_acc[idx]:.3f}")

            param_list.append(self.model_parameters)
            if self.args.personalized:
                share_embedding_param_list.append(
                    self.embedder[idx].word_embeddings.weight[self.share_words_index].cpu()
                )
            self.embedder[idx].cpu()

        self.train_round += 1
        if aggregate is True and self.aggregator is not None:
            # aggregate model parameters of this client group
            aggregated_parameters = self.aggregator(param_list)
            if self.args.personalized:
                aggregated_share_parameters = self.aggregator(share_embedding_param_list)
            else:
                aggregated_share_parameters = None
            return aggregated_parameters, aggregated_share_parameters
        else:
            return param_list, share_embedding_param_list

    def test(self, model, test_dl, eval_batch_size, idx):
        test_acc, test_loss = self._test_alone(
            model, test_dl,
            eval_batch_size, idx
        )
        self.embedder[idx].cpu()
        return test_acc, test_loss
        # result_list = []
        # for idx in id_list:
        #     self.logger.info(
        #         "Starting test procedure of client [{}]".format(idx))
        #     if self.test_dataset is None:
        #         test_data_loader = test_global
        #     else:
        #         test_data_loader = self._get_dataloader(dataset=self.test_dataset, client_id=idx)
        #     test_acc, test_loss = self._test_alone(test_data_loader, idx)
        #     result_list.append(test_acc)
        #     self.logger.debug(f"Round {round_id} - client id {idx} test acc {round(test_acc, 3)}")
        #     self.embedder[idx].cpu()
        #     if test_acc > self.results["EachMaxAcc"][idx]:
        #         self.logger.debug(f"client {idx} has new best acc is {test_acc}")
        #         self.every_model_parameters[idx] = [self.embedder[idx].state_dict(), self.model.state_dict()]

        # self.results = update_results(self.results, round_id, id_list, result_list)
        # EachMaxAcc = self.results["EachMaxAcc"]
        # avg_acc = round(sum(EachMaxAcc)/len(EachMaxAcc), 3)
        # cur_acc = round(sum(result_list)/len(result_list), 3)
        #
        # self.logger.warning(
        #     f"{self.args.dataset} {self.args.model_type} test with {self.args.test_mode}"
        # )
        # self.logger.critical(
        #     f"Round: {round_id} AvgAcc: {avg_acc} CurAcc: {cur_acc}"
        # )
