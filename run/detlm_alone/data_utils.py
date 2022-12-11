import os

from globalhost import machine_dict
from training.utils.register import registry
from data_manager.data_attributes import tc_data_attributes
from data_preprocessing.text_classification_preprocessor import TLMPreprocessor
from data_manager.text_classification_data_manager import TextClassificationDataManager


def load_and_processing_data(args, model_args, tokenizer, client_index_list=None):
    logger = registry.get("logger")
    logger.warning(f"loading {args.dataset} dataset with niid {args.niid} and {args.alpha}")
    postfix = "pkl"

    if not args.data_file_path:
        args.data_file_path = os.path.join(machine_dict[args.machine_name]["data_path"],
            f"{args.dataset}_data.{postfix}")
    logger.debug(f"data_file_path is {args.data_file_path}")

    if not args.partition_file_path:
        if args.niid:
            if args.partition_method:
                args.partition_file_path = os.path.join(machine_dict[args.machine_name]["partition_path"],
                    f"{args.dataset}_partition.{postfix}")
            else:
                args.partition_file_path = os.path.join(machine_dict[args.machine_name]["partition_path"],
                    f"niid_{args.dataset}_pdata.{postfix}")
                args.partition_method = f"niid_label_clients={args.client_num_in_total}_alpha={args.alpha}"
        else:
            args.partition_file_path = os.path.join(machine_dict[args.machine_name]["partition_path"],
                f"{args.dataset}_partition.{postfix}")
            args.partition_method = "uniform"
    logger.debug(f"partition_file_path is {args.partition_file_path}")

    attributes = TextClassificationDataManager.load_attributes(args.data_file_path)
    label_vocab = attributes["label_vocab"]
    num_labels = len(label_vocab)

    preprocessor = TLMPreprocessor(
        args=model_args, label_vocab=label_vocab,
        tokenizer=tokenizer)
    dm = TextClassificationDataManager(args, model_args, preprocessor, args.rank, client_index_list)

    train_data_num, train_data_global, \
    test_data_num, test_data_global, dev_data_num, dev_data_global, \
    train_data_local_num_dict, train_data_local_dict, dev_data_local_num_dict, dev_data_local_dict, \
    test_data_local_num_dict, test_data_local_dict = \
        dm.load_federated_data(rank=args.rank, decompose=False, stand_alone=True)

    return train_data_num, train_data_global, \
           test_data_num, test_data_global, dev_data_num, dev_data_global, \
           train_data_local_num_dict, train_data_local_dict, dev_data_local_num_dict, dev_data_local_dict, \
           test_data_local_num_dict, test_data_local_dict
