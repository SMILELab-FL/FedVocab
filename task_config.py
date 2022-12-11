stand_task_config_dir = {
    "20news":
        {
            "epochs": "1", "lr": "5e-3", "wd": "0.0001",
            "lstm_dropout": "0.5", "num_layers": "1",
            "embedding_file": "glove.6B.300d.txt", "embedding_dropout": "0.0",
            "do_remove_stop_words": "True", "do_remove_low_freq_words": "5",
            "batch_size": "32", "max_seq_len": "512", "hidden_size": "300",
            "comm_round": "100", "num_filter": "192", "cnn_dropout": "0.1"
        },
    "agnews":
        {
            "epochs": "1", "lr": "5e-3", "wd": "0.0001",
            "lstm_dropout": "0.1", "num_layers": "1",
            "embedding_file": "glove.6B.300d.txt", "embedding_dropout": "0.0",
            "do_remove_stop_words": "False", "do_remove_low_freq_words": "0",
            "batch_size": "128", "max_seq_len": "128", "hidden_size": "300",
            "comm_round": "100", "num_filter": "192", "cnn_dropout": "0.1"
        },
    "sst_2":
        {
            "epochs": "1", "lr": "5e-3", "wd": "0.0001",
            "lstm_dropout": "0.5", "hidden_size": "300", "num_layers": "1",
            "embedding_file": "glove.840B.300d.txt", "embedding_dropout": "0.3",
            "do_remove_stop_words": "False", "do_remove_low_freq_words": "0",
            "batch_size": "400", "max_seq_len": "32",
            "comm_round": "100", "num_filter": "192", "cnn_dropout": "0.1"
        },
}

tune_task_config_dir = {
    "20news":
        {
            "epochs": "10", "lr": "5e-3", "wd": "0.0001",
            "lstm_dropout": "0.5",
            "embedding_file": "glove.6B.300d.txt", "embedding_dropout": "0.0",
            "do_remove_stop_words": "True", "do_remove_low_freq_words": "5",
            "batch_size": "400", "max_seq_len": "512",
            "comm_round": "100", "ci": "1", "num_filter": "50", "cnn_dropout": "0.5"
        },
    "agnews":
        {
            "epochs": "10", "lr": "5e-3", "wd": "0.0001",
            "lstm_dropout": "0.1",
            "embedding_file": "glove.6B.300d.txt", "embedding_dropout": "0.0",
            "do_remove_stop_words": "False", "do_remove_low_freq_words": "0",
            "batch_size": "256", "max_seq_len": "128",
            "comm_round": "60", "ci": "1", "num_filter": "50", "cnn_dropout": "0.1"
        },
    "sst_2":
        {
            "epochs": "1", "lr": "5e-3", "wd": "0.0001",
            "lstm_dropout": "0.5",
            "embedding_file": "glove.840B.300d.txt", "embedding_dropout": "0.3",
            "do_remove_stop_words": "False", "do_remove_low_freq_words": "0",
            "batch_size": "400", "max_seq_len": "32",
            "comm_round": "150", "ci": "1", "num_filter": "50", "cnn_dropout": "0.5"
        },
}

tlm_task_config_dir = {
    "20news":
        {
            "distilbert": {"epochs": "1", "lr": "5e-5", "weight_decay": "0.0",
                           "train_batch_size": "32", "eval_batch_size": "200",
                           "max_seq_len": "256", "comm_round": "100"},
            "bert": {"epochs": "1", "lr": "5e-5", "weight_decay": "0.0",
                     "train_batch_size": "32", "eval_batch_size": "200",
                     "max_seq_len": "256", "comm_round": "100"},
            "albert": {"epochs": "1", "lr": "5e-5", "weight_decay": "0.0",
                       "train_batch_size": "64", "eval_batch_size": "64",
                       "max_seq_len": "256", "comm_round": "100"},
        },
    "agnews":
        {
            "albert": {"epochs": "2", "lr": "5e-5", "weight_decay": "0.0",
                       "train_batch_size": "64", "eval_batch_size": "200",
                       "max_seq_len": "128", "comm_round": "100", },
            "bert": {"epochs": "2", "lr": "5e-5", "weight_decay": "0.0",
                     "train_batch_size": "64", "eval_batch_size": "200",
                     "max_seq_len": "128", "comm_round": "100", },
            "distilbert": {"epochs": "2", "lr": "5e-5", "weight_decay": "0.0",
                           "train_batch_size": "64", "eval_batch_size": "200",
                           "max_seq_len": "128", "comm_round": "100"},
        },
    "sst_2":
        {
            "albert": {"epochs": "3", "lr": "5e-5", "weight_decay": "0.0",
                       "train_batch_size": "256", "eval_batch_size": "128",
                       "max_seq_len": "32", "comm_round": "100", },
            "bert": {"epochs": "3", "lr": "5e-5", "weight_decay": "0.0",
                     "train_batch_size": "256", "eval_batch_size": "200",
                     "max_seq_len": "32", "comm_round": "100", },
            "distilbert": {"epochs": "3", "lr": "5e-5", "weight_decay": "0.0",
                           "train_batch_size": "256", "eval_batch_size": "128",
                           "max_seq_len": "32", "comm_round": "100", },
        },
}
