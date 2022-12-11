def add_args(parser):
    parser.add_argument('--dataset', type=str, default='agnews', metavar='N',
        help='dataset used for training')

    parser.add_argument(
        '--data_file_path', type=str,
        default='',
        help='data h5 file path')

    parser.add_argument(
        '--partition_file_path', type=str,
        default='',
        help='partition h5 file path')

    parser.add_argument('--partition_method', type=str, default=None,
        help='partition method')

    parser.add_argument('--machine_name', type=str, default='v100', metavar='MN',
        help='machine used in training')

    # Model related
    parser.add_argument('--model_type', type=str, default='bert', metavar='N',
        help='transformer model type')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', metavar='N',
        help='transformer model name')
    parser.add_argument('--do_lower_case', type=bool, default=True, metavar='N',
        help='transformer model name')

    parser.add_argument('--do_train', type=lambda x: (str(x).lower() == 'true'), default=False, metavar='N')
    parser.add_argument('--do_test', type=lambda x: (str(x).lower() == 'true'), default=False, metavar='N')
    parser.add_argument('--test_mode', type=str, default="after")
    parser.add_argument('--random_p', type=list, default=[0.5, 0.5])

    # Learning related
    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N',
        help='input batch size for training (default: 8)')

    parser.add_argument('--eval_batch_size', type=int, default=8, metavar='N',
        help='input batch size for evaluation (default: 8)')

    parser.add_argument('--epochs', type=int, default=3, metavar='EP',
        help='how many epochs will be trained locally')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
        help='how many steps for accumulate the loss.')

    parser.add_argument('--max_seq_length', type=int, default=128, metavar='N',
        help='maximum sequence length (default: 128)')

    parser.add_argument('--n_gpu', type=int, default=1, metavar='EP',
        help='how many gpus will be used ')

    parser.add_argument('--fp16', default=False, action="store_true",
        help='if enable fp16 for training')

    parser.add_argument('--seed', type=int, default=42, metavar='N',
        help='random seed')

    parser.add_argument('--client_optimizer', type=str, default='adam',
        help='Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
        help='learning rate on the client (default: 0.001)')

    parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
        help='L2 penalty')

    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--server_optimizer', type=str, default='sgd',
        help='Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--server_lr', type=float, default=0.1,
        help='server learning rate (default: 0.001)')

    parser.add_argument('--server_momentum', type=float, default=0,
        help='server momentum (default: 0)')

    parser.add_argument('--fedprox_mu', type=float, default=0.1,
        help='server momentum (default: 1)')

    parser.add_argument(
        '--evaluate_during_training_steps', type=int, default=100, metavar='EP',
        help='the frequency of the evaluation during training')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
        help='the frequency of the algorithms')

    parser.add_argument('--warmup_ratio', type=float, default=0.0)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--max_grad_norm', type=float, default=15.0)
    parser.add_argument('--embedding_tuning_ratio', type=float, default=0.2)
    parser.add_argument('--optimizer', type=str, default="adamw")

    # visual related
    parser.add_argument('--wandb_time', type=str, default='0')
    parser.add_argument("--wandb_enable", type=lambda x: (str(x).lower() == 'true'), default=False,
        metavar="WE",
        help="wandb enable")

    parser.add_argument('--vocabulary_type', type=str, default='all')

    # IO related
    parser.add_argument('--output_dir', type=str, default="/tmp/", metavar='N',
        help='path to save the trained results and ckpts')

    # cached related
    parser.add_argument('--reprocess_input_data', action='store_true',
        help='whether generate features')

    # freeze related
    parser.add_argument('--freeze_layers', type=str, default='', metavar='N',
        help='freeze which layers')

    # fednpm learning config
    parser.add_argument('--fl_algorithm', type=str, default="",
        help='Algorithm list: FedAvg; FedOPT; FedProx; "" ')

    parser.add_argument('--client_num_in_total', type=int, default=-1, metavar='NN',
        help='number of clients in a distributed cluster')

    parser.add_argument('--client_num_per_in_round', type=int, default=10, metavar='NN',
        help='number of clients in a distributed cluster')

    parser.add_argument('--gpu_server_num', type=int, default=2,
        help='gpu_server_num')

    parser.add_argument('--gpu_num_per_server', type=int, default=5,
        help='gpu_num_per_server')

    parser.add_argument('--gpu_num_per_sub_server', type=int, default=5,
        help='gpu_num_per_sub_server using in scale')

    parser.add_argument('--share_private_words_index_path', type=str, default=None)
    parser.add_argument("--random_ratio", type=float, default=0.0)
    parser.add_argument("--ip", type=str, help="fednpm options.")
    parser.add_argument("--port", type=str, default="", help="fednpm options.")
    parser.add_argument("--world_size", type=int, default=-1, help="fednpm options.")
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--gpu", type=str, default="-1")
    parser.add_argument("--ethernet", type=str, default=None)
    parser.add_argument('--comm_round', type=int, default=10,
        help='how many round of communications we shoud use')
    parser.add_argument('--ci', type=float, default=0.1,
        help='CI')
    parser.add_argument("--niid", type=lambda x: (str(x).lower() == 'true'), default=False,
        metavar="Niid",
        help="niid enable")
    parser.add_argument('--alpha', type=str, default='1.0')

    args = parser.parse_args()
    return args