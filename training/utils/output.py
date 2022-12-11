import os
from training.utils.register import registry


def MyReport(args, global_max_acc=None):
    types = registry.get("role")
    if types == "client":
        all_vocb_size = registry.get("all_vocb_size")
        pretrained_vocb_size = registry.get("pretrained_vocb_size")
        using_pre_size = registry.get("using_pre_size")
        training_vocb_size = registry.get("training_vocb_size")

        registry.register("training_vocb_size",
                          str(training_vocb_size) + "(" + str(round(using_pre_size / training_vocb_size, 3)) + ")")
        registry.register("pretrained_vocb_size",
                          str(pretrained_vocb_size) + "(" + str(round(using_pre_size / pretrained_vocb_size, 2)) + ")")
        registry.register("all_vocb_size",
                          str(all_vocb_size) + "(" + str(round(using_pre_size / all_vocb_size, 2)) + ")")

    if args.grid_search and types == "server":
        report_path = f"./fed_results/{args.dataset}/bs={args.batch_size}_e={args.epochs}_" \
                      f"lr={args.lr}_r={args.comm_round}/"
        grid_search_best_file = f"./fed_results/{args.dataset}/grid_search.report"
    else:
        report_path = os.path.join(args.output_dir, f"{args.dataset}/stand_hp/")
        # report_path = f"./fed_results/{args.dataset}/stand_hp/"
        grid_search_best_file = None

    if not os.path.exists(report_path):
        os.mkdir(report_path)
    report_file_name = "server" if types == "server" else str(args.rank - 1)
    report_file = os.path.join(report_path,
                               f"{args.dataset}_{report_file_name}.report")

    registry.unregister("wandb")
    registry.unregister("logger")
    registry.unregister("args")

    with open(report_file, "w") as file:
        keys = registry.get_keys()
        for key in keys:
            value = registry.get(key)
            file.write(",".join([key, str(value)]) + "\n")

    if grid_search_best_file:
        with open(grid_search_best_file, "a+") as file:
            info = f"bs={args.batch_size}_e={args.epochs}_lr={args.lr}_r={args.comm_round}"
            global_max_acc = str(round(global_max_acc, 3))
            file.write(",".join([info, global_max_acc]) + "\n ")