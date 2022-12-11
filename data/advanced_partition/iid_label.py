import h5py
import json
import argparse
from tqdm import tqdm
from iid import homo_partition, new_homo_partition


def add_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--client_number",
        type=int,
        default="10",
        metavar="CN",
        help="client number for lda partition",
    )

    parser.add_argument(
        "--data_file",
        type=str,
        help="data pickle file path",
    )

    parser.add_argument(
        "--partition_file",
        type=str,
        help="partition pickle file path",
    )

    args = parser.parse_args()
    return args


def decode_data_from_h5(data):
    if isinstance(data, bytes):
        return data.decode("utf8")
    return data


def get_partition_label(h5_data, attributes, client_num, mode="train"):
    partition_label_dict = dict()
    label_vocab = attributes["label_vocab"]

    # plist指的是ilist在， ilist指的是Index列表
    for key in list(label_vocab.keys()):
        partition_label_dict[key] = {"plist": [], "ilist": []}

    for idx in attributes[f"{mode}_index_list"]:
        idx = str(idx)
        label = decode_data_from_h5(h5_data["Y"][idx][()])
        partition_label_dict[label]["ilist"].append(idx)

    for key, value in partition_label_dict.items():
        value["plist"] = homo_partition(len(value["ilist"]), client_num)
        # value["plist"] = new_homo_partition(len(value["ilist"]), client_num)

    return partition_label_dict


def get_pdata(partition_id, partition_label_dict):
    pdata = []
    for key, value in partition_label_dict.items():
        pindex = value["plist"][partition_id]
        pdata.extend([int(value["ilist"][int(idx)]) for idx in pindex])
    return pdata


def iid_partition(h5_data, attributes, h5_partition_path=None, client_num=100):
    train_partition_label_dict = get_partition_label(h5_data, attributes, client_num, mode="train")
    test_partition_label_dict = get_partition_label(h5_data, attributes, client_num, mode="test")
    valid_index_list = list(attributes["valid_index_list"])
    h5_partition_data = h5py.File(h5_partition_path, "a")

    print("del /uniform/n_clients and /uniform/partition_data")
    if "/uniform/n_clients" in h5_partition_data:
        del h5_partition_data["/uniform/n_clients"]
    if "/uniform/partition_data" in h5_partition_data:
        del h5_partition_data["/uniform/partition_data"]

    h5_partition_data["/uniform/n_clients"] = client_num

    for partition_id in tqdm(range(client_num)):
        train_pdata = get_pdata(partition_id, train_partition_label_dict)
        test_pdata = get_pdata(partition_id, test_partition_label_dict)

        train_path = "/uniform" + "/partition_data/" + str(partition_id) + f"/train/"
        test_path = "/uniform" + "/partition_data/" + str(partition_id) + f"/test/"
        valid_path = "/uniform" + "/partition_data/" + str(partition_id) + f"/valid/"
        h5_partition_data[train_path] = train_pdata
        h5_partition_data[test_path] = test_pdata
        h5_partition_data[valid_path] = valid_index_list

    h5_partition_data.close()


if __name__ == "__main__":
    args = add_args()
    h5_data = h5py.File(args.data_file, "r")
    try:
        attributes = json.loads(decode_data_from_h5(h5_data["attributes"][()]))
    except:
        attributes = h5_data["attributes"]

    iid_partition(h5_data, attributes, h5_partition_path=args.partition_file,
        client_num=args.client_number
    )

