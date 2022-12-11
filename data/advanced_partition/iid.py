import numpy as np


def average_sample_nums(sample_num, client_num):
    step = int(sample_num / client_num)
    client_sample_nums = [step for _ in range(client_num - 1)]
    client_sample_nums.append(step + sample_num % client_num)
    return np.array(client_sample_nums)


def split_indices(num_cumsum, rand_perm):
    client_indices_pairs = [(cid, idxs) for cid, idxs in
                            enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    client_dict = dict(client_indices_pairs)
    return client_dict


def homo_partition(sample_num, client_num):
    """Partition data indices in IID way given sample numbers for each clients.
    Args:
        client_num (int): Number of clients.
        sample_num (int): Number of samples.
    Returns:
        dict: ``{ client_id: indices}``.
    """
    client_sample_nums = average_sample_nums(sample_num, client_num)

    rand_perm = np.random.permutation(sample_num)
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    client_dict = split_indices(num_cumsum, rand_perm)
    return {key: list(value.astype(str)) for key, value in client_dict.items()}


def func(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


def new_homo_partition(sample_num, client_num):
    split_list = []
    step = int(sample_num / client_num)
    new_number = step*client_num
    all_list = [i for i in range(sample_num)]
    target_list = all_list[0:new_number]
    split_list = list(func(target_list, step))
    for idx, num in enumerate(all_list[new_number:]):
        split_list[idx%client_num].append(num)
    return {i: [str(j) for j in split_list[i]] for i in range(client_num)}
