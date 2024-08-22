import logging
import os
import sys

# sys.path.append({os.path.dirname(os.path.dirname(__file__))}/code")
from constant import dataset_list
from train_base import train_base, get_args_by_mr_and_dataset
from common_util import prepare_log, train_wrapper
from config import log_dir


def train_duibi(dataset_name, miss_ratio, args):
    return train_base(dataset_name, miss_ratio, args)


if __name__ == "__main__":
    train_kind = "train_duibi"
    gpuid = 0

    log_file_name = f"{train_kind}"
    prepare_log(3, log_dir, log_file_name)

    miss_ratio_list = [0.5]
    dataset_list = dataset_list[0:1]
    recorder = [["miss_ratio", "dataset", "results[acc(avg, std), nmi, fscore]"]]

    for mr_index in range(len(miss_ratio_list)):
        for dataset_index in range(len(dataset_list)):
            miss_ratio = miss_ratio_list[mr_index]
            dataset_name = dataset_list[dataset_index]
            logging.info(
                f"duibi progress: miss_ratio_list[{mr_index+1}/{len(miss_ratio_list)}], dataset_list[{dataset_index+1}/{len(dataset_list)}]"
            )

            args = get_args_by_mr_and_dataset(miss_ratio, dataset_name)
            results = train_wrapper(
                train_duibi, dataset_name, miss_ratio, args, gpu_id=gpuid
            )
            recorder.append([miss_ratio, dataset_name, results])

    for row in recorder:
        logging.info(row)
