import logging, os, torch
import numpy as np
import random, time


def train_wrapper(train_func, dataset_name, miss_ratio, args, seed=1234, gpu_id=0):
    logging.info(f"set gpu_id: {gpu_id}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    logging.info(f"set random seed: {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    start_time = time.time()
    try:
        logging.info(
            f"train start: train_func: {train_func.__name__}, dataset: {dataset_name}, miss_ratio: {miss_ratio}, args: {args}"
        )
        results = train_func(dataset_name, miss_ratio, args)
        logging.info("train end")
    except Exception as e:
        logging.exception(f"Exception occurred: {str(e)}")
    logging.info(f"Elapsed time: {time.time() - start_time:.2f} seconds")
    return results


def prepare_log(log_kind=1, log_dir=None, log_file_name=None):
    # log_kind:1表示输出到控制台，2表示输出到文件，3表示同时输出到控制台和文件

    log_format = "%(asctime)s [%(filename)s:%(lineno)d] %(levelname)s: %(message)s"
    log_level = logging.INFO
    date_format = "%Y-%m-%d %H-%M-%S"

    if log_kind == 1:
        logging.basicConfig(format=log_format, level=log_level, datefmt=date_format)
    elif log_kind == 2:
        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir)
        cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        log_path = f"{log_dir}/{log_file_name}-{cur_time}.log"
        logging.basicConfig(
            filename=log_path,
            filemode="a",
            format=log_format,
            level=log_level,
            datefmt=date_format,
        )
    elif log_kind == 3:
        if os.path.exists(log_dir) is False:
            os.makedirs(log_dir)
        logging.basicConfig(format=log_format, level=log_level, datefmt=date_format)
        cur_time = time.strftime("%Y-%m-%d %H-%M-%S", time.localtime(time.time()))
        log_path = f"{log_dir}/{log_file_name}-{cur_time}.log"
        # 创建一个FileHandler对象，用于将日志输出到文件
        file_handler = logging.FileHandler(log_path,encoding="utf-8")
        # 设置FileHandler对象的日志级别为DEBUG，即输出所有级别的日志信息
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))

        # 将FileHandler添加到logging对象中
        logging.getLogger("").addHandler(file_handler)
        return log_path,file_handler
