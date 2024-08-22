import numpy as np
from sklearn import preprocessing
import scipy.io as sio
import logging
import math, random
import scipy.sparse
import os

from config import dataset_dir

def load_dataset(dataset_name):
    # X: VxNxD, Y:Nx1

    # dataset_dir = f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/dataset"
    # dataset_dir="/home/baishunshun/no_complete_cluster_all/dataset"
    data_path = f"{dataset_dir}/{dataset_name}.mat"
    data = sio.loadmat(data_path)

    # if dataset_name == "ADNI_3classes":
    #     X = [x.T for x in data["X"][0]]
    #     Y = data["gt"]
    # elif dataset_name == "3sources_incomplete":
    #     X = data["X"][0]
    #     Y = data["Y"].T
    if dataset_name == "Digit-Product":
        X = [data["X1"].reshape((30000, 1024)), data["X2"].reshape((30000, 1024))]
        Y = data["Y"].T
    elif dataset_name == "NoisyMNIST_30000n_784d":
        X = [data["X1"], data["X2"]]
        Y = data["Y"].T
    elif dataset_name == "NoisyMNIST_50000n_784d":
        X = [data["X1"], data["X2"]]
        Y = data["trainLabel"]
    elif dataset_name == "Multi-MNIST":
        X = [data["X1"].reshape((70000, 1024)), data["X2"].reshape((70000, 1024))]
        Y = data["Y"].T
    else:
        X = data["X"][0]
        Y = data["Y"]

    X = normalize_feature(X, dataset_name)
    Y = Y.astype(int)
    n_cluster = len(np.unique(Y))
    n_sample = len(Y)
    n_view = len(X)

    dims = [x.shape[1] for x in X]
    logging.info(
        f"dataset: {dataset_name}, n_cluster: {n_cluster}, n_sample: {n_sample}, n_view: {n_view}, dims: {dims}"
    )
    return X, Y, n_cluster, n_sample, n_view


def normalize_feature(X, dataset_name):
    if isinstance(X[0], scipy.sparse.csc_matrix):
        # 每个视角的数据类型为<class 'scipy.sparse._csc.csc_matrix'>稀疏矩阵，不能进行中心化
        X = [preprocessing.scale(x, with_mean=False) for x in X]
    else:
        if dataset_name in ["coil20mv", "hdigit"]:
            # 这两个数据集用scale方法处理，报如下警告：UserWarning: Numerical issues were encountered when centering the data and might not be solved. Dataset may contain too large values. You may need to prescale your features.
            scaler = preprocessing.StandardScaler()
            X = [scaler.fit_transform(x) for x in X]
        else:
            X = [preprocessing.scale(x) for x in X]
    return X


def compute_mask(view_num, data_size, missing_ratio):
    """
    对于有缺失视角的样本，可以保证至少有一个视角存在、至少有一个视角缺失，即产生任意视角缺失的数据，适用于两个及两个以上视角
    :param view_num: number of views
    :param data_size: size of data
    :param missing_ratio: missing ratio
    :return: mask matrix [data_size, view_num]
    """
    assert view_num >= 2
    miss_sample_num = math.floor(data_size * missing_ratio)
    data_ind = [i for i in range(data_size)]
    random.shuffle(data_ind)
    miss_ind = data_ind[:miss_sample_num]
    mask = np.ones([data_size, view_num])
    for j in range(miss_sample_num):
        while True:
            rand_v = np.random.rand(view_num)
            v_threshold = np.random.rand(1)
            observed_ind = rand_v >= v_threshold
            ind_ = ~observed_ind
            rand_v[observed_ind] = 1
            rand_v[ind_] = 0
            # np.sum(rand_v) > 0保证至少有一个视角存在
            # np.sum(rand_v) < view_num保证至少有一个视角缺失
            if np.sum(rand_v) > 0 and np.sum(rand_v) < view_num:
                break
        mask[miss_ind[j]] = rand_v

    return mask


def get_mask(view_num, data_size, missing_ratio, dataset_name):
    # mask_dir = f"{os.path.dirname(os.path.dirname(__file__))}/dataset"
    mask_dir = dataset_dir
    if dataset_name == "ADNI_3classes":
        mask_path = f"{mask_dir}/ADNI_sn.mat"
        data = sio.loadmat(mask_path)
        mask = data["Sn"]
    elif dataset_name == "3sources_incomplete":
        mask_path = f"{mask_dir}/3sources_incomplete.mat"
        data = sio.loadmat(mask_path)
        mask = data["mask"]
    else:
        mask = compute_mask(view_num, data_size, missing_ratio)
    logging.info(f"mask.shape: {mask.shape}")
    return mask


if __name__ == "__main__":
    from common_util import prepare_log

    prepare_log(log_kind=1)

    # load_dataset("100leaves")
    get_mask(3,1600,0,"100leaves")

    # load_dataset("ADNI_3classes")
    # get_mask(1,1,1,"ADNI_3classes")

    # load_dataset("3sources_incomplete")
    # get_mask(1,1,1,"3sources_incomplete")

    # load_dataset("mnist_usps")
    # load_dataset("fashion")
    # load_dataset("animal")
    # load_dataset("Digit-Product")
    # load_dataset("NoisyMNIST_30000n_784d")
    # load_dataset("NoisyMNIST_50000n_784d")
    # load_dataset("Multi-MNIST")
