import os
import scipy.io as sio
import logging

from config import construct_w_dir

save_dir = construct_w_dir


def get_w_mat_file_path(dataset, view, mr, way):
    w_mat_file = f"{save_dir}/w_ds_{dataset}_mr_{mr}_view_{view}_way_{way}.mat"
    return w_mat_file


def is_w_exist(dataset, view, mr, way):
    w_mat_file = get_w_mat_file_path(dataset, view, mr, way)
    is_exist = os.path.exists(w_mat_file)
    logging.info(f"w_mat_file exists: {is_exist}, w_mat_file: {w_mat_file}")
    return is_exist


def load_w(dataset, view, mr, way):
    w_mat_file = get_w_mat_file_path(dataset, view, mr, way)
    data = sio.loadmat(w_mat_file)
    w = data["w"]
    logging.info(f"load w_mat_file success: {w_mat_file}")
    return w


def save_w(dataset, view, mr, way, w):
    w_mat_file = get_w_mat_file_path(dataset, view, mr, way)
    sio.savemat(w_mat_file, {"w": w})
    logging.info(f"save w_mat_file success: {w_mat_file}")
