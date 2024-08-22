import logging
import numpy as np
import torch
import torch.optim as optim
from itertools import chain

from data_process import load_dataset, get_mask
from model import DegradationModel
from metrics import cluster
from certain_util import (
    constructW_by_sklearn,
    constructW_by_faiss,
    find_neighbor_index,
    make_imputation_near_first,
    make_imputation_by_mean,
)
from construct_w_helper import is_w_exist, save_w, load_w
from constant import (
    fixed_args,
    mr05_dataset_args_map,
)


def get_args_by_mr_and_dataset(miss_ratio, dataset):
    if miss_ratio != 0.5:
        raise NotImplementedError(f"the args of miss_ratio {miss_ratio} is not found")
    args = mr05_dataset_args_map[dataset]
    args.update(fixed_args)
    return args


def train_base(
    dataset_name,
    miss_ratio,
    args,
    is_tuihua_kehuode=True,
    is_tuzhengze=True,
    is_keshihua=False,
    is_shoulian=False
):
    """基础训练函数，其他训练函数通过给该函数传递不同参数得到"""
    logging.info(f"dataset: {dataset_name}, miss_ratio: {miss_ratio}, args: {args}")

    logging.info("load complete dataset")
    X, Y, n_cluster, n_sample, n_view = load_dataset(dataset_name)

    logging.info("build incomplete dataset")
    mask = get_mask(n_view, n_sample, miss_ratio, dataset_name)
    X = [X[i] * mask[:, i][:, np.newaxis] for i in range(n_view)]

    logging.info("train without imputation")

    logging.info("construct graph")
    Ls = []
    for i in range(n_view):
        ind_0 = np.where(mask[:, i] == 0)[0]
        Gv = np.diag(mask[:, i])
        Gv = np.delete(Gv, ind_0, axis=1)

        Xv = np.delete(X[i], ind_0, axis=0)
        options = {"NeighborMode": "KNN", "WeightMode": "Binary", "k": args["k"]}
        # 两种构造方式计算出来的相似度矩阵不一致，两次constructW_fast最后指标不一致，两次constructW最后指标一致
        # Wv = constructW_fast(Xv, options)
        way_func_map = {"sklearn": constructW_by_sklearn, "faiss": constructW_by_faiss}
        construct_w_func = way_func_map[args["construct_w_way"]]
        if is_w_exist(dataset_name, i, miss_ratio, args["construct_w_way"]):
            logging.info(f"w exists, load w")
            Wv = load_w(dataset_name, i, miss_ratio, args["construct_w_way"])
        else:
            logging.info(f"w not exists, construct w")
            Wv = construct_w_func(Xv, options)
            save_w(dataset_name, i, miss_ratio, args["construct_w_way"], Wv)
        logging.info(f"construct graph: view {i+1} finish")

        Sv = np.dot(np.dot(Gv, Wv + Wv.T), Gv.T)
        Sum_S = np.sum(Sv, axis=0)
        Lsv = np.diag(Sum_S) - Sv
        Ls.append(Lsv)

    logging.info("prepare train: change data to tensor, build model and optimizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = [torch.from_numpy(x).float().to(device) for x in X]
    mask = torch.from_numpy(mask).float().to(device)
    # 计算torch.trace要求为double
    Ls = [torch.from_numpy(Lsv).to(torch.double).to(device) for Lsv in Ls]
    H = torch.normal(mean=torch.zeros([n_sample, args["h_dim"]]), std=0.01).to(device)
    H.requires_grad_(True)
    model = DegradationModel(args["h_dim"], [x.shape[1] for x in X]).to(device)
    op = optim.Adam(chain(model.parameters(), [H]), lr=args["lr"])

    logging.info("start train")
    shoulian_record=[]
    for epoch in range(args["epochs"]):
        x_re_list = model(H)

        # re_loss = 0
        re_loss = torch.tensor(0.0).to(device)
        if is_tuihua_kehuode:
            for v in range(n_view):
                loss_single = (X[v] - x_re_list[v]) ** 2
                loss_single = (loss_single * mask[:, v][:, np.newaxis]).sum() / mask[
                    :, v
                ].sum()
                re_loss += loss_single

        # tr_loss = 0
        tr_loss = torch.tensor(0.0).to(device)
        if is_tuzhengze:
            for v in range(n_view):
                h_double = H.double()
                xv_tr_loss = torch.trace(
                    torch.matmul(h_double.T, torch.matmul(Ls[v], h_double))
                )
                tr_loss += xv_tr_loss / mask[:, v].sum()

        loss = re_loss + args["lam"] * tr_loss
        loss.backward()
        op.step()
        op.zero_grad()

        if (epoch + 1) % args["print_loss_freq"] == 0:
            logging.info(
                f'Epoch[{epoch+1}/{args["epochs"]}] L: {loss.item():.4f} Lre: {re_loss.item():.4f} Ltr: {tr_loss.item():.4f}'
            )
            if is_shoulian:
                results = cluster(n_cluster, H.detach().cpu().numpy(), Y, count=10)
                shoulian_record.append([epoch+1,loss.item(), results])
    if not is_shoulian:
        results = cluster(n_cluster, H.detach().cpu().numpy(), Y, count=10)
    if is_keshihua:
        return H.detach().cpu().numpy(), Y
    if is_shoulian:
        return shoulian_record
    return results


def train_base_buquan(
    dataset_name,
    miss_ratio,
    args,
    is_imputed_by_mean=False,
    is_imputed_by_neighbor=False,
):
    logging.info(f"dataset: {dataset_name}, miss_ratio: {miss_ratio}, args: {args}")

    logging.info("load complete dataset")
    X, Y, n_cluster, n_sample, n_view = load_dataset(dataset_name)

    logging.info("build incomplete dataset")
    mask = get_mask(n_view, n_sample, miss_ratio, dataset_name)
    X = [X[i] * mask[:, i][:, np.newaxis] for i in range(n_view)]

    if is_imputed_by_mean:
        logging.info("start imputation: by mean")
        X = make_imputation_by_mean(X)
    elif is_imputed_by_neighbor:
        logging.info("start imputation: by cross view neighbor")
        indices = find_neighbor_index(X, args["k_find"])
        make_imputation_near_first(X, mask, indices, args["k"], args["k_find"])
    else:
        raise NotImplementedError(f"the args is wrong")

    logging.info("construct graph")
    Ls = []
    miss_ratio = 0.0
    for i in range(n_view):
        Xv = X[i]
        options = {"NeighborMode": "KNN", "WeightMode": "Binary", "k": args["k"]}
        # 两种构造方式计算出来的相似度矩阵不一致，两次constructW_fast最后指标不一致，两次constructW最后指标一致
        way_func_map = {"sklearn": constructW_by_sklearn, "faiss": constructW_by_faiss}
        construct_w_func = way_func_map[args["construct_w_way"]]
        if is_w_exist(dataset_name, i, miss_ratio, args["construct_w_way"]):
            logging.info(f"w exists, load w")
            Wv = load_w(dataset_name, i, miss_ratio, args["construct_w_way"])
        else:
            logging.info(f"w not exists, construct w")
            Wv = construct_w_func(Xv, options)
            save_w(dataset_name, i, miss_ratio, args["construct_w_way"], Wv)
        logging.info(f"construct graph: view {i+1} finish")

        Sv = Wv + Wv.T
        Sum_S = np.sum(Sv, axis=0)
        Lsv = np.diag(Sum_S) - Sv
        Ls.append(Lsv)

    logging.info("prepare train: change data to tensor, build model and optimizer")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = [torch.from_numpy(x).float().to(device) for x in X]
    mask = torch.from_numpy(mask).float().to(device)
    # 计算torch.trace要求为double
    Ls = [torch.from_numpy(Lsv).to(torch.double).to(device) for Lsv in Ls]
    H = torch.normal(mean=torch.zeros([n_sample, args["h_dim"]]), std=0.01).to(device)
    H.requires_grad_(True)
    model = DegradationModel(args["h_dim"], [x.shape[1] for x in X]).to(device)
    op = optim.Adam(chain(model.parameters(), [H]), lr=args["lr"])

    logging.info("start train")
    for epoch in range(args["epochs"]):
        x_re_list = model(H)

        re_loss = torch.tensor(0.0).to(device)
        for v in range(n_view):
            loss_single = (X[v] - x_re_list[v]) ** 2
            loss_single = loss_single.sum() / n_sample
            re_loss += loss_single

        tr_loss = torch.tensor(0.0).to(device)
        for v in range(n_view):
            h_double = H.double()
            xv_tr_loss = torch.trace(
                torch.matmul(h_double.T, torch.matmul(Ls[v], h_double))
            )
            tr_loss += xv_tr_loss / n_sample

        loss = re_loss + args["lam"] * tr_loss
        loss.backward()
        op.step()
        op.zero_grad()

        if (epoch + 1) % args["print_loss_freq"] == 0:
            logging.info(
                f'Epoch[{epoch+1}/{args["epochs"]}] L: {loss.item():.4f} Lre: {re_loss.item():.4f} Ltr: {tr_loss.item():.4f}'
            )
    results = cluster(n_cluster, H.detach().cpu().numpy(), Y, count=10)
    return results
