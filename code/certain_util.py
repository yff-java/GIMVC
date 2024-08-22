import numpy as np
import copy
from scipy.sparse import issparse
from sklearn.cluster import SpectralClustering
import torch
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import logging

def make_imputation_by_mean(data_list):
    # data_list: VxNxD
    # mask: NxV
    complete_data_list = []
    for incomplete_data in data_list:
        complete_data = copy.deepcopy(incomplete_data)
        # 确定缺失样本的位置
        missing_rows = np.all(complete_data == 0, axis=1)

        # 计算整个数据集的均值
        dataset_mean = np.mean(complete_data[complete_data != 0])

        # 使用整个数据集的均值填补缺失样本的行
        complete_data[missing_rows] = dataset_mean
        complete_data_list.append(complete_data)
    return complete_data_list


def find_neighbor_index(X, k):
    import faiss

    # indices:(n_view, n_sample, k)
    n_view = len(X)
    indices = []
    for v in range(n_view):
        fea = np.ascontiguousarray(X[v].astype(np.float32))
        n, dim = fea.shape[0], fea.shape[1]
        # 找到最近的topk个邻居的下标
        index = faiss.IndexFlatIP(dim)
        # index.add(np.ascontiguousarray(fea.astype(np.float32)))
        index.add(fea)
        # Sample itself is included
        _, ind = index.search(fea, k + 1)
        indices.append(ind[:, 1:])
    return indices


def make_imputation_near_first(data_list, mask, indices, k, k2):
    # 按照论文中的方法补全缺失视角
    # 如果寻找邻居时只寻找k个邻居的话，由于缺失，可能最后的邻居数不足k个
    n_view = len(data_list)
    data_size = data_list[0].shape[0]
    for v in range(n_view):
        for i in range(data_size):
            if mask[i, v] == 0:
                # print(f"view {v} miss sample {i}")
                predicts = []
                neighbor_index = []
                is_finished = False
                for n_w in range(k2):
                    for w in range(n_view):
                        if w != v and mask[i, w] != 0:
                            neigh_w = indices[w][i]
                            if (
                                mask[neigh_w[n_w], v] != 0
                                and mask[neigh_w[n_w], w] != 0
                            ):
                                # 保证邻居不会重复
                                if neigh_w[n_w] not in neighbor_index:
                                    neighbor_index.append(neigh_w[n_w])
                                    predicts.append(data_list[v][neigh_w[n_w]])
                                    # print(v, i, w, neigh_w[n_w])
                                    # 保证邻居数不会超过k
                                    if len(predicts) >= k:
                                        is_finished = True
                                        break
                    if is_finished:
                        break

                # for w in range(n_view):
                #     # only the available views are selected as neighbors
                #     if w != v and mask[i, w] != 0:
                #         neigh_w = indices[w][i]
                #         for n_w in range(neigh_w.shape[0]):
                #             if mask[neigh_w[n_w], v] != 0 and mask[neigh_w[n_w], w] != 0:
                #                 predicts.append(data_list[v][neigh_w[n_w]])
                #                 print(v,i,w,neigh_w[n_w])
                # if len(predicts) >= args.K:
                #     break

                # assert len(predicts) >= args.K
                # print("neighbor num: ", len(neighbor_index),
                #       " neighbor_index: ", neighbor_index)
                # print()
                fill_sample = np.mean(predicts, axis=0)
                data_list[v][i] = fill_sample


def construct_similarity_matrix(X, ind_folds, num_view):
    # 参考论文：Adaptive Graph Completion Based Incomplete Multi-View Clustering
    # 先构造可获得样本的相似度矩阵S，再计算所有样本的相似度矩阵GSG^T
    S_part = []
    S_all = []
    for iv in range(num_view):
        # X1是可获得样本
        # N * Dv
        X1 = X[iv]
        # X1 = NormalizeFea(X1, 0)
        ind_0 = np.where(ind_folds[:, iv] == 0)[0]
        # Nv * Dv
        X1 = np.delete(X1, ind_0, axis=0)
        X1 = NormalizeFea(X1, 1)

        # So是可获得样本的knn图
        options = {"NeighborMode": "KNN", "k": 11, "WeightMode": "HeatKernel"}
        # w: Nv * Nv
        w = constructW_by_sklearn(X1, options)
        S_part.append(w)

        # Sor是所有样本的knn图
        # G: N * N
        G = np.diag(ind_folds[:, iv].astype(np.int))
        # G: N * Nv
        G = np.delete(G, ind_0, axis=1)
        # Sor: N * N
        S_all.append(G.dot(S_part[-1]).dot(G.T))
    return S_part, S_all


def construct_similarity_matrix_whole(X, ind_folds, num_view, neighbor):
    # 参考论文：Adaptive Graph Completion Based Incomplete Multi-View Clustering
    # 先构造可获得样本的相似度矩阵S，再计算所有样本的相似度矩阵GSG^T
    S_part_raw_mv = []
    S_part_sys_mv = []
    S_part_lap_mv = []

    S_all_raw_mv = []
    S_all_sys_mv = []
    S_all_lap_mv = []
    for iv in range(num_view):
        # X1是可获得样本
        # N * Dv
        X1 = X[iv]
        # X1 = NormalizeFea(X1, 0)
        ind_0 = np.where(ind_folds[:, iv] == 0)[0]
        # Nv * Dv
        X1 = np.delete(X1, ind_0, axis=0)
        X1 = NormalizeFea(X1, 1)

        # So是可获得样本的knn图
        # options = {"NeighborMode": "KNN", "k": neighbor, "WeightMode": "HeatKernel"}
        # # w: Nv * Nv
        # w = constructW(X1, options)
        X1 = torch.from_numpy(X1).float()
        we, raw_we = cal_weights_via_CAN(X1.t(), neighbor)
        w = raw_we.cpu().numpy()
        S_part_raw_mv.append(w)

        S_part_sys = (w + w.T) / 2
        S_part_sys_mv.append(S_part_sys)

        degree = np.sum(S_part_sys, axis=1)
        degree = np.power(degree, -0.5)
        S_part_lap_mv.append((S_part_sys * degree).T * degree)

        # Sor是所有样本的knn图
        # G: N * N
        G = np.diag(ind_folds[:, iv].astype(np.int))
        # G: N * Nv
        G = np.delete(G, ind_0, axis=1)
        # Sor: N * N
        S_all_raw_mv.append(G.dot(S_part_raw_mv[-1]).dot(G.T))
        S_all_sys_mv.append(G.dot(S_part_sys_mv[-1]).dot(G.T))
        S_all_lap_mv.append(G.dot(S_part_lap_mv[-1]).dot(G.T))
    return (
        S_part_raw_mv,
        S_part_sys_mv,
        S_part_lap_mv,
        S_all_raw_mv,
        S_all_sys_mv,
        S_all_lap_mv,
    )


def NormalizeFea(fea, row=1):
    """
    If row == 1, normalize each row of fea to have unit norm;
    If row == 0, normalize each column of fea to have unit norm.

    Args:
    - fea: numpy array or sparse matrix, input matrix
    - row: int, indicate which dimension to normalize

    Returns:
    - fea: numpy array or sparse matrix, output matrix with normalized rows or columns
    """
    if row == 1:
        nSmp = fea.shape[0]
        feaNorm = np.maximum(1e-14, np.sum(fea**2, axis=1))
        feaNorm = np.sqrt(feaNorm)
        if issparse(fea):
            fea = fea.transpose()
            for i in range(nSmp):
                fea[:, i] = fea[:, i] / np.maximum(1e-10, feaNorm[i])
            fea = fea.transpose()
        else:
            fea = fea / feaNorm[:, np.newaxis]
    else:
        nSmp = fea.shape[1]
        feaNorm = np.maximum(1e-14, np.sum(fea**2, axis=0)).transpose()
        feaNorm = np.sqrt(feaNorm)
        if issparse(fea):
            for i in range(nSmp):
                fea[:, i] = fea[:, i] / np.maximum(1e-10, feaNorm[i])
        else:
            fea = fea / feaNorm[np.newaxis, :]
    return fea


def constructW_by_sklearn(X, options):
    """
    构造样本数据的权重矩阵

    参数：
    - X：numpy数组，输入数据矩阵，形状为 (n_samples, n_features)
    - options：字典类型，包含以下选项：
        - NeighborMode：字符串类型，邻居模式，取值为 'KNN' 或者 'Supervised'
        - k：整数型，KNN 模式下的最近邻个数
        - WeightMode：字符串类型，权重模式，取值为 'Binary' 或者 'HeatKernel'

    返回：
    - Z：numpy数组，权重矩阵，形状为 (n_samples, n_samples)
    """
    logging.info("constructW_by_sklearn")
    # 计算样本间距离矩阵
    dist = cdist(X, X, metric="euclidean")

    # 根据邻居模式创建邻居索引矩阵
    if options["NeighborMode"] == "KNN":
        knn = min(options["k"] + 1, X.shape[0])
        nbrs = NearestNeighbors(n_neighbors=knn, algorithm="auto").fit(X)
        _, indices = nbrs.kneighbors(X)
    elif options["NeighborMode"] == "Supervised":
        raise NotImplementedError("Not implemented yet")
    else:
        raise ValueError("Unknown neighbor mode %s" % options["NeighborMode"])

    # 根据权重模式计算权重矩阵
    if options["WeightMode"] == "Binary":
        Z = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            Z[i, indices[i, 1:]] = 1
    elif options["WeightMode"] == "HeatKernel":
        if "t" not in options:
            # nSmp = X.shape[0]
            # if nSmp > 3000:
            #     # EuDist2 函数的 Python 实现
            #     sub_fea = X[np.random.choice(nSmp, 3000), :]
            #     D = cdist(sub_fea, sub_fea)
            # else:
            #     D = cdist(X, X)
            options["t"] = np.mean(dist)
        t = options["t"]
        # print("t: ", t)
        Z = np.exp(-(dist**2) / (2 * t**2))
    else:
        raise ValueError("Unknown weight mode %s" % options["WeightMode"])

    return Z


def constructW_by_faiss(X, options):
    import faiss
    logging.info("constructW_by_faiss")
    k = options["k"]

    if options["NeighborMode"] == "KNN":
        fea = np.ascontiguousarray(X.astype(np.float32))
        n, dim = fea.shape[0], fea.shape[1]
        # 找到最近的topk个邻居的下标
        index = faiss.IndexFlatIP(dim)
        # index.add(np.ascontiguousarray(fea.astype(np.float32)))
        index.add(fea)
        # Sample itself is included
        _, ind = index.search(fea, k + 1)
    else:
        raise NotImplementedError(f"unknown NeighborMode: { options['NeighborMode']}")

    if options["WeightMode"] == "Binary":
        Z = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            Z[i, ind[i, 1:]] = 1
    else:
        raise NotImplementedError(f"unknown WeightMode: { options['WeightMode']}")
    return Z


def constructW_by_hnswlib(X, options):
    """
    构建相似度矩阵W
    :param X: 数据矩阵 [n_samples, n_features]
    :param options: 参数字典, 包括 NeighborMode, WeightMode, k 三个参数
        - NeighborMode: 近邻模式, {'KNN', 'Supervised'}
        - WeightMode: 权重计算方式, {'Binary', 'HeatKernel', 'Cosine', 'Euclidean'}
        - k: 近邻数目
    :return: 相似度矩阵W [n_samples, n_samples]
    """
    import hnswlib

    # 获取参数
    NeighborMode = options["NeighborMode"]
    WeightMode = options["WeightMode"]
    k = options["k"]

    # 计算相似度矩阵
    if NeighborMode == "KNN":
        # 使用KNN方法构建相似度矩阵
        num_threads = 4  # 线程数
        M = 16  # 质心数量
        ef_construction = 100  # hnsw 算法参数
        ef_search = 50  # hnsw 算法参数
        dim = X.shape[1]  # 数据维度
        p = hnswlib.Index(space="l2", dim=dim)  # 创建 HNSW 实例
        # p.ef_search_seed  =1234
        # 初始化 HNSW 实例
        p.init_index(
            max_elements=X.shape[0],
            ef_construction=ef_construction,
            M=M,
            random_seed=1234,
        )
        p.add_items(X)  # 添加数据点到 HNSW 实例中
        p.set_ef(ef_search)  # 设置 ef_search 参数，加速搜索
        # 搜索 KNN，因为最近邻一定是自己，所以要搜索 k+1 个近邻
        labels, distances = p.knn_query(X, k=k + 1)
        labels = labels[:, 1:]  # 去掉自己和自己的相似度
        distances = distances[:, 1:]
        W = np.zeros([X.shape[0], X.shape[0]], dtype=np.int32)
        for i in range(X.shape[0]):
            inds = labels[i]
            W[i, inds] = 1  # 将距离转换为二值相似度
    else:
        raise ValueError("Unknown NeighborMode: %s" % NeighborMode)

    # 计算权重
    if WeightMode == "Binary":
        # 二进制权重
        W[W > 0] = 1
    elif WeightMode == "HeatKernel":
        # 热核权重
        sigma = np.median(distances)  # 根据距离矩阵的中位数计算sigma
        W = np.exp(-(distances**2) / (2 * sigma**2))
    elif WeightMode == "Cosine":
        # 余弦相似度
        D = np.diag(1.0 / np.sqrt(np.sum(W, axis=1)))  # 计算度矩阵D
        W = np.dot(D, np.dot(W, D))
    elif WeightMode == "Euclidean":
        # 欧氏距离相似度
        W = np.exp(-(distances**2) / 2)
    else:
        raise ValueError("Unknown WeightMode: %s" % WeightMode)

    return W


def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    return result


def cal_weights_via_CAN(X, num_neighbors, links=0):
    """
    用论文中的方法计算数据点的权重，距离越近、权重越大
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    # top_k是每个样本和它最远邻居的距离
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    # sum_top_k是每个样本和它前num_neighbors个邻居的距离之和
    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))

    sorted_distances = None
    torch.cuda.empty_cache()
    # T[i,j] 表示第 i 个数据点与第 j 个数据点之间的接近程度（权重），离得越近，T值越大
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    # weights相当于对T做归一化
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).cuda()
        weights += torch.eye(size).cuda()
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.cuda()
    weights = weights.cuda()
    return weights, raw_weights


def cluster_by_multi_ways(z_list, L_list, Y, n_cluster, fusion_kind, view_index=0):
    from metrics import cluster
    import numpy as np

    if fusion_kind == "pinjiezv":
        z = z_list[view_index].detach().cpu().numpy()
        cluster(n_cluster, z, Y, desc=f"X{view_index}")

        z = torch.hstack(z_list).detach().cpu().numpy()
        cluster(n_cluster, z, Y, desc=fusion_kind)

    elif fusion_kind == "pingjunzv":
        z = z_list[view_index].detach().cpu().numpy()
        cluster(n_cluster, z, Y, desc=f"X{view_index}")

        array_list = [zv.detach().cpu().numpy() for zv in z_list]
        z = np.mean(array_list, axis=0)
        cluster(n_cluster, z, Y, desc=fusion_kind)

    elif fusion_kind == "pingjunlvpujulei":
        Lv = L_list[view_index]
        _, vectors = torch.linalg.eigh(Lv, UPLO="U")
        indicator = vectors[:, -n_cluster:]
        indicator = (
            indicator / (indicator.norm(dim=1) + 10**-10).repeat(n_cluster, 1).t()
        )
        indicator = indicator.cpu().numpy()
        cluster(n_cluster, indicator, Y, desc=f"X{view_index}")

        L = torch.mean(torch.stack(L_list), dim=0)
        _, vectors = torch.linalg.eigh(L, UPLO="U")
        indicator = vectors[:, -n_cluster:]
        indicator = (
            indicator / (indicator.norm(dim=1) + 10**-10).repeat(n_cluster, 1).t()
        )
        indicator = indicator.cpu().numpy()
        cluster(n_cluster, indicator, Y, desc=fusion_kind)

    elif fusion_kind == "lvxzvpinjie":
        new_z = torch.matmul(L_list[view_index], z_list[view_index])
        new_z = new_z.detach().cpu().numpy()
        cluster(n_cluster, new_z, Y, desc=f"X{view_index}")

        new_z_list = [torch.matmul(lv, zv) for lv, zv in zip(L_list, z_list)]
        z = torch.hstack(new_z_list)
        z = z.detach().cpu().numpy()
        cluster(n_cluster, z, Y, desc=fusion_kind)
    elif fusion_kind == "lvxzvpingjun":
        new_z = torch.matmul(L_list[view_index], z_list[view_index])
        new_z = new_z.detach().cpu().numpy()
        cluster(n_cluster, new_z, Y, desc=f"X{view_index}")

        new_z_list = [torch.matmul(lv, zv) for lv, zv in zip(L_list, z_list)]
        z = torch.mean(torch.stack(new_z_list), dim=0)
        z = z.detach().cpu().numpy()
        cluster(n_cluster, z, Y, desc=fusion_kind)

    elif fusion_kind == "pinjiezv_pingjunlv_lxz":
        new_z = torch.matmul(L_list[view_index], z_list[view_index])
        new_z = new_z.detach().cpu().numpy()
        cluster(n_cluster, new_z, Y, desc=f"X{view_index}")

        z = torch.hstack(z_list)
        L = torch.mean(torch.stack(L_list), dim=0)
        new_z = torch.matmul(L, z).detach().cpu().numpy()
        cluster(n_cluster, new_z, Y, desc=fusion_kind)

    elif fusion_kind == "pinjiezv_zxzt_lxz":
        zv = z_list[view_index]
        lv = torch.matmul(zv, zv.t())
        new_zv = torch.matmul(lv, zv).detach().cpu().numpy()
        cluster(n_cluster, new_zv, Y, desc=f"X{view_index}")

        z = torch.hstack(z_list)
        L = torch.matmul(z, z.t())
        new_z = torch.matmul(L, z).detach().cpu().numpy()
        cluster(n_cluster, new_z, Y, desc=fusion_kind)

    elif fusion_kind == "pingjunzv_pingjunlv_lxz":
        zv = z_list[view_index]
        lv = L_list[view_index]
        new_zv = torch.matmul(lv, zv)
        new_zv = new_zv.detach().cpu().numpy()
        cluster(n_cluster, new_zv, Y, desc=f"X{view_index}")

        z = torch.mean(torch.stack(z_list), dim=0)
        L = torch.mean(torch.stack(L_list), dim=0)
        new_z = torch.matmul(L, z).detach().cpu().numpy()
        cluster(n_cluster, new_z, Y, desc=fusion_kind)

    elif fusion_kind == "spectral":
        lv = L_list[view_index].detach().cpu().numpy()
        sc_model = SpectralClustering(
            n_clusters=n_cluster, affinity="precomputed", eigen_solver="arpack"
        )
        y_pred = sc_model.fit_predict(lv)
        cluster_by_assignment(
            torch.tensor(y_pred),
            Y,
            1,
            desc=f"spectral X{view_index}",
            assignment_is_y_pred=True,
        )

        L = torch.mean(torch.stack(L_list), dim=0).detach().cpu().numpy()
        sc_model = SpectralClustering(
            n_clusters=n_cluster, affinity="precomputed", eigen_solver="arpack"
        )
        y_pred = sc_model.fit_predict(L)
        cluster_by_assignment(
            torch.tensor(y_pred),
            Y,
            1,
            desc=f"spectral fusion",
            assignment_is_y_pred=True,
        )
    else:
        raise ValueError("fusion_kind error")


def cluster_by_assignment(assignment, Y, count, desc, assignment_is_y_pred=False):
    from metrics import get_avg_matric
    import numpy as np
    import logging

    if not assignment_is_y_pred:
        y_pred = assignment.detach().cpu().numpy()
        y_pred = y_pred.argmax(axis=1)
    else:
        y_pred = assignment.detach().cpu().numpy()
    gt = np.reshape(Y, np.shape(y_pred))
    if np.min(gt) == 1:
        gt -= 1
    acc_avg, acc_std, nmi_avg, nmi_std, RI_avg, RI_std, f1_avg, f1_std = get_avg_matric(
        gt, [y_pred], count
    )
    logging.info(
        f"{desc} Kmeans({count} times average) ACC={acc_avg:.2f}±{acc_std:.2f}, NMI={nmi_avg:.2f}±{nmi_std:.2f},"
        f"RI={RI_avg:.2f}±{RI_std:.2f}, f1={f1_avg:.2f}±{f1_std:.2f}"
    )


def test_constructW_speed():
    from data_process import load_dataset
    from datetime import datetime
    from common_util import prepare_log
    import logging
    from constant import dataset_list

    prepare_log(log_kind=1)
    for dataset_name in dataset_list[5:6]:
        X, Y, n_cluster, n_sample, n_view = load_dataset(dataset_name)

        logging.info("construct start")
        start = datetime.now()
        options = {"NeighborMode": "KNN", "WeightMode": "Binary", "k": 10}

        # w1 = constructW_by_sklearn(X[0], options)
        # w2 = constructW_by_sklearn(X[0], options)
        w1 = constructW_by_faiss(X[0], options)
        w2 = constructW_by_faiss(X[0], options)

        logging.info(f"w1: {w1}")
        logging.info(f"w2: {w2}")
        logging.info(f"w1.shape: {w1.shape}, w2.shape: {w2.shape}")
        logging.info(f"w1==w2: {(w1==w2).all()}")

        logging.info("construct end")
        cost_seconds = (datetime.now() - start).total_seconds()
        logging.info(
            f"dataset: {dataset_name}, n_sample: {n_sample}, dim: {X[0].shape[1]}, seconds: {cost_seconds}"
        )


if __name__ == "__main__":
    test_constructW_speed()
