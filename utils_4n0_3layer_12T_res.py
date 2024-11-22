# -*- coding:utf-8 -*-

import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

def construct_model(config, new=0):
    from models.stsgcn_4n_res import STSGCN  # Ensure this is the PyTorch version

    module_type = config['module_type']
    act_type = config['act_type']
    temporal_emb = config['temporal_emb']
    spatial_emb = config['spatial_emb']
    use_mask = config['use_mask']

    num_of_vertices = config['num_of_vertices']
    num_of_features = config['num_of_features']
    points_per_hour = config['points_per_hour']
    num_for_predict = config['num_for_predict']
    adj_filename = config['adj_filename']
    id_filename = config['id_filename']
    if id_filename is not None:
        if not os.path.exists(id_filename):
            id_filename = None

    if new == 1:
        adj = get_adjacency_matrix_new(adj_filename)
    else:
        adj = get_adjacency_matrix(adj_filename, num_of_vertices,
                                   id_filename=id_filename)
    adj_dtw = np.array(pd.read_csv(config['adj_dtw_filename'], header=None))
    adj_mx = construct_adj_fusion(adj, adj_dtw, 4)
    print("The shape of localized adjacency matrix: {}".format(
        adj_mx.shape), flush=True)
    
    print(f"adj shape: {adj.shape}")
    print(f"adj_dtw shape: {adj_dtw.shape}")
    print(f"adj_mx shape: {adj_mx.shape}")
    # Convert adjacency matrix to PyTorch tensor
    adj_mx = torch.tensor(adj_mx, dtype=torch.float32)

    first_layer_embedding_size = config.get('first_layer_embedding_size', None)
    filters = config['filters']

    # Initialize the model
    net = STSGCN(
        num_nodes=num_of_vertices,
        input_dim=num_of_features,
        horizon=points_per_hour,
        num_layers=len(filters),
        filters=filters,
        module_type=module_type,
        activation=act_type,
        use_mask=use_mask,
        adj=adj_mx,
        temporal_emb=temporal_emb,
        spatial_emb=spatial_emb,
        first_layer_embedding_size=first_layer_embedding_size,
        predict_length=config.get('predict_length', 12),
        rho=config.get('rho', 1)
    )

    return net



def get_adjacency_matrix_new(distance_df_filename):
    A = np.load(distance_df_filename, allow_pickle=True)
    for i in range(A.shape[0]):
        A[i, i] = 0
        for j in range(i + 1, A.shape[1]):
            try:
                A[i, j] = 1 / A[i, j]
            except:
                A[i, j] = 0
            A[j, i] = A[i, j]
    return A

def get_adjacency_matrix(distance_df_filename, num_of_vertices,
                         type_='connectivity', id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    type_: str, {connectivity, distance}

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                 dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A

    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A

def construct_adj(A, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: np.ndarray, shape is (N * steps, N * steps)
    '''
    N = len(A)
    adj = np.zeros([N * steps, N * steps])

    for i in range(steps):
        adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj

def construct_adj_fusion(A, A_dtw, steps):
    '''
    construct a bigger adjacency matrix using the given matrix

    Parameters
    ----------
    A: np.ndarray, adjacency matrix, shape is (N, N)

    steps: how many times does the new adj mx bigger than A

    Returns
    ----------
    new adjacency matrix: np.ndarray, shape is (N * steps, N * steps)

    This is 4N_1 mode:

    [T, 1, 1, T
     1, S, 1, 1
     1, 1, S, 1
     T, 1, 1, T]

    '''

    N = len(A)
    print(f"N: {N}")
    print(f"A shape: {A.shape}")
    print(f"A_dtw shape: {A_dtw.shape}")
    adj = np.zeros([N * steps, N * steps])  # "steps" = 4 !!!

    for i in range(steps):
        if (i == 1) or (i == 2):
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A
        else:
            adj[i * N: (i + 1) * N, i * N: (i + 1) * N] = A_dtw

    for i in range(N):
        for k in range(steps - 1):
            adj[k * N + i, (k + 1) * N + i] = 1
            adj[(k + 1) * N + i, k * N + i] = 1

    adj[3 * N: 4 * N, 0:  N] = A_dtw
    adj[0: N, 3 * N: 4 * N] = A_dtw

    adj[2 * N: 3 * N, 0: N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[0: N, 2 * N: 3 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[1 * N: 2 * N, 3 * N: 4 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]
    adj[3 * N: 4 * N, 1 * N: 2 * N] = adj[0 * N: 1 * N, 1 * N: 2 * N]

    for i in range(len(adj)):
        adj[i, i] = 1

    return adj

def generate_from_train_val_test(data, transformer):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y

def generate_from_data(data, length, transformer, isnew):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        if isnew == 1:
            x, y = generate_seq(data[line1: line2], 12, 12)
        else:
            x, y = generate_seq(data['data'][line1: line2], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y

def generate_data(graph_signal_matrix_filename, transformer=None, isnew=0):
    '''
    shape is (num_of_samples, 12, num_of_vertices, 1)
    '''
    data = np.load(graph_signal_matrix_filename, allow_pickle=True)
    if isnew == 1:
        length = data.shape[0]
        for i in generate_from_data(data, length, transformer, isnew):
            yield i
    else:
        keys = data.keys()
        if 'train' in keys and 'val' in keys and 'test' in keys:
            for i in generate_from_train_val_test(data, transformer):
                yield i
        elif 'data' in keys:
            length = data['data'].shape[0]
            for i in generate_from_data(data, length, transformer, isnew):
                yield i
        else:
            raise KeyError("Neither 'data' nor 'train', 'val', 'test' is in the data")

def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(array)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')

def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= np.mean(mask)
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100

def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= np.mean(mask)
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))

def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= np.mean(mask)
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))
