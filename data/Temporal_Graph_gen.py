#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import time
import argparse



def gen_data(data, ntr, N):
    '''
    if flag:
        data=pd.read_csv(fname)
    else:
        data=pd.read_csv(fname,header=None)
    '''
    #data=data.as_matrix()
    data=np.reshape(data,[-1,288,N])
    return data[0:ntr]

def normalize(a):
    mu=np.mean(a,axis=1,keepdims=True)
    std=np.std(a,axis=1,keepdims=True)
    return (a-mu)/std

def compute_dtw(a,b,order=1,Ts=12,normal=True):
    if normal:
        a=normalize(a)
        b=normalize(b)
    T0=a.shape[1]
    d=np.reshape(a,[-1,1,T0])-np.reshape(b,[-1,T0,1])
    d=np.linalg.norm(d,axis=0,ord=order)
    D=np.zeros([T0,T0])
    for i in range(T0):
        for j in range(max(0,i-Ts),min(T0,i+Ts+1)):
            if (i==0) and (j==0):
                D[i,j]=d[i,j]**order
                continue
            if (i==0):
                D[i,j]=d[i,j]**order+D[i,j-1]
                continue
            if (j==0):
                D[i,j]=d[i,j]**order+D[i-1,j]
                continue
            if (j==i-Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j])
                continue
            if (j==i+Ts):
                D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i,j-1])
                continue
            D[i,j]=d[i,j]**order+min(D[i-1,j-1],D[i-1,j],D[i,j-1])
    return D[-1,-1]**(1.0/order)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="C:/Users/palli/OneDrive/Desktop/STFGNNfinal/data/temporal_graph_matrix_bicocca.npz", help="Dataset path.")
parser.add_argument("--order", type=int, default=1, help="DTW order.")
parser.add_argument("--lag", type=int, default=12, help="Fast DTW search lag.")
parser.add_argument("--period", type=int, default=288, help="Time series period.")
parser.add_argument("--sparsity", type=float, default=0.01, help="sparsity of spatial graph")

args = parser.parse_args()

# Estrai solo il nome del file dal percorso completo
dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]

# Crea la directory di output se non esiste
output_dir = "./newdataset"
os.makedirs(output_dir, exist_ok=True)

# Carica il file .npz e stampa le chiavi disponibili
npzfile = np.load(args.dataset)
print("Available keys in the dataset:", npzfile.files)


df = np.load(args.dataset)['temporal_graph_matrix']
num_samples, ndim, _ = df.shape
num_train = int(num_samples * 0.6)
num_dtw = int(num_train / args.period) * args.period
data = df[:num_dtw, :, :1].reshape([-1, args.period, ndim])

d = np.zeros([ndim, ndim])

for i in range(ndim):
    t1 = time.time()
    for j in range(i + 1, ndim):
        d[i, j] = compute_dtw(data[:, :, i], data[:, :, j], order=args.order, Ts=args.lag)
    t2 = time.time()
    print('Line', i, 'finished in', t2 - t1, 'seconds.')

dtw = d + d.T

# Salva il risultato con un percorso valido
output_path = os.path.join(output_dir, f"{dataset_name}-dtw-{args.period}-{args.order}.npy")
np.save(output_path, dtw)
print(f"The calculation of time series is done! Saved to {output_path}")

# Carica il risultato
adj = np.load(output_path)
adj = adj+ adj.T

w_adj = np.zeros([ndim,ndim])

adj_percent = args.sparsity

top = int(ndim * adj_percent)
for i in range(adj.shape[0]):
    a = adj[i,:].argsort()[0:top]
    for j in range(top):
        w_adj[i, a[j]] = 1

for i in range(ndim):
    for j in range(ndim):
        if (w_adj[i][j] != w_adj[j][i] and w_adj[i][j] ==0):
            w_adj[i][j] = 1
        if( i==j):
            w_adj[i][j] = 1

print("Total route number: ", ndim)
print("Sparsity of adj: ", len(w_adj.nonzero()[0])/(ndim*ndim))

# Salva il CSV con un nome di file valido
output_csv = os.path.join(output_dir, f"adj_tg_{dataset_name}.csv")
pd.DataFrame(w_adj).to_csv(output_csv, index=False, header=None)

print(f"The weighted matrix of temporal graph is generated and saved to {output_csv}!")