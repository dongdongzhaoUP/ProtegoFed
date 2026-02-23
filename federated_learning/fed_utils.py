import torch
import copy 
import random
import math
from hdbscan import HDBSCAN
from scipy.fftpack import dct, idct
import numpy as np
def get_proxy_dict(fed_args, global_dict):
    opt_proxy_dict = None
    proxy_dict = None
    if fed_args.fed_alg in ['fedadagrad', 'fedyogi', 'fedadam']:
        proxy_dict, opt_proxy_dict = {}, {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
            opt_proxy_dict[key] = torch.ones_like(global_dict[key]) * fed_args.fedopt_tau**2
    elif fed_args.fed_alg == 'fedavgm':
        proxy_dict = {}
        for key in global_dict.keys():
            proxy_dict[key] = torch.zeros_like(global_dict[key])
    return proxy_dict, opt_proxy_dict

def get_clients_this_round(config, round):

    if config['sample_ratio']>=1:
        clients_this_round = list(range(config['num_clients']))
    else:
        random.seed(round)
        clients_this_round = sorted(random.sample(range(config['num_clients']), int(config['num_clients']*config['sample_ratio'])))
    return clients_this_round

def global_aggregate(global_dict, local_dict_list, sample_num_list, clients_this_round):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
 
    for key in global_dict.keys():
        global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])

    return global_dict

def gaussian_noise(data_shape, fed_args, script_args, device):
    if script_args.dp_sigma is None:
        delta_l = 2 * script_args.learning_rate * script_args.dp_max_grad_norm / (script_args.dataset_sample / fed_args.num_clients)
        q = fed_args.sample_clients / fed_args.num_clients
        sigma = delta_l * math.sqrt(2*q*fed_args.num_rounds*math.log(1/script_args.dp_delta)) / script_args.dp_epsilon
    else:
        sigma = script_args.dp_sigma
    return torch.normal(0, sigma, data_shape).to(device)

def tensorDCT(weight_tensor):
    weight_tensor_cpu = weight_tensor.cpu()
    dct_rows = torch.tensor([dct(row.detach().numpy(), type=2, norm='ortho') for row in weight_tensor_cpu])
    dct_matrix = torch.tensor([dct(col.detach().numpy(), type=2, norm='ortho') for col in dct_rows.T]).T

    return dct_matrix
    
def filtering(V):
    F=[]
    height,width=V.shape
    subV=V[:int(height/2),:int(width/2)]
    for i in range(int(height/2)):
        for j in range(int(width/2)):
            if i+j<= int((height+width)/4):
                F.append(subV[i][j])
    
    F_tensor=torch.tensor(F)
    return F_tensor

def clustering(F_list):
    K = len(F_list)
    
    distances_matrix = torch.zeros((K, K), dtype=torch.float64)
    
    for i in range(K):
        for j in range(K):
            distances_matrix[i, j] = 1 - torch.nn.functional.cosine_similarity(F_list[i].unsqueeze(0),F_list[j].unsqueeze(0))
            distances_matrix[j, i] = distances_matrix[i, j]
    
    print("距离矩阵：")
    print(distances_matrix.numpy())
    
    distances_matrix_np = distances_matrix.numpy()

    clusterer = HDBSCAN(metric="precomputed", min_cluster_size=2, min_samples=1)
    
    cluster_ids = clusterer.fit_predict(distances_matrix_np)
    
    unique_clusters, counts = np.unique(cluster_ids, return_counts=True)
    
    if len(unique_clusters) == 0:
        return set()
    
    max_cluster = unique_clusters[np.argmax(counts)]
    
    B = set()
    for i in range(K):
        if cluster_ids[i] == max_cluster:
            B.add(i)
            
    return B,cluster_ids