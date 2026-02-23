import copy
from importlib.metadata import distribution
import logging
from math import log
import select

from scipy import cluster
from sklearn import metrics
from sympy import farthest_points
from .defender import Defender
from openbackdoor.victims import CasualLLMVictim
from openbackdoor.data import getCasualDataloader
from openbackdoor.utils import logger
from typing import *
from torch.utils.data import DataLoader
import random
import numpy as np
import pandas as pd
import torch
from umap import UMAP
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from torch_dct import dct_2d
from sklearn.cluster import AgglomerativeClustering, KMeans,HDBSCAN
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, silhouette_score,confusion_matrix
from scipy.spatial.distance import cdist,cosine
import json
import os
from datetime import datetime
import pickle
from torch import autograd, rand
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.colors as mcolors
import hdbscan
css4_colors=mcolors.CSS4_COLORS



class ProtegoFedDefender(Defender):
    name = "protegofed"
    def __init__(
        self,
        targetPara:Optional[str]="lm_head.weight",
        targetDataset:Optional[str] = "webqa",
        pcaRank:Optional[int]=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.pre = True
        self.targetPara = targetPara
        self.targetDataset = targetDataset
        self.pcaRank = pcaRank
        self.visPath = os.path.join(
            './protegofed/', 
            targetDataset,
            str(datetime.fromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%d-%H-%M-%S'))
        )
        os.makedirs(self.visPath, exist_ok=True)
    
    def correct(
        self, 
        poison_data: List,
        clean_data: Optional[List] = None,
        model: Optional[CasualLLMVictim] = None,
        client_id:int=None,
    ):
        # Step 1. Feature Representation
        embeddings, poisonLabels = self.encode(poison_data, model)
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )
        embUmap = umap.fit(embeddings).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        hdbscan_predLabels = self.clustering_hdbscan(lowrankEmb)
        agglomerative_predLabels = self.clustering_agglomerative(lowrankEmb)
    
        predLabels = hdbscan_predLabels
        try:
            silhouetteScore_hdbscan = silhouette_score(lowrankEmb, hdbscan_predLabels)
            silhouetteScore_agglomerative = silhouette_score(lowrankEmb, agglomerative_predLabels)
            if silhouetteScore_hdbscan>silhouetteScore_agglomerative:
                predLabels = hdbscan_predLabels
                logger.info(f'using hdbscan clustering')
            else:
                predLabels = agglomerative_predLabels
                logger.info(f'using agglomerative clustering')
            logger.info(f'initial silhouette score of {client_id}: {silhouetteScore_hdbscan:.4f} {silhouetteScore_agglomerative:.4f}')
            if min(silhouetteScore_hdbscan,silhouetteScore_agglomerative)<0.2 and max(silhouetteScore_hdbscan,silhouetteScore_agglomerative)<0.4:
                predLabels[:]=0
        except:
            logger.info(f'initial silhouette score of {client_id}: Only one label in the cluster') 
        print('poisonLabels.shape',poisonLabels.shape)
        print('predLabels.shape',predLabels.shape)
        
        local_metrics=self.calculate_metrics(poisonLabels,predLabels)
        logger.info(json.dumps(local_metrics, indent=4))
        
        # plot hdbscan 
        
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        cleanIdx, poisonIdx = np.where(poisonLabels == 0)[0], np.where(poisonLabels == 1)[0]
        plt.scatter(lowrankEmb[cleanIdx, 0], lowrankEmb[cleanIdx, 1], edgecolors="blue", facecolors='none', s=15, label="clean")
        plt.scatter(lowrankEmb[poisonIdx,0], lowrankEmb[poisonIdx, 1], s=10, c="red", label='poison', marker='x')
        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 2, 2)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:, 0], lowrankEmb[:, 1], s=10, c=hdbscan_predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        logger.info(f'saving figure to {self.visPath}')
        plt.savefig(os.path.join(self.visPath, f'{client_id}_hdbscan_visDefense.png'), dpi=600)
        plt.close()
        # plot agglomerative clustering
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        cleanIdx, poisonIdx = np.where(poisonLabels == 0)[0], np.where(poisonLabels == 1)[0]
        plt.scatter(lowrankEmb[cleanIdx, 0], lowrankEmb[cleanIdx, 1], edgecolors="blue", facecolors='none', s=15, label="clean")
        plt.scatter(lowrankEmb[poisonIdx,0], lowrankEmb[poisonIdx, 1], s=10, c="red", label='poison', marker='x')
        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 2, 2)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:, 0], lowrankEmb[:, 1], s=10, c=agglomerative_predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        logger.info(f'saving figure to {self.visPath}')
        plt.savefig(os.path.join(self.visPath, f'{client_id}_agglomerative_visDefense.png'), dpi=600)
        plt.close()
        
        plotData = {
            "emb":lowrankEmb,
            "poisonLabel":poisonLabels,
            "predLabel":predLabels
        }
        with open(os.path.join(self.visPath, f'{client_id}_plotData.pkl'), "wb") as f:
            pickle.dump(plotData, f)
        
        # Step 3. Filtering
        filteredDataset = self.filtering(poison_data, predLabels, poisonLabels)

        return filteredDataset,local_metrics

    def plot_global(self,embeddings,poisonLabels,predLabels,local_num,local_centroids,global_centroid):
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )
        n_clients=len(local_num)
        start_poi=embeddings.shape[0]
        local_len=local_centroids.shape[0]
        
        extend_embeddings=np.vstack((embeddings,local_centroids,global_centroid))
        
        embUmap = umap.fit(extend_embeddings).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        
        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        cleanIdx, poisonIdx = np.where(poisonLabels == 0)[0], np.where(poisonLabels == 1)[0]
        plt.scatter(lowrankEmb[cleanIdx, 0], lowrankEmb[cleanIdx, 1], edgecolors="blue", facecolors='none', s=15, label="clean")
        plt.scatter(lowrankEmb[poisonIdx,0], lowrankEmb[poisonIdx, 1], s=10, c="red", label='poison', marker='x')
        plt.scatter(lowrankEmb[start_poi:start_poi+local_len,0], lowrankEmb[start_poi:start_poi+local_len, 1], s=50, c="purple", label='local', marker='*')
        plt.scatter(lowrankEmb[-1,0], lowrankEmb[-1, 1], s=50, c="orange", label='global', marker='h')
        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 3, 2)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:start_poi, 0], lowrankEmb[:start_poi, 1], s=10, c=predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 3, 3)

        colors=css4_colors
        cmap = ListedColormap(colors)
        start=0
        end=0
        color_list=[]

        for i in range(n_clients):
            color_list.extend(np.full(local_num[i],i))
        plt.scatter(lowrankEmb[:start_poi, 0], lowrankEmb[:start_poi, 1], s=10, c=color_list, cmap=cmap, marker='o')

        handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), label=f'client{i}') for i in range(n_clients)]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        
        logger.info(f'saving figure to {self.visPath}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visPath, 'global_visDefense.pdf'))
        plt.savefig(os.path.join(self.visPath, 'global_visDefense.png'), dpi=600)
        plt.close()
        
        silhouetteScore = silhouette_score(lowrankEmb[:start_poi], predLabels)

    def plot_local(self,embeddings,poisonLabels,revise_predLabels,original_predLabels,client_id):
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )
        
        embUmap = umap.fit(embeddings).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        
        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        cleanIdx, poisonIdx = np.where(poisonLabels == 0)[0], np.where(poisonLabels == 1)[0]
        plt.scatter(lowrankEmb[cleanIdx, 0], lowrankEmb[cleanIdx, 1], edgecolors="blue", facecolors='none', s=15, label="clean")
        plt.scatter(lowrankEmb[poisonIdx,0], lowrankEmb[poisonIdx, 1], s=10, c="red", label='poison', marker='x')
        plt.scatter(lowrankEmb[-1,0], lowrankEmb[-1, 1], s=50, c="purple", label='local', marker='*')
        plt.tick_params(labelsize='large', length=2)
        plt.legend(fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 3, 2)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:, 0], lowrankEmb[:, 1], s=10, c=revise_predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='revise pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='revise pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        plt.subplot(1, 3, 3)
        cmap = ListedColormap(['blue', 'red'])
        plt.scatter(lowrankEmb[:-1, 0], lowrankEmb[:-1, 1], s=10, c=original_predLabels, cmap=cmap, marker='o')

        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0), label='pred clean'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(1), label='pred poison')
        ]
        plt.legend(handles=handles, fontsize=14, markerscale=5, loc='lower right')
        
        logger.info(f'saving figure to {self.visPath}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.visPath, f'revise_{client_id}_visDefense.pdf'))
        plt.savefig(os.path.join(self.visPath, f'revise_{client_id}_visDefense.png'), dpi=600)
        plt.close()
        
 
    def calculate_metrics(self,true_labels, pred_labels):
        num=true_labels.shape[0]
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_labels,labels=[0,1]).ravel()
        
        # Calculate other metrics
        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, zero_division=0)
        recall = recall_score(true_labels, pred_labels, zero_division=0)
        f1 = f1_score(true_labels, pred_labels, zero_division=0)
        
        metrics_dict={
            'num':int(num),
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Accuracy": float(accuracy),
            "Precision": float(precision),
            "Recall": float(recall),
            "F1": float(f1)
        }
        # Return all metrics
        return metrics_dict
            
    def local_centroid(
        self, 
        poison_data: List,
        model: Optional[CasualLLMVictim] = None,
    ):
        print('cal local_centroid')
        embeddings, poisonLabels,dctgrads = self.encode(poison_data, model,fl=True)
        
        if 'lm_head.weight' in self.targetPara:
            predLabels = self.clustering(embeddings)
            local_metrics=self.calculate_metrics(poisonLabels,predLabels)
            logger.info(json.dumps(local_metrics, indent=4))
            umap = UMAP( 
                n_neighbors=100, 
                min_dist=0,
                n_components=2,
                random_state=42,
                transform_seed=42,
                metric="cosine"
            )
            embUmap = umap.fit(embeddings).embedding_
            lowrankEmb = StandardScaler().fit_transform(embUmap)
        else:
            umap = UMAP( 
                n_neighbors=100, 
                min_dist=0,
                n_components=2,
                random_state=42,
                transform_seed=42,
                metric="cosine"
            )
            embUmap = umap.fit(embeddings).embedding_
            lowrankEmb = StandardScaler().fit_transform(embUmap)
            predLabels = self.clustering_hdbscan(lowrankEmb)
            local_metrics=self.calculate_metrics(poisonLabels,predLabels)
            logger.info(json.dumps(local_metrics, indent=4))
        
        
        
        clean_index=np.where( poisonLabels== 0)[0].tolist()
        poison_index=np.where( poisonLabels== 1)[0].tolist()
        
        main_index=np.where(predLabels == 0)[0].tolist()
        sub_index=np.where(predLabels == 1)[0].tolist()
        local_metrics=self.calculate_metrics(poisonLabels,predLabels)
        

        main_cluster=embeddings[main_index]
        
        main_centroid=np.mean(main_cluster,axis=0)
        main_centroid=main_centroid.reshape(1,-1)
        print('centroid.shape',main_centroid.shape)
        
        return main_centroid,embeddings,poisonLabels,predLabels,local_metrics
    

    def global_centroid(
        self, 
        local_centroids: List,
    ):
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )

        print('local_centroids.shape',local_centroids.shape)
        embUmap = umap.fit(local_centroids).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        predLabels = self.clustering_hdbscan(lowrankEmb)
        main_index=np.where(predLabels == 0)[0].tolist()
        main_cluster=local_centroids[main_index]
        centroid=np.mean(main_cluster,axis=0)
        centroid=centroid.reshape(1,-1)
        print('global centroid.shape',centroid.shape)
        
        return centroid
    
    def local_recluster(self,embeddings,poisonLabels,centroid,client_id):
        start_poi=embeddings.shape[0]
        extend_embeddings=np.vstack((embeddings,centroid))
        umap = UMAP( 
            n_neighbors=100, 
            min_dist=0,
            n_components=2,
            random_state=42,
            transform_seed=42,
            metric="cosine"
        )
        embUmap = umap.fit(extend_embeddings).embedding_
        lowrankEmb = StandardScaler().fit_transform(embUmap)
        centroid=lowrankEmb[-1]
        centroid=centroid.reshape(1,-1)
        
        hdbscan_predLabels = self.clustering_hdbscan(lowrankEmb)
        agglomerative_predLabels = self.clustering_agglomerative(lowrankEmb)
    
        predLabels = hdbscan_predLabels
        cluster_labels=copy.deepcopy(hdbscan_predLabels)
        try:
            silhouetteScore_hdbscan = silhouette_score(lowrankEmb, hdbscan_predLabels)
            silhouetteScore_agglomerative = silhouette_score(lowrankEmb, agglomerative_predLabels)
            if silhouetteScore_hdbscan>silhouetteScore_agglomerative:
                predLabels = hdbscan_predLabels
                logger.info(f'using hdbscan clustering')
            else:
                predLabels = agglomerative_predLabels
                logger.info(f'using agglomerative clustering')
            logger.info(f'initial silhouette score of {client_id}: {silhouetteScore_hdbscan:.4f} {silhouetteScore_agglomerative:.4f}')
            cluster_labels=copy.deepcopy(predLabels)
            if min(silhouetteScore_hdbscan,silhouetteScore_agglomerative)<0.2 and max(silhouetteScore_hdbscan,silhouetteScore_agglomerative)<0.4:
                
                predLabels[:]=0
        except:
            logger.info(f'initial silhouette score of {client_id}: Only one label in the cluster') 
        
        labels=predLabels
        select_label=labels[-1]
        selected_clusters=np.where(labels==select_label)[0].tolist()
        labels[:]=1
        labels[selected_clusters]=0
        
        predLabels = labels

        local_metrics=self.calculate_metrics(poisonLabels,predLabels[:-1])
        try:
            silhouetteScore = silhouette_score(lowrankEmb, cluster_labels)
            logger.info(f'silhouette score of {client_id}: {silhouetteScore:.4f}')
        except Exception as e:
            logger.error(f'Error calculating silhouette score: {e}')
        return extend_embeddings,predLabels,cluster_labels,local_metrics



    def local_filter(self,poison_data, predLabels, poisonLabels):
        cleanIdx = np.where(predLabels == 0)[0]
        total_num=len(predLabels)
        logger.info('total_num')
        
        logger.info(total_num)
        if len(cleanIdx)==0:
            return None
        logger.info('predLabels')
        logger.info(predLabels)
        cleanIdx = np.where(predLabels == 0)[0]
        logger.info('cleanidx')
        logger.info(cleanIdx)
        filteredDataset = self.filtering(poison_data, predLabels, poisonLabels)

        return filteredDataset
        
    def encode(self, dataset, model,fl=None):
        dataloader = getCasualDataloader(dataset, batch_size=1, shuffle=False)
        dctGrads, poisonLabels = self.computeGradients(model, dataloader, "train")
        logger.info("Reducing the dimension of hidden states")
        if 'lm_head' in self.targetPara or 'embed_tokens' in self.targetPara:
            embeddings = self.dimensionReduction(dctGrads, pcaRank=self.pcaRank)
        else:
            embeddings=dctGrads.cpu().numpy()

            

        if fl is not None:
            dctGrads=dctGrads.numpy()
            return embeddings,poisonLabels,dctGrads


        return embeddings, poisonLabels


    def clustering(self, embeddings, metric="cosine", linkage='average'):
        logger.info("Clustering the low dimensional embeddings")
        clusting = AgglomerativeClustering(n_clusters=2, metric=metric, linkage=linkage)

        clusterLabels = clusting.fit_predict(embeddings)
        
        clusterLabels = np.array(clusterLabels)
        
        unique, counts = np.unique(clusterLabels, return_counts=True)
        labelCounts = dict(zip(unique, counts))
        majority = max(labelCounts, key=labelCounts.get)
        
        predLabels = np.where(clusterLabels == majority, 0, 1)

        return np.array(predLabels)
    
    def clustering_hdbscan(self, embeddings, metric="cosine", linkage='average'):
    
        clusterer=HDBSCAN(min_cluster_size=max(int(0.04*len(embeddings)),2))
        
        # clusterer=hdbscan.HDBSCAN(min_cluster_size=5,metric='cosine',gen_min_span_tree=True)
        clusterer.fit(embeddings)
        labels=clusterer.labels_
        
        # 获取标签中出现次数最多和第二多的元素
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)  # 按出现次数降序排序
        for label,count in zip(unique_labels,counts):
            logger.info(f'label: {label}, count: {count}')
        most_common = unique_labels[sorted_indices[0]]  # 出现最多的标签
        
        select_label=most_common
        selected_clusters=np.where(labels==select_label)[0].tolist()
        labels[:]=1
        labels[selected_clusters]=0
        
        predLabels = labels
        return np.array(predLabels)

    def clustering_agglomerative(self, embeddings, metric="cosine", linkage='average'):
    
        clusting=AgglomerativeClustering(n_clusters=2, metric=metric, linkage=linkage)
        clusterLabels = clusting.fit_predict(embeddings)
        
        clusterLabels = np.array(clusterLabels)
        
        unique, counts = np.unique(clusterLabels, return_counts=True)
        labelCounts = dict(zip(unique, counts))
        majority = max(labelCounts, key=labelCounts.get)
        
        predLabels = np.where(clusterLabels == majority, 0, 1)
        return np.array(predLabels)

    def filtering(self, dataset: List, predLabels:np.ndarray, trueLabels:np.ndarray=None):
        
        logger.info("Filtering suspicious samples")
                
        cleanIdx = np.where(predLabels == 0)[0]
        
        filteredDataset = [data for i, data in enumerate(dataset) if i in cleanIdx]
        logger.info(f'detect {len(predLabels) - len(filteredDataset)} poison examples, {len(filteredDataset)} examples remain in the training set')
        
        if trueLabels is not None:
            f1 = f1_score(trueLabels, predLabels, average=None)
            r = recall_score(trueLabels, predLabels, average=None)
            logger.info(f'f1 score of clean and poison: {np.around(f1 * 100, 2)}')
            logger.info(f'recall score of clean and poison: {np.around(r * 100, 2)}')
        
        return filteredDataset
    
    def computeGradients(self, model:CasualLLMVictim, dataLoader:DataLoader, name):
        model.train()
        # param_name=[n for n, p in model.named_parameters() if p.requires_grad]
        # print(param_name)
        
        assert any([self.targetPara in n for n, p in model.named_parameters() if p.requires_grad]), "no corresponding parameter for compute"

        dctGrads, poisonLabels = [], []
        dct2 = lambda tensor: dct_2d(torch.tensor(tensor))
        for i, batch in tqdm(enumerate(dataLoader), desc=f"Calculating gradients of {name}", total=len(dataLoader)):
            poisonLabels.extend(batch["poison_label"])
            model.zero_grad()
            batch_inputs, batch_labels, attentionMask = model.process(batch)
            output = model.forward(inputs=batch_inputs, labels=batch_labels, attentionMask=attentionMask)
            
            loss = output.loss
            # loss.backward()
            grad = autograd.grad(
                loss,
                [p for n, p in model.named_parameters() if (p.requires_grad) and (self.targetPara in n)],
                allow_unused=True
            )
            targetGrad = grad[0].detach()
            if targetGrad.dtype==torch.float16:
                targetGrad=targetGrad.to(torch.float32)
            dctGrad = dct2(targetGrad)
            if "lm_head" in self.targetPara or "embed_tokens" in self.targetPara:
                dctGrad = dctGrad[:int(dctGrad.shape[0] // 8), :int(dctGrad.shape[1] // 8)]
            dctGrads.append(dctGrad.cpu().flatten())
        dctGrads = torch.stack(dctGrads, dim=0)
        logger.info(f'dctGrads_size:{dctGrads.size()}')
        poisonLabels = np.array(poisonLabels)
        return dctGrads, poisonLabels
    
    def dimensionReduction(
        self, hiddenStates: torch.Tensor, 
        pcaRank: Optional[int] = 32
    ):
        _, _, V = torch.pca_lowrank(hiddenStates, q=pcaRank, center=True)
        
        embPCA = torch.matmul(hiddenStates, V[:, :pcaRank])
        
        embStd = StandardScaler().fit_transform(embPCA)
        
        return embStd

    

    
    