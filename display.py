import os
import torch
import numpy as np
import pandas as pd
import random
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cluster import MiniBatchKMeans
from torch_kmeans import KMeans
from tqdm import tqdm

import seaborn as sb
import matplotlib.pyplot as plt

from dataset import TrainDataset, TestDataset, ClusteringDataset
from model import ModifiedResNet18
from train import Run
from clustering import Clustering
from functions import assembly,f1_score

# Dataset and DataLoader
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor()
])

N_clusters = 100

seed = 42
# Set seed for Python's built-in random number generator
torch.manual_seed(seed)
# Set seed for numpy
np.random.seed(seed)
# Set seed for CUDA if available
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    # Set cuDNN's random number generator seed for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Load Training metadata
train_data_path = "data/cubes/GLC24-PA-train-bioclimatic_monthly/" 
train_metadata_path = 'data/metadata/GLC24-PA-metadata-train.csv'
train_metadata = pd.read_csv(train_metadata_path)
cluster_dataset = ClusteringDataset(train_metadata)
clustering = Clustering(n_clusters= N_clusters, n_init="auto", verbose = True, batch_size= 64, max_no_improvement= 20)
cluster_dataloader = DataLoader(cluster_dataset,batch_size = len(cluster_dataset))
#top_k = np.argsort(np.sum(next(iter(cluster_dataloader)).numpy(), axis = 0))[-200:]
cluster = clustering.fit(next(iter(cluster_dataloader)).numpy())


# sb.scatterplot(x = cluster_dataset.metadata["lon"], y = cluster_dataset.metadata["lat"], hue = cluster)
# plt.show()
Size = np.bincount(cluster)
print(Size)

cluster_dt = next(iter(cluster_dataloader)).numpy()

Ck_spec, Score_norm = assembly(cluster,cluster_dt, N_clusters, score = True)


Labels = clustering.labels_
Centers = clustering.cluster_centers_
Score_km = []
Score_km_F1 = []
Clusters_F1 = []
Clusters = []
Score_F1  = []
Score_f1 = []
Score = []

Assembly = []
Assembly_F1 = []

Diff = []

for i in tqdm(range(len(cluster_dt))):
    idx = cluster_dt[i]
    Assembly.append(np.linalg.norm(idx/np.linalg.norm(idx) - Ck_spec[Labels[i]]/np.linalg.norm(Ck_spec[Labels[i]])))
    Assembly_F1.append(f1_score(idx, Ck_spec[Labels[i]]))

    idx = cluster_dt[i]
    Score_km.append(np.linalg.norm(idx/np.linalg.norm(idx) - Centers[Labels[i]]/np.linalg.norm(Centers[Labels[i]])))
    Score_km_F1.append(f1_score(idx, Centers[Labels[i]]))

    cl = np.argmax([f1_score(idx,center) for center in Ck_spec])
    Clusters_F1.append(cl)
    Score_F1.append(f1_score(idx,Ck_spec[cl]))

    cl = np.argmin([np.linalg.norm(idx/np.linalg.norm(idx) - center/np.linalg.norm(center)) for center in Ck_spec])
    Clusters.append(cl)
    Score.append(np.linalg.norm(idx/np.linalg.norm(idx) - Ck_spec[cl]/np.linalg.norm(Ck_spec[cl])))
    Score_f1.append(f1_score(idx, Ck_spec[cl]))

    Diff.append(f1_score(Ck_spec[Labels[i]],Ck_spec[cl]))

beta = np.array(Score_F1)/(1 - np.array(Score)**2/2)
sb.scatterplot(x = Labels, y = Diff, hue = Labels == 4)
plt.show() 
plt.figure()
sb.scatterplot(x = np.arange(0,sum(Labels == 4)), y = np.array(Score_F1)[Labels == 4]/(1 - np.array(Score)**2/2)[Labels == 4])
plt.show()

