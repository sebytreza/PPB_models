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

import seaborn as sb
import matplotlib.pyplot as plt

from dataset import TrainDataset, TestDataset, ClusteringDataset
from model import ModifiedResNet18
from train import Run
from clustering import Clustering
from functions import assembly

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


S
sb.scatterplot(x = cluster_dataset.metadata["lon"], y = cluster_dataset.metadata["lat"], hue = cluster)
plt.show()
Size = np.bincount(cluster)
print(Size)