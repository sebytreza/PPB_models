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

# Load Training metadata
train_data_path = "data/cubes/GLC24-PA-train-bioclimatic_monthly/" 
train_metadata_path = 'data/metadata/GLC24-PA-metadata-train.csv'
train_metadata = pd.read_csv(train_metadata_path)
cluster_dataset = ClusteringDataset(train_metadata)
clustering = Clustering(n_clusters= N_clusters, n_init="auto", verbose = True, batch_size= 64, max_no_improvement= 20)
#clustering = KMeans(n_clusters= N_clusters)
cluster_dataloader = DataLoader(cluster_dataset,batch_size = len(cluster_dataset))
cluster = clustering.normed_fit(next(iter(cluster_dataloader)).numpy())
train_dataset = TrainDataset(train_data_path, cluster, cluster_dataset, N_clusters, subset="train", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Load Test metadata
test_data_path = "data/cubes/GLC24-PA-test-bioclimatic_monthly/"
test_metadata_path = 'data/metadata/GLC24-PA-metadata-test.csv'
test_metadata = pd.read_csv(test_metadata_path)
test_dataset = TestDataset(test_data_path, test_metadata, subset="test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Check if cuda is available
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("DEVICE = CUDA")

# Hyperparameters

learning_rate = 0.0002


model = ModifiedResNet18(N_clusters).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=25, verbose=True)

Exp = Run(model,optimizer,scheduler,device)

if __name__ == '__main__' :

    run_kmeans = False
    new_model = True
    num_epochs = 10

    if run_kmeans :
        Ck_spec = assembly(cluster,next(iter(cluster_dataloader)).numpy(), N_clusters, save = True)
    else : 
        Ck_spec = np.load('models/Ck_species.npy') 


    if not new_model :
        model.load_state_dict(torch.load('models/resnet18-with-bioclimatic-cubes.pth',torch.device('cpu')))
        model.eval()
    
    Exp.train(train_loader, num_epochs, Ck_spec)
    Exp.test(test_loader, Ck_spec)