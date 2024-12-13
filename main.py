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

from dataset import TrainDataset, TestDataset, SpeciesDataset
from model import ModifiedResNet18
from train import Run,Run_baseline
from clustering import Clustering, HClustering, MClustering
from functions import assembly
from functions import dist1

# Dataset and DataLoader
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor()
])

N_clusters = 100

run_kmeans = False
new_model = False
num_epochs = 20



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
p_validation = 0.2
train_metadata = pd.read_csv(train_metadata_path)
spec_dataset = SpeciesDataset(train_metadata)
spec_dl = iter(DataLoader(spec_dataset, batch_size= int(len(spec_dataset)*(1 - p_validation))))
train_id, train_spec = next(spec_dl)
validation_id, validation_spec = next(spec_dl)


if not run_kmeans :
    cluster = np.load('clusterisation/cluster.npy')
    val_cluster = np.load('clusterisation/val_cluster.npy')
    Ck_spec = np.load('clusterisation/Ck_spec.npy') 

else : 

    clustering = Clustering(n_clusters= N_clusters, n_init="auto", verbose = True, batch_size= 640, max_no_improvement= 40)
    #clustering = HClustering(n_clusters = N_clusters, metric = dist1, method = 'single')
    #clustering = MClustering(n_clusters = N_clusters, metric = dist1, method = 'alternate')

    cluster = clustering.fit(train_spec)
    val_cluster= clustering.predict(validation_spec)

    np.save('clusterisation/cluster.npy', cluster)
    np.save('clusterisation/val_cluster.npy', val_cluster)

    Ck_spec = assembly(cluster, train_spec, N_clusters, save = True, method = 'opti')


train_dataset = TrainDataset(train_data_path, train_id, train_spec, N_clusters, cluster = cluster, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle= True, num_workers=4)

validation_dataset = TrainDataset(train_data_path, validation_id, validation_spec, N_clusters, cluster = val_cluster, transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle= True, num_workers=4)


# Load Test metadata


test_data_path = "data/cubes/GLC24-PA-test-bioclimatic_monthly/"
test_metadata_path = 'data/metadata/GLC24-PA-metadata-test.csv'
test_metadata = pd.read_csv(test_metadata_path)
test_dataset = TestDataset(test_data_path, test_metadata, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


# Check if cuda is available
device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("DEVICE = CUDA")

# Hyperparameters

learning_rate_1 = 0.00004
learning_rate_2 = 0.00001

model = ModifiedResNet18(N_clusters).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate_2)
scheduler = CosineAnnealingLR(optimizer, T_max=25)


model_baseline = ModifiedResNet18(11255).to(device)


if not new_model :
    model_baseline.load_state_dict(torch.load('models/resnet18-cube_baseline.pth',torch.device('cpu')))
    model_baseline.eval()

    model.load_state_dict(torch.load('models/resnet18.pth',torch.device('cpu')))
    model.eval()

Exp = Run(model,optimizer,scheduler,device)

optimizer_baseline = torch.optim.AdamW(model_baseline.parameters(), lr=learning_rate_1)
scheduler_baseline = CosineAnnealingLR(optimizer_baseline, T_max=25)

Exp_baseline = Run_baseline(model_baseline, optimizer_baseline, scheduler_baseline, device)
#Exp_baseline.train(train_loader, validation_loader, num_epochs, torch.Tensor(Ck_spec), save= "resnet18-cube_baseline",save_loss="Evo_F1_baseline_")
Exp_baseline.test(test_loader)
#Exp.train(train_loader, validation_loader,  num_epochs, Ck_spec,save = "resnet18", save_loss = "Evo_F1_classif")
#Exp.test(test_loader, Ck_spec)