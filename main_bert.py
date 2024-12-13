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
from sklearn.cluster import MiniBatchKMeans, KMeans

from dataset import TrainDataset, TestDataset, SpeciesDataset, PostBERTDataset
from model import ModifiedResNet18
from train import Run,Run_baseline
from functions import assembly
from functions import dist1

# Dataset and DataLoader
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor()
])

N_clusters = 10000

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
bert_metadata_path = 'data/latent_space.csv'

p_validation = 0.2
train_metadata = pd.read_csv(train_metadata_path)
bert_metadata = pd.read_csv(bert_metadata_path)

spec_dataset = SpeciesDataset(train_metadata)
spec_dl = iter(DataLoader(spec_dataset, batch_size= int(len(spec_dataset)*(1 - p_validation))))
train_id, train_spec = next(spec_dl)
validation_id, validation_spec = next(spec_dl)

cluster_dataset = PostBERTDataset(bert_metadata, train_id.numpy())
val_cluster_dataset = PostBERTDataset(bert_metadata, validation_id.numpy())

# clustering = MiniBatchKMeans(n_clusters= N_clusters, n_init="auto", verbose = True, batch_size= 640, max_no_improvement= 40)
clustering = KMeans(n_clusters= N_clusters, n_init="auto")
cluster = clustering.fit(cluster_dataset[:]).labels_
val_cluster= clustering.predict(val_cluster_dataset[:])

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

learning_rate = 0.00001

model = ModifiedResNet18(N_clusters).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = CosineAnnealingLR(optimizer, T_max=25)

Exp = Run(model,optimizer,scheduler,device)



if __name__ == '__main__' :

    run_kmeans = False
    new_model = True
    num_epochs = 20

    if run_kmeans :
        Ck_spec = assembly(cluster,train_spec, N_clusters, save = True, method = 'opti')
    else : 
        Ck_spec = np.load('models/Ck_species.npy') 


    if not new_model :
        model.load_state_dict(torch.load('models/resnet18-cube_bert.pth',torch.device('cpu')))
        model.eval()

    Exp.train(train_loader, validation_loader, num_epochs, Ck_spec, save = 'resnet18-cube_bert')
    Exp.test(test_loader, Ck_spec)