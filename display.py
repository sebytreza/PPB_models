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
from clustering import Clustering, MClustering
from functions import assembly,f1_score, dist1

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
clustering = MClustering(n_clusters = N_clusters, metric = dist1, method = 'alternate')
#clustering = Clustering(n_clusters= N_clusters, n_init="auto", verbose = True, batch_size= 64, max_no_improvement= 20)
cluster_dataloader = DataLoader(cluster_dataset,batch_size = len(cluster_dataset))
#top_k = np.argsort(np.sum(next(iter(cluster_dataloader)).numpy(), axis = 0))[-200:]
cluster = clustering.fit(next(iter(cluster_dataloader)).numpy())

test_metadata_path = "/home/gigotleandri/Documents/GLC24_SOLUTION_FILE.csv"
test_metadata = pd.read_csv(test_metadata_path)
test_metadata.columns = ['surveyId', 'speciesId']
test_dataset = ClusteringDataset(test_metadata, concat = True)
test_dataloader = DataLoader(test_dataset,batch_size = len(test_dataset))
test_clusters = clustering.predict(next(iter(test_dataloader)).numpy())

test_Size = np.bincount(test_clusters)

# sb.scatterplot(x = cluster_dataset.metadata["lon"], y = cluster_dataset.metadata["lat"], hue = cluster)
# plt.show()
Size = np.bincount(cluster)
print(Size)

cluster_dt = next(iter(cluster_dataloader)).numpy()

Ck_spec, Score_norm = assembly(cluster,cluster_dt, N_clusters, score = True, method = 'medoid')


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


## test on test set cluster attribution

test_data = next(iter(test_dataloader)).numpy()

assembly_F1 = []

for i in tqdm(range(len(test_data))):
    idx = test_data[i]
    assembly_F1.append(f1_score(idx, Ck_spec[test_clusters[i]]))

sb.boxplot(x = test_clusters, y = assembly_F1)
plt.show()

sb.histplot(x = assembly_F1)
plt.show()



surveys = []
Spec = None
for i in tqdm(range(len(test_data))):       
    ck_spec = Ck_spec[test_clusters[i]]

    if Spec is None:
        Spec = ck_spec
    else:
        Spec = np.concatenate((Spec, ck_spec), axis = 0)

    surveys.extend(test_data.metadata.surveyId[i])

surveys = test_dataset.metadata.surveyId.values
Spec = Ck_spec[test_clusters]
    
data_concatenated = [' '.join(map(str, np.where(row == 1)[0])) for row in Spec]
pd.DataFrame({'surveyId': surveys, 'predictions': data_concatenated,}).to_csv("submissions/test_prediction.csv", index = False)