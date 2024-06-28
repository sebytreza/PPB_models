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

train_metadata_path = 'data/metadata/GLC24-PA-metadata-train.csv'
train_metadata = pd.read_csv(train_metadata_path)
cluster_dataset = ClusteringDataset(train_metadata)
cluster_dataloader = DataLoader(cluster_dataset,batch_size = len(cluster_dataset))

# Load Training metadata

test_metadata_path = "/home/gigotleandri/Documents/GLC24_SOLUTION_FILE.csv"
test_metadata = pd.read_csv(test_metadata_path)
test_metadata.columns = ['surveyId', 'speciesId']
test_dataset = ClusteringDataset(test_metadata, concat = True)
test_dataloader = DataLoader(test_dataset,batch_size = len(test_dataset))


# sb.scatterplot(x = cluster_dataset.metadata["lon"], y = cluster_dataset.metadata["lat"], hue = cluster)
# plt.show()



def score_distrib(n_clusters):

    #clustering = MClustering(n_clusters = n_clusters, metric = dist1, method = 'alternate')
    clustering = Clustering(n_clusters= n_clusters, n_init="auto", verbose = True, batch_size= 64, max_no_improvement= 20)
    cluster_dt = next(iter(cluster_dataloader)).numpy()
    train_cluster = clustering.fit(cluster_dt.copy())
    Ck_spec, _ = assembly(train_cluster,cluster_dt, n_clusters, score = True, method = 'centroïdes')
    test_dt = next(iter(test_dataloader)).numpy()
    test_cluster = clustering.predict(test_dt)


    cluster = train_cluster
    score = np.zeros(len(cluster))
   
    score_f1 = np.zeros(len(cluster))

    for i in tqdm(range(len(cluster))):
        idx = cluster_dt[i]
        score_f1[i] = f1_score(idx,Ck_spec[cluster[i]])
        score[i] = f1_score(idx/np.linalg.norm(idx),Ck_spec[cluster[i]]/np.linalg.norm(Ck_spec[cluster[i]]))

    cluster = test_cluster
    test_score = np.zeros(len(cluster))
    test_score_f1 = np.zeros(len(cluster))

    for i in tqdm(range(len(cluster))):
        idx = test_dt[i]
        test_score_f1[i] = f1_score(idx,Ck_spec[cluster[i]])
        test_score[i] = f1_score(idx/np.linalg.norm(idx),Ck_spec[cluster[i]]/np.linalg.norm(Ck_spec[cluster[i]]))
    
    return score, score_f1, test_score, test_score_f1


Score = None
Score_f1 = None
Test_score = None
Test_score_f1 = None
N_clusters = [10,50,100,200,500,1000]

for n in N_clusters:
    sc, sc_f1, tsc, tsc_f1 = score_distrib(n)

    if Score is None:
        Score = [sc]
        Score_f1 = [sc_f1]
        Test_score = [tsc]
        Test_score_f1 = [tsc_f1]

    else :
        Score    = np.concatenate((Score, [sc]))
        Score_f1 = np.concatenate((Score_f1, [sc_f1]))
        Test_score = np.concatenate((Test_score, [tsc]))
        Test_score_f1 = np.concatenate((Test_score_f1, [tsc_f1]))
                                
np.save('models/Score.npy',Score)
np.save('models/Score_F1.npy', Score_f1)
np.save('models/Score_test.npy', Test_score)
np.save('models/Score_F1_test.npy', Test_score_f1)


'''
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

beta = np.array(Score_F1)/(1 - np.array(Score)**2/2)
sb.scatterplot(x = Labels, y = Diff, hue = Labels == 4)
plt.show() 
plt.figure()
sb.scatterplot(x = np.arange(0,sum(Labels == 4)), y = np.array(Score_F1)[Labels == 4]/(1 - np.array(Score)**2/2)[Labels == 4])
plt.show()
'''

## test on test set cluster attribution
'''
test_data = next(iter(test_dataloader)).numpy()

assembly_F1 = []

for i in tqdm(range(len(test_data))):
    idx = test_data[i]
    assembly_F1.append(f1_score(idx, Ck_spec[test_clusters[i]]))
'''
'''
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
'''