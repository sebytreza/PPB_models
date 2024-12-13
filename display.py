import os
import torch
import numpy as np
import pandas as pd
import timeit
import random
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics import precision_recall_fscore_support, r2_score
from tqdm import tqdm

import seaborn as sb
import matplotlib.pyplot as plt

from dataset import TrainDataset, TestDataset, SpeciesDataset,PostBERTDataset
from model import ModifiedResNet18
from train import Run
from clustering import Clustering, MClustering, T_SNE
from functions import assembly,f1_score, dist1

# Dataset and DataLoader
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor()
])

N_clusters = 600

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

train_data_path = "data/cubes/GLC24-PA-train-bioclimatic_monthly/" 
train_metadata_path = 'data/metadata/GLC24-PA-metadata-train.csv'
bert_metadata_path = 'data/latent_space.csv'

p_validation = 0.2
train_metadata = pd.read_csv(train_metadata_path)
bert_metadata = pd.read_csv(bert_metadata_path)

spec_dataset = SpeciesDataset(train_metadata)
spec_dl = iter(DataLoader(spec_dataset, batch_size= int(len(spec_dataset)*(1 - p_validation)),shuffle= False))
train_id, train_spec = next(spec_dl)
validation_id, validation_spec = next(spec_dl)
# Load Training metadata

test_data_path = "data/cubes/GLC24-PA-test-bioclimatic_monthly/"
test_metadata_path = 'data/metadata/GLC24-PA-metadata-test.csv'
test_metadata = pd.read_csv(test_metadata_path)
test_dataset = TestDataset(test_data_path, test_metadata, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# sb.scatterplot(x = cluster_dataset.metadata["lon"], y = cluster_dataset.metadata["lat"], hue = cluster)
# plt.show()


cluster_dataset = PostBERTDataset(bert_metadata, train_id.numpy())
val_cluster_dataset = PostBERTDataset(bert_metadata, validation_id.numpy())
start = timeit.default_timer()

# clustering = KMeans(n_clusters= N_clusters, n_init="auto")
# train_cluster = clustering.fit(cluster_dataset[:]).labels_
# val_cluster = clustering.predict(val_cluster_dataset[:])
# clustering = MiniBatchKMeans(n_clusters= N_clusters, n_init="auto", verbose = True, batch_size= 640, max_no_improvement= 40)
# clustering = MClustering(n_clusters = N_clusters, metric = dist1, method = 'alternate')

clustering = Clustering(n_clusters= N_clusters, n_init="auto", verbose = True, batch_size= 640, max_no_improvement= 40)
train_cluster = clustering.fit(train_spec)
val_cluster= clustering.predict(validation_spec)

Ck_spec = assembly(train_cluster,train_spec, N_clusters, save = True, method = 'opti')

delta =  timeit.default_timer() - start
print(delta)

Ck_spec = clustering.cluster_centers_
cluster = train_cluster
score_f1 = np.zeros(len(cluster))
dist = np.zeros(len(cluster))

for i in tqdm(range(len(cluster))) :
    idx = train_spec[i].numpy()
    score_f1[i] = f1_score(idx,Ck_spec[cluster[i]]/max(Ck_spec[cluster[i]]))
    dist[i] = np.sqrt((np.sum(idx/np.linalg.norm(idx) - clustering.cluster_centers_[cluster[i]])**2))


cluster = val_cluster
test_score_f1 = np.zeros(len(cluster))

for i in tqdm(range(len(cluster))):
    idx = validation_spec[i].numpy()
    test_score_f1[i] = f1_score(idx,Ck_spec[cluster[i]])


Score_cl = []
for i in range(N_clusters):
    Score_cl.append(np.mean(score_f1[(train_cluster == i)]))

Dist = []
Centers = clustering.cluster_centers_
for i in range(N_clusters) : 
    Dist.append(np.mean([np.sqrt(np.sum((id/np.linalg.norm(id) - Centers[i])**2)) for id in  train_spec[(train_cluster == i)].numpy()]))

Nb_spec = []
for i in range(N_clusters):
    Nb_spec.append(torch.mean(torch.sum(train_spec[(train_cluster == i)], dim  = 1)).item())

mean = np.mean(score_f1)
Size = np.bincount(train_cluster)

'''
Score_f1 = None
Test_score_f1 = None
N_clusters = [100, 300, 500, 600, 800, 1000]
#N_clusters = [600, 1000, 2000, 3000, 5000, 7000]
for n in N_clusters:
    sc_f1, tsc_f1 = score_distrib(n)

    if Score_f1 is None:
        #Score = [sc]
        Score_f1 = [sc_f1]
        #Test_score = [tsc]
        Test_score_f1 = [tsc_f1]

    else :
        #Score    = np.concatenate((Score, [sc]))
        Score_f1 = np.concatenate((Score_f1, [sc_f1]))
        #Test_score = np.concatenate((Test_score, [tsc]))
        Test_score_f1 = np.concatenate((Test_score_f1, [tsc_f1]))
                                
#np.save('models/Score_nn.npy',Score)
np.save('models/Score_F1_nn.npy', Score_f1)
#np.save('models/Score_test_nn.npy', Test_score)
np.save('models/Score_F1_test_nn.npy', Test_score_f1)


n_clusters = 600
clustering = Clustering(n_clusters= n_clusters, n_init="auto", verbose = True, batch_size= 64, max_no_improvement= 20)
cluster_dt = next(iter(cluster_dataloader)).numpy()
train_cluster = clustering.fit(cluster_dt.copy())


Ck_spec, F1 = assembly(train_cluster,cluster_dt, n_clusters, score = True, method = 'centroid')
#np.save('models/nb_barycenter.npy', F1)

test_dt = next(iter(test_dataloader)).numpy()
test_cluster = clustering.predict(test_dt)

cluster = train_cluster
score_f1 = np.zeros(len(cluster))

for i in tqdm(range(len(cluster))):
    idx = cluster_dt[i]
    score_f1[i] = f1_score(idx,Ck_spec[cluster[i]])
    #score[i] = f1_score(idx/np.linalg.norm(idx),Ck_spec[cluster[i]]/np.linalg.norm(Ck_spec[cluster[i]]))

cluster = test_cluster
test_score_f1 = np.zeros(len(cluster))

for i in tqdm(range(len(cluster))):
    idx = test_dt[i]
    test_score_f1[i] = f1_score(idx,Ck_spec[cluster[i]])

np.save('models/centroid_test.npy', test_score_f1)
np.save('models/opti.npy', score_f1)

for i in range(len(F1)):
    if isinstance(F1[i], float):
        F1[i] = np.zeros(1000)

# tsne = T_SNE()
# mask = np.random.choice(len(cluster_dt), 1000)
# Y = tsne.fit_transform(cluster_dt[mask])

# sb.scatterplot(x = Y.T[0], y = Y.T[1], hue = train_cluster[mask])
# plt.show()

test_dt = next(iter(test_dataloader)).numpy()
test_cluster = clustering.predict(test_dt)
'''
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



N = 100000
f1 = []
f1_bar = []
for i in range(N):
    id_x, id_y= np.random.randint(0, len(cluster_dt),2)
    f1.append(f1_score(cluster_dt[id_x], cluster_dt[id_y]))
    f1_bar.append(f1_score(cluster_dt[id_x]/np.linalg.norm(cluster_dt[id_x]), cluster_dt[id_y]/np.linalg.norm(cluster_dt[id_y])))

