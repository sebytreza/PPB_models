
import torch
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import seaborn as sb

def f1_score(survey1,survey2):
    VP = np.sum(np.logical_and(survey1,survey2))
    F = np.sum(np.logical_xor(survey1,survey2))
    return 2*VP/(2*VP + F)

def bio_prox(df1, df2, id_1, id_2, marker):
    if 'surveyId' not in df2.columns:
        df2 = df1
    spec_1 = df1.loc[df1.surveyId == id_1].values[0,:11254]
    spec_2 = df2.loc[df2.surveyId == id_2].values[0,:11254]

    dmarker = abs(df1.loc[df1.surveyId == id_1, marker].values[0]- df2.loc[df2.surveyId == id_2, marker].values[0])
    return dmarker, f1_score(spec_1,spec_2)

def true_pos(df1, df2, id_1, id_2S):
    if 'surveyId' not in df2.columns:
        df2 = df1
    spec_1 = df1.loc[df1.surveyId == id_1].values[0,:11254]
    spec_2 = df2.loc[df2.surveyId == id_2].values[0,:11254]

    VP = np.sum(np.logical_and(spec_1,spec_2))
    return VP


def Ck_species(df,method = 'all'): # method is 'all' or int (number of random picks)
    spec_ord = np.argsort(np.sum(df, axis = 0))
    assemblage = np.zeros_like(spec_ord)
    id = 0
    f1max, f1  = 0, 0
    while f1 >= 0.5*f1max :
        if f1 > f1max :
            f1max = f1
            assemblage_max = assemblage.copy()
        f1 = 0
        spec_id = spec_ord[-id-1]
        assemblage[spec_id] = 1
        if method == 'all':
            for survey in df:
                f1 += f1_score(assemblage,survey)
            f1 = f1/len(df)
        else :
            for _ in range(method):
                f1 += f1_score(assemblage,random.choice(df))
            f1 = f1/method
        id += 1
    return assemblage_max, f1max

def flatten_species(df):
    location = df.drop_duplicates('surveyId').drop('speciesId',axis = 1)
    target_arr = np.zeros([len(location),11255],dtype = 'float')
    #target_arr[:,-1] = target_arr[:,-1].astype('int32')
    for i,survey_id in tqdm(enumerate(location["surveyId"].values), total = len(location)):
        presense_species_ids = df.loc[df.surveyId == survey_id,'speciesId'].values
        target_arr[i,[ids-1 for ids in presense_species_ids]] = True
        target_arr[i,-1] = survey_id
    
    species_target_columns = [str(sp_id) for sp_id in range(1,11255)] + ['surveyId']
    #target_arr[:,:-1] = normalize(target_arr[:,:-1], axis=1, norm='l1')
    train_df = pd.DataFrame(target_arr)
    train_df.columns = species_target_columns
    train_df.surveyId = train_df.surveyId.astype('int32')
    return train_df.merge(location, on = 'surveyId')

train_data = pd.read_csv('/home/gigotleandri/Documents/GLC_24/data/GLC24-PA-metadata-train.csv',dtype={'speciesId':int, 'surveyId':int})

'''
test_data= pd.read_csv('/home/gigotleandri/Documents/GLC_24/data/GLC24-PA-metadata-test.csv',dtype={'speciesId':int, 'surveyId':int})

train_elevation = pd.read_csv('/home/gigotleandri/Documents/GLC_24/data/GLC24-PA-train-elevation.csv')
train_footprint = pd.read_csv('/home/gigotleandri/Documents/GLC_24/data/GLC24-PA-train-human-footprint.csv')
train_climate_y= pd.read_csv('/home/gigotleandri/Documents/GLC_24/data/GLC24-PA-train-bioclimatic-average.csv')
train_climate_m= pd.read_csv('/home/gigotleandri/Documents/GLC_24/data/GLC24-PA-train-bioclimatic-monthly.csv')
train_soil= pd.read_csv('/home/gigotleandri/Documents/GLC_24/data/GLC24-PA-train-soilgrids.csv')
train_cover= pd.read_csv('/home/gigotleandri/Documents/GLC_24/data/GLC24-PA-train-landcover.csv')
'''


train_df = flatten_species(train_data[['lat','lon','surveyId','speciesId']])
del train_data

N_spec = 11254

## Investigation of impact of one feature on f1_score ##
L_surveyId = train_df.surveyId.values
Dist = []
Prox = []
for i in range(10000):
    Id1 = random.choice(L_surveyId)
    Id2 = random.choice(L_surveyId)
    dist, prox = bio_prox(train_df, Id1, Id2,'lat')
    Dist.append(dist)
    Prox.append(prox)

plt.scatter(Dist,Prox)
plt.show()

test = pd.merge(test, train_elevation, on = "surveyId")
del train_elevation
test = pd.merge(test, train_cover, on = "surveyId")
del train_cover



## K-means train clustering, metric = specieID ##

N = 10
kmeans = MiniBatchKMeans(n_clusters= N, n_init="auto", verbose = True, batch_size= int(len(train)/12), max_no_improvement= 20)
cluster = kmeans.fit(train.iloc[:,:11254])
train['cluster'] = cluster.labels_



#test clustering
Score = []
True_pos = []
N_cl = kmeans.predict(test.iloc[:,:11254])
test['cluster'] = N_cl


for Id1 in tqdm(test['surveyId'].values):
    n_cl = test.loc[test.surveyId == Id1, 'cluster'].values[0]
    cl = train.loc[train.cluster == n_cl,"surveyId"].values
    Id2 = random.choice(cl)
    _ , prox = bio_prox(test, train, Id1, Id2,'lat')
    Score.append(prox)
    True_pos.append(true_pos(test, train, Id1, Id2))
print(np.mean(Score))

id_cluster, Size = np.unique(train.cluster, return_counts = True)


#isolation of small clusters -> need test clustering
mask = (Size > (np.mean(Size)))
big_cluster = id_cluster[mask]
ratio_cluster = 1 - len(big_cluster)/N
ratio_id = 1 - np.sum(Size[mask])/len(train)
sb.scatterplot(test, x = 'lon', y = 'lat', palette= sb.color_palette("tab10"))
sb.scatterplot(test.loc[np.isin(test.cluster, big_cluster, invert = True)], x = 'lon', y = 'lat', palette= sb.color_palette("tab10"))
plt.show()

#incidence of cluster size on f1 score -> need test clustering
Score_cluster = []
for clust in id_cluster:
    Score_cluster.append(np.mean(Score, where = (N_cl == clust)))

plt.scatter(Size, Score_cluster)
plt.show()

sb.scatterplot(test.loc[test.cluster == np.argmax(Size)], x = 'lon', y = 'lat', palette= sb.color_palette("tab10"))
plt.show()

#investigation on the presence one big cluster -> need test clustering
Dist = kmeans.transform(test.iloc[:,:11254])
sb.scatterplot(x = Size[np.argmin(Dist, axis = 1)],y = np.min(Dist, axis = 1), hue = N_cl)
plt.show()

# relation between f1 score and metric dist
Truepos_cluster = []
for clust in id_cluster:
    Truepos_cluster.append(np.mean(True_pos, where = (N_cl == clust)))
plt.scatter(Truepos_cluster,Score_cluster)
plt.show()

## Ck assemblage f1 score evolution ## -> need train clustering
Ck = np.zeros((N,N_spec))
Score_norm = np.zeros(N)
for cl in tqdm(range(N)):
    spec_k, f1 = Ck_species(train.loc[train.cluster == cl].values[:,:N_spec])
    Ck[cl] = spec_k
    Score_norm[cl] = f1

Size = train.cluster.value_counts().sort_index().values
plt.scatter(Size,Score_norm)
plt.show()



## Clustering with normalisation on test dataset only## -> need flatten
del train
test2 = test.copy()
test2.iloc[:,:11254] = normalize(test2.iloc[:,:11254],axis = 1,norm = 'l1')
N2 = 200
kmeans = MiniBatchKMeans(n_clusters= N2, n_init="auto", verbose = True, batch_size= int(len(test2)/12), max_no_improvement= 100)
cluster = kmeans.fit(test2.iloc[:,:11254])
test2['cluster'] = cluster.labels_

Size2 = np.bincount(cluster.labels_)
Ck2 = np.zeros((N2,N_spec))
Score_norm = np.zeros(N2)
Nb_spec = np.zeros(N2)
for cl in tqdm(range(N2)):
    if len(test2.loc[test2.cluster == cl]) != 0:
        spec_k, f1 = Ck_species(test2.loc[test2.cluster == cl].values[:,:N_spec])
        Ck2[cl] = spec_k
        Score_norm[cl] = f1
        Nb_spec[cl] = np.mean(np.sum((test2.loc[test2.cluster == cl].values[:,:N_spec]!= 0), axis = 1))



sb.scatterplot(x = np.sum(Ck2, axis = 1), y = Nb_spec, hue = Score_norm)
plt.xlabel('Mean number of present species')
plt.ylabel('Mean F1 score between cluster survey and cluster assembly')
plt.title('F1 score relative to number of present species by cluster size (N = 50)')
plt.show()

sb.scatterplot(test2, x = 'lon', y = 'lat', palette= sb.color_palette("tab10"))
sb.scatterplot(test2.loc[test2.cluster == np.argmax(Size2)], x = 'lon', y = 'lat', palette= sb.color_palette("tab10"))
plt.show()

