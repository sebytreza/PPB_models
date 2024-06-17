import numpy as np 
import random
from tqdm import tqdm
from sklearn.preprocessing import normalize

def f1_score(survey1,survey2):
    VP = np.array(survey1) @ np.array(survey2).T
    return 2*VP/(np.linalg.norm(survey1)**2 + np.linalg.norm(survey2)**2)


def bio_prox(df1, df2, id_1, id_2, marker):

    if 'surveyId' not in df2.columns:
        df2 = df1
    spec_1 = df1.loc[df1.surveyId == id_1].values[0,:11254]
    spec_2 = df2.loc[df2.surveyId == id_2].values[0,:11254]

    dmarker = abs(df1.loc[df1.surveyId == id_1, marker].values[0]- df2.loc[df2.surveyId == id_2, marker].values[0])
    return dmarker, f1_score(spec_1,spec_2)


def true_pos(df1, df2, id_1, id_2):

    if 'surveyId' not in df2.columns:
        df2 = df1
    spec_1 = df1.loc[df1.surveyId == id_1].values[0,:11254]
    spec_2 = df2.loc[df2.surveyId == id_2].values[0,:11254]

    VP = np.sum(np.logical_and(spec_1,spec_2))
    return VP


def Ck_species(df,method = 'all'): # method is 'all' or int (number of random picks)
    normalize(df,'l2', axis = 1, copy = False)
    spec_ord = np.argsort(np.sum(df, axis = 0))
    assemblage = np.zeros_like(spec_ord)
    id = 0
    f1max, f1  = 0, 0
    while f1 >= 0.5*f1max  and id < len(spec_ord) - 1:
        if f1 > f1max :
            f1max = f1
            assemblage_max = assemblage.copy()
        f1 = 0
        spec_id = spec_ord[-id-1]
        assemblage[spec_id] = 1
        if method == 'all':
            for survey in df:
                f1 += f1_score(assemblage/np.linalg.norm(assemblage),survey)
            f1 = f1/len(df)
        else :
            for _ in range(method):
                f1 += f1_score(assemblage/np.linalg.norm(assemblage),random.choice(df))
            f1 = f1/method
        id += 1
    return assemblage_max, f1max


def assembly(clusters, cluster_dt, N_cluster, save = False, score = False):

    N_spec = len(cluster_dt[0])
    Score = np.zeros(N_cluster)
    Ck = np.zeros((N_cluster,N_spec))
    for cl in tqdm(range(N_cluster)):
        if sum((clusters == cl)) != 0:
            spec_k, f1 = Ck_species(cluster_dt[(clusters == cl)])
            Score[cl] = f1
            Ck[cl] = spec_k
    if save:
        np.save('models/Ck_species.npy', Ck)
    if score :
        return Ck, Score
    return Ck

def dist1(x,y):
    return np.sqrt(1 - f1_score(x,y))
