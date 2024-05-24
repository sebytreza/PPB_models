import numpy as np 
import random
from tqdm import tqdm

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


def true_pos(df1, df2, id_1, id_2):

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


def assembly(clusters, cluster_dt, N_cluster):

    N_spec = len(cluster_dt[0])
    Ck = np.zeros((N_cluster,N_spec))
    for cl in tqdm(range(N_cluster)):
        if sum((clusters.labels_ == cl)) != 0:
            spec_k, f1 = Ck_species(cluster_dt[(clusters.labels_ == cl)])
            #print(f1)
            Ck[cl] = spec_k
    np.save('models/Ck_species.npy', Ck)
    return Ck
