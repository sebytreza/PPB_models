import numpy as np 
import random
import torch
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


def centroid(df,method = 'all'): # method is 'all' or int (number of random picks)
    mask = np.where(torch.sum(df, dim = 0) > 0)[0]
    spec = np.zeros(len(df[0]))
    df = df[:, mask]
    ndf = normalize(df,'l2', axis = 1)
    spec_ord = np.argsort(np.sum(ndf, axis = 0))
    assemblage = np.zeros_like(spec_ord)
    id = 0
    f1max, f1  = 0, 0
    while f1 >= 0.5*f1max  and id < len(spec_ord):
        f1 = 0
        id += 1
        spec_id = spec_ord[-id]
        assemblage[spec_id] = 1
        if method == 'all' or len(ndf) < method:
            for survey in ndf:
                f1 += f1_score(assemblage/np.linalg.norm(assemblage),survey)
            f1 = f1/len(ndf)
        else :
            for _ in range(method):
                f1 += f1_score(assemblage/np.linalg.norm(assemblage),random.choice(ndf))
            f1 = f1/method

        if f1 > f1max :
            f1max = f1


    f1 = 0
    spec[mask[spec_ord[-id+1:]]] = 1
    for survey in df:
        f1 += f1_score(assemblage,survey)
    
    return spec, f1/len(df)


def medoid(df, method = 'all'):
    f_med = 0
    if method != 'all' and len(df) > method:
        df = df[np.random.choice(len(df), method)]
    for survey1 in df:
        f1 = 0
        for survey2 in df :
            f1 += f1_score(survey1, survey2)
        if f1 >= f_med :
            med = survey1
            f_med = f1
    return med, f_med/len(df)

def mc_uniform(df, nb, method = 'all'):
    f_max = 0
    f_mc = np.zeros(nb)
    if method != 'all' and len(df) > method:
        df = df[np.random.choice(len(df), method)]
    distr = np.mean(df,axis = 0)
    for n in range(nb):
        mc = np.random.randint(0,2, len(df[0]))
        mc = np.logical_and(distr,mc).astype(int)
        f1 = 0
        for survey in df:
            f1 += f1_score(survey, mc)
        if f1 >= f_max:
            f_max = f1
            spec = mc
        f_mc[n] = f_max/len(df)
    return spec, f_mc

def mc_distri(df, nb, method = 'all'):
    f_max = 0
    f_mc = np.zeros(nb)

    if method != 'all' and len(df) > method:
        df = df[np.random.choice(len(df), method)]
    distr = np.mean(df,axis = 0)
    for n in range(nb):
        mc = np.random.rand(len(df[0]))
        mc = np.array(distr > mc).astype(int)
        f1 = 0
        for survey in df:
            f1 += f1_score(survey, mc)
        if f1 >= f_max:
            f_max = f1
            spec = mc
        f_mc[n] = f_max/len(df)
    return spec, f_mc

def mc_norm(df, nb, method = 'all'):
    f_max = 0
    f_mc = np.zeros(nb)
    if method != 'all' and len(df) > method:
        df = df[np.random.choice(len(df), method)]
    distr = np.mean(normalize(df),axis = 0)
    for n in range(nb):
        mc = np.random.rand(len(df[0]))
        mc = np.array(distr > mc).astype(int)
        f1 = 0
        for survey in df:
            f1 += f1_score(survey, mc)
        if f1 >= f_max:
            f_max = f1
            spec = mc
        f_mc[n] = f_max/len(df)
    return spec, f_mc

def euclidian_bar(df, method = 'all'):

    if method != 'all' and len(df) > method:
        df = df[np.random.choice(len(df), method)]

    bar = np.mean(normalize(df), axis = 0)
    spec = bar/max(bar)
    spec_ord = np.argsort(spec)
    assemblage = np.zeros_like(spec_ord)
    id = 0
    f1max, f1  = 0, 0
    while f1 >= f1max :
        f1max = f1
        assemblage_max = assemblage.copy()

        spec_id = spec_ord[-id-1]
        assemblage[spec_id] = 1
        f1 = f1_score(assemblage,spec)
        id += 1

    f1  = 0
    for survey in df:
        f1 += f1_score(assemblage_max, survey)
     
    return assemblage_max, f1/len(df)

    
def opti(df,method = 'all'): # method is 'all' or int (number of random picks)
    mask = np.where(torch.sum(df, dim = 0) > 0)[0]
    spec = np.zeros(len(df[0]))
    df = df[:, mask]
    N = len(df[0])
    id = 0
    f1max, f1  = 0, 0
    norm = np.linalg.norm(df, axis = 1)

    while id < N :
        assemblage = np.zeros(N)
        spec_ord = np.argsort(torch.sum(df.T/(id + 1 + norm**2), dim = 1))
        f1 = 0
        spec_id = spec_ord[-id-1:]
        assemblage[spec_id] = 1

        if method == 'all' or len(df) < method:
            for survey in df:
                f1 += f1_score(assemblage,survey)
            f1 = f1/len(df)
        else :
            for _ in range(method):
                f1 += f1_score(assemblage,random.choice(df))
            f1 = f1/method

        if f1 > f1max :
            f1max = f1
            assemblage_max = assemblage.copy()

        id += 1
    spec[mask] = assemblage_max
    return spec, f1max

def assembly(clusters, cluster_dt, N_cluster, save = False, score = False, method = False):

    N_spec = len(cluster_dt[0])
    Score = list(np.zeros(N_cluster))
    Ck = np.zeros((N_cluster,N_spec))
    for cl in tqdm(range(N_cluster)):
        if sum((clusters == cl)) != 0:
            if method == 'medoid' :
                spec_k, f1 = medoid(cluster_dt[(clusters == cl)])
            elif method == 'mc_uniform':
                spec_k, f1 = mc_uniform(cluster_dt[(clusters == cl)],nb = 1000)
            elif method == 'mc_distri':
                spec_k, f1 = mc_distri(cluster_dt[(clusters == cl)], nb = 1000)
            elif method == 'mc_norm':
                spec_k, f1 = mc_norm(cluster_dt[(clusters == cl)], nb = 1000)
            elif method == 'centroid' :
                spec_k, f1 = centroid(cluster_dt[(clusters == cl)])
            elif method == 'barycenter' :
                spec_k, f1 = euclidian_bar(cluster_dt[(clusters == cl)])
            elif method == 'opti' :
                spec_k, f1 = opti(cluster_dt[(clusters == cl)])
            else : 
                raise NameError
            Score[cl] = f1
            Ck[cl] = spec_k
    if save:
        np.save('models/Ck_species.npy', Ck)
    if score :
        return Ck, Score
    return Ck



def dist1(x,y):
    dist = 1 - f1_score(x,y)
    if dist < 0 : # allow to pass float rounding problems
        return 0
    return np.sqrt(dist)

    

def weighted_assignment(p,n_centers):

    spec = np.zeros(len(n_centers[0]))

    mask = np.where(p >= 0.3*torch.max(p))[0]

    p, n_centers = p[mask], n_centers[mask]

    mask = np.where(torch.sum(n_centers, dim = 0) > 0)[0]
    ndf = n_centers[:, mask]

    W = p[:, None]*ndf

    spec_ord = np.argsort(torch.sum(W, dim  = 0))
    assemblage = np.zeros_like(spec_ord)
    id = 0
    f1max, f1  = 0, 0
    while f1 >= f1max and id < len(spec_ord) :

        f1max = f1
        f1 = 0
        id += 1
        spec_id = spec_ord[-id]
        assemblage[spec_id] = 1
        for i  in range(len(ndf)):
            f1 += p[i] * f1_score(assemblage/np.sqrt(id),ndf[i])
        f1 = f1/len(ndf)

    if id != len(spec_ord): id += -1
    spec[mask[spec_ord[-id :]]] = 1

    return spec
    