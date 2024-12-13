import numpy as np
from scipy.stats import beta
from functions import f1_score
from scipy.stats import nchypergeom_wallenius,binom
import matplotlib.pyplot as plt
import math

def Vect_p(p):
    V  = np.ones(2**len(p))
    for i in range(len(p)):
        j = 2**i
        while j < len(V):
            for l in range(2**i):
                V[j+l] *= p[i]
                V[j+l - 2**i] *= 1 - p[i]
            j += 2**(i+1)
    return V



def Vect_F1(Card,k,N1):
    Num = Card[k-1] + N1
    Den = Card[-1] + k + 2*N1
    return 2*np.divide(Num,Den)

def pred(p,N,k):

    Card = []
    V = np.zeros(2**N)
    for i in range(N):
        j = 2**i
        while j < 2**N:
            for l in range(2**i):
                V[j+l] += 1
            j += 2**(i+1)
        Card.append(V.copy())
    Sc1, Sc2 = 0,0
    spec = np.zeros(len(p))
    N1 = 0
    id_sorted = np.argsort(-p)
    p_sorted = np.concatenate((p[id_sorted], np.zeros(N)), axis = 0)
    while Sc1 >= Sc2 and N1 < len(p)+1:
        N1 += 1
        Sc2 = Sc1
        if N1 <= k :
            Sc1 = Vect_F1(Card,N1,0) @ Vect_p(p_sorted[:N])
        else :
            Sc1 = Vect_F1(Card,k, N1- k) @ Vect_p(p_sorted[N1-k : N1-k+N])
    spec[id_sorted[:N1-1]] = 1
    return spec

def pred_ord2(p):
    Sc1, Sc2 = 0,0
    spec = np.zeros(len(p))
    k = 0
    id_sorted = np.argsort(-p)
    p_sorted = p[id_sorted]
    prod = np.ones(len(p))
    for i in range(len(p)):
        for j in range(len(p)):
            if j != i and j!= i+1 :
                prod[j] *= 1 - p_sorted[i]
        
    n1, n2 = 0,0
    while Sc1 >= Sc2 and k < len(p)-1:
        k += 1
        Sc2 = Sc1
        n1 *= k + 1
        n2 *= k + 2
        for j in range(k):
            n1 += (1- p_sorted[j])*p_sorted[k]*prod[k-1]
            n2 += 2*p_sorted[k]*p_sorted[j]*prod[k-1]
        n1 = n1/(k +2)
        n2 = n2/(k + 3)
        Sc1 = n1 + n2
    spec[id_sorted[:k-1]] = 1
    return spec

def pred_gauss(p):
    spec = np.zeros(len(p))
    N = len(p)
    k = 0
    id_sorted = np.argsort(-p)
    p_sorted = p[id_sorted]
    s = 3
    pm = np.mean(p)
    psd = np.std(p)
    nlist  = np.arange(1,min(int(N*abs(pm + s*psd)),N))
    pond = np.exp(-(nlist/N-pm)**2/psd**2)
    pond = binom(N,pm).pmf(nlist)

    Sc1,Sc2 = 0,0
    while Sc1 >= Sc2  and k < len(p)-1:
        k += 1
        Sc2 = Sc1
        Sc1 = 0
        for j in range(len(nlist)):
            n = nlist[j]
            Pp =nchypergeom_wallenius(N,k,n,np.mean(p_sorted[k-1])/np.mean(p_sorted[k:])).pmf(np.arange(1,min(n,k)+1))
            for i in range(1,min(n,k)+1):
                Sc1 +=  i/(n + k)*pond[j]*Pp[i-1]
    spec[id_sorted[:k-1]] = 1
    return spec

def pred_classic(p):
    spec = p/max(p)
    proj = np.zeros_like(spec)
    spec_ord = np.argsort(spec)
    assemblage = np.zeros_like(spec_ord)
    id = 0
    f1max, f1  = 0, 0
    while f1 >= f1max and id < len(p):
        f1max = f1
        id += 1
        spec_id = spec_ord[-id]
        assemblage[spec_id] = 1
        f1 = f1_score(assemblage,spec)
                
    proj[spec_ord[-id+1:]]  = 1
    return proj


a,b = 1,999
N_spec = 10000
p = beta.rvs(a,b, size = N_spec)
#alpha = np.random.exponential(scale = 0.01, size = N_spec)
#alpha = beta.rvs(1,2, size = N_spec)
#p = np.random.dirichlet(alpha)

#n_top = int(a/(a+b)*N_spec+0.5)

Sc = 0
Sc_topk = 0
Sc2 = 0
Sc3 = 0
Sc_g = 0
Sc_max = 0
Sc_kf = np.zeros(50)
Sc_seuil = np.zeros(100)

Lmax = []
Win = []
Nummax = []
selec = np.argsort(-np.sum(Pre, axis = 0))[:15]
#selec = np.arange(len(Tar[0]))
tar = Tar[:,selec]
pre = Pre[:,selec]

tar = dataset[:,np.argsort(-np.sum(dataset, axis = 0))]
pre = Test[:,np.argsort(-np.sum(dataset, axis = 0))]

seuil = np.exp(np.mean(np.log(pre)*tar + np.log(1-pre)*(1-tar), axis = 0))

N_mc = 200
Num_spec = []
for id in range(N_mc):
    sp = tar[id] #np.array((p > np.random.rand(len(p))))
    p = pre[id]
    mask = np.argsort(-p)[:100]
    topk = np.zeros_like(p)
    topk[np.argsort(-p)[:max(1,int(sum(p)))]] = 1

    N = 15
    k = 15
    challenger = np.zeros_like(topk)
    challenger2 = np.zeros_like(topk)
    challenger_classic = np.zeros_like(topk)
    challenger[mask]= pred(p[mask],N,k)
    challenger2[mask] = pred_ord2(p[mask])
    challenger_classic[mask]= pred_classic(p[mask])
    challenger_gauss = np.zeros_like(topk)
    challenger_gauss[mask] = pred_gauss(p[mask])

    sptry = np.zeros_like(topk)
    spmax = sptry.copy()
    f1max = 0
    for i in range(len(mask)):
        sptry[mask[i]] = 1
        f1 = f1_score(sptry,sp)
        if f1 >= f1max :
            f1max = f1
            spmax = sptry.copy()
    
    spkf = np.zeros_like(topk)
    for i in range(len(Sc_kf)):
        spkf[mask[i]] = 1
        Sc_kf[i] += f1_score(sp, spkf)
    
    for i in range(len(Sc_seuil)):
        Sc_seuil[i] += f1_score(sp, np.array(p>i/100))

    
    print(id, int(sum(spmax)), int(sum(sp)), int(sum(topk)), int(sum(challenger)),int(sum(challenger2)),int(sum(challenger_classic)), int(sum(challenger_gauss)))

    Num_spec.append(sum(sp))
    Sc_max += f1max
    Lmax.append(f1max)
    Nummax.append(int(sum(spmax)))
    Win.append(f1_score(sp, challenger_gauss) -f1_score(sp,topk))
    Sc_topk += f1_score(sp,topk)
    Sc += f1_score(sp, challenger)
    Sc2 += f1_score(sp, challenger2)
    Sc3 += f1_score(sp, challenger_classic)
    Sc_g += f1_score(sp, challenger_gauss)


Sc_topk /= N_mc
Sc /= N_mc
Sc2 /= N_mc
Sc3/= N_mc
Sc_g /= N_mc
Sc_max /= N_mc

print(Sc_max, Sc_topk, Sc, Sc2, Sc3, Sc_g, max(Sc_kf)/N_mc, max(Sc_seuil)/N_mc, np.mean(Num_spec), np.std(Num_spec))


import seaborn as sb

sb.scatterplot(x = Num_spec, y = Lmax, hue = Win, hue_norm= (-1,1), palette= 'vlag')
plt.show()




import pandas as pd
from dataset import SpeciesDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch 

train_data_path = "data/cubes/GLC24-PA-train-bioclimatic_monthly/" 
train_metadata_path = 'data/metadata/GLC24-PA-metadata-train.csv'
p_validation = 0.2
train_metadata = pd.read_csv(train_metadata_path)
spec_dataset = SpeciesDataset(train_metadata)
spec_dl = iter(DataLoader(spec_dataset, batch_size= int(len(spec_dataset)*(1 - p_validation))))
train_id, train_spec = next(spec_dl)
validation_id, validation_spec = next(spec_dl)


test_path = "../../GLC24_SOLUTION_FILE.csv"
test_data = pd.read_csv(test_path)
dataset = np.zeros((len(test_data),11255))
data = list(map(lambda x: np.array(x.split(' ')).astype('int'), test_data["predictions"].values))
for row in range(len(data)):
    dataset[row][data[row]] = 1
Test = np.load("data/pred_test.npy") 
Pre = np.load("data/pred.npy")
Tar = np.load("data/true.npy")


plt.hist(pre[id], bins = np.arange(0,0.8,0.01))
plt.hist(pre[id][np.where(tar[id]>0)], bins = np.arange(0,0.8,0.01),color= '#ff0f0f80')
print(sum(tar[id]))


plt.show()

plt.hist(np.mean(tar, axis = 0), bins = 100)
plt.show()

plt.hist(np.log(Pre.flatten()), bins = np.arange(-20,0,0.1))
plt.hist(np.log(Pre.flatten())[np.where(Tar.flatten() == 1)],bins = np.arange(-20,0,0.1))
plt.show()


SUM = np.sum(Pre, axis = 1)
plt.hist(np.sum(Tar, axis = 1), bins = np.arange(100))
pm = np.mean(SUM)
psd = np.std(SUM)

plt.show()