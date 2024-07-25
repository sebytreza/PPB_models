import numpy as np
import random
import math
import seaborn as sb
import matplotlib.pyplot as plt
from clustering import Clustering
from functions import assembly
import matplotlib as mpl

############################################

from tqdm import tqdm
from functions import f1_score
'''
def Ck_species(df, method = 'all'): # method is 'all' or int (number of random picks)

    spec_ord = np.argsort(np.sum(df, axis = 0))
    assemblage = np.zeros_like(spec_ord)
    id = 0
    F1 = []
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
                f1 += f1_score(assemblage,survey)
            f1 = f1/len(df)
        else :
            for _ in range(method):
                f1 += f1_score(assemblage,random.choice(df))
            f1 = f1/method
        id += 1
        F1.append(f1)
    return assemblage_max, f1max, F1


def assembly(clusters, cluster_dt, N_cluster, save = False, score = False):
    F1k = []
    N_spec = len(cluster_dt[0])
    Score = np.zeros(N_cluster)
    Ck = np.zeros((N_cluster,N_spec))
    for cl in tqdm(range(N_cluster)):
        if sum((clusters.labels_ == cl)) != 0:
            spec_k, f1, F1 = Ck_species(cluster_dt[(clusters.labels_ == cl)])
            Score[cl] = f1
            Ck[cl] = spec_k
            F1k.append(F1)
    if save:
        np.save('models/Ck_species.npy', Ck)
    if score :
        return Ck, Score, F1k
    return Ck

############################################

p = 0.1

N, M = 10000, 1000
dt = np.zeros((N,M))

for i in range(len(dt)) :
    for j in range(len(dt[0])) :
        dt[i,j]= (random.random() < p)

N_clusters = 100
clustering = Clustering(n_clusters= N_clusters, n_init="auto", verbose = True, batch_size= 64, max_no_improvement= 20)
cluster = clustering.normed_fit(dt)

Ck_spec, F1, F1k = assembly(cluster, dt, N_clusters, score = True)
Size = np.bincount(cluster.labels_, minlength = N_clusters)

labels = cluster.labels_


for i in range(len(F1k)):
    Spe = np.arange(0, len(F1k[i]))
    sb.lineplot(x= Spe, y = F1k[i], hue = Size[i], hue_norm = mpl.colors.Normalize(vmin=0, vmax= np.max(Size), clip=True))

plt.figure()


Dist = clustering.transform(dt)
sb.scatterplot(x = np.argmin(Dist, axis = 1),y = np.min(Dist, axis = 1), hue = Size[np.argmin(Dist, axis = 1)])


plt.figure()



Y = np.zeros(N_clusters)
for i,survey in enumerate(dt):
    Y[labels[i]] += np.sum((survey != False))

Y = np.divide(Y,Size)

#sb.scatterplot(y = np.sum(Ck_spec, axis = 1), x = Size, hue = F1)
sb.scatterplot(x = np.sum(Ck_spec, axis = 1), y = Size, hue = Size)
plt.show()



## Experiment uniformité de la normalization ##


Alpha = np.linspace(0, 2*np.pi, 100)
Prob = np.zeros_like(Alpha)
n = 100
for a in tqdm(range(len(Alpha))):
    alpha = Alpha[a]
    for x in range(n):
        for y in range(n):
            A = int(alpha*x*y + 0.5)
            if np.isclose(alpha*x*y, A, rtol = 1e-3):
                Prob[a] += math.comb(n,A)*math.comb(n,x)*math.comb(n,y)
print(sum(Prob)/(4**(2*n)))

plt.plot(Alpha, Prob/(4**(2*n)))
plt.show()



Alpha = np.linspace(0, 2*np.pi,100)[1:]
Prob = np.zeros_like(Alpha)
n = 10
for a in tqdm(range(len(Alpha))):
    alpha = Alpha[a]
    for x in range(n):
        for y in range(n):
            A = int(alpha*x*y + 0.5)
            if np.isclose(alpha*x*y, A, rtol = 1e-1):
                Prob[a] += math.comb(n,A)*math.comb(n,x)*math.comb(n,y)
print(sum(Prob)/(4**(2*n)))

Alpha = np.linspace(0, 1,1000)
Prob = np.zeros_like(Alpha)
n = 100
for a in tqdm(range(len(Alpha))):
    alpha = Alpha[a]
    for x in range(n):
        for y in range(n):
            A = int(alpha*x*y + 0.5)
            if np.isclose(alpha*x*y, A, rtol = 1e-2):
                Prob[a] += math.comb(x,A)*math.comb(n-x,y-A)*math.comb(n,x)

print(sum(Prob)/(4**n))

plt.plot(Alpha, Prob)
plt.show()
'''

Cos = []
Cos2 = []
Cos3 = []
n = 10000
N = 1000000
for i in range(N):
    X = np.random.randint(0,2,size = n)
    Y = np.random.randint(0,2,size = n)
    Cos.append(X@Y.T/np.sqrt(np.sum(X)*np.sum(Y)))

    X = abs(np.random.randn(n))
    X = X/np.linalg.norm(X)
    Y = abs(np.random.randn(n))
    Y = Y/np.linalg.norm(Y)
    Cos2.append(X@Y.T)

    X = abs(np.random.rand(n))
    X = X/np.linalg.norm(X)
    Y = abs(np.random.rand(n))
    Y = Y/np.linalg.norm(Y)
    Cos3.append(X@Y.T)



plt.hist(Cos, 100)
plt.figure()
plt.hist(Cos2, 100)
plt.figure()
plt.hist(Cos3, 100)

plt.show()


#####################################

import matplotlib.pyplot as plt
import scipy.special as sp
import seaborn as sb
X = np.sum(cluster_dt,axis = 1)
E = np.mean(X)
V = np.var(X)
p = E/V
n  = E**2/(V- E)
f = lambda k : sp.gamma(k + n)/sp.factorial(k)/sp.gamma(n)*p**n*(1-p)**k

T = np.arange(0,100, 1)
F = [f(t) for t in T]
plt.style.use('tex')
c = {'cyan' : '#41a39e',
         'yellow' : '#e1ae36',
         'red' : '#ef4036',
         'lila' : '#a589ff',
         'green' : '#52b400',
         'pink' : '#f5766e',
         'blue' : '#00b7eb'}

sb.histplot(x = X, bins = 50, stat = 'density', color = c['cyan'])
sb.lineplot(x = T,y = F, color = c['yellow'])
plt.xlabel('Number of species')
plt.legend(['Histogram of the survey dataset','_no_legend',f'Negative binomial fit n= {n:.2f}, p = {p:.2f}'])



N = 100000
F2 = []
for _ in range(N):
    x_y = np.random.negative_binomial(n,p,2)
    F2.append(2*np.sqrt(x_y[0])*np.sqrt(x_y[1])/(x_y[0] + x_y[1]))


sb.histplot(x = F2, stat = 'density', binwidth = 0.02, color = c['blue'])

plt.xlabel(r'$\alpha$ value')
hist = np.bincount(X.astype(int))
F3 = []
T2 = []
for i in range(1, len(hist)) :
    for j in range(1, len(hist)):
        F3.append(hist[i] * hist[j])
        T2.append(2*(np.sqrt(i)*np.sqrt(j))/(i + j))

sb.histplot(x = T2, weights= F3, binwidth = 0.02, stat = 'density',color = c['yellow'] )
plt.legend(['Negative binomial', 'Survey dataset'])
plt.show()

