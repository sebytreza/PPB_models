import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as p
import numba
from numba import cuda
from tqdm import tqdm
import math
from time import perf_counter

@cuda.jit(fastmath=True, nogil=True, parallel = True)
def f1_score(survey1,survey2, f1):
    VP = np.logical_and(survey1,survey2)
    f1 = 2*np.sum(VP)/(np.sum(survey1) + np.sum(survey2))

@cuda.jit(fastmath=True, nogil=True, parallel = True)
def decide(p, kmax, Fmax):
    p = np.sort(p)[::-1]
    n = len(p)
    Fmax, kmax  = -1, 0
    C = np.zeros((n+1,n+1))
    C[0,0] = 1
    for i in range(n):   #O(n²) terme à écrire, tft
        C[i+1,0] = (1 - p[i])*C[i,0]
        for j in range(i+1):
            C[i+1,j+1] = p[i]*C[i,j] + (1 - p[i])*C[i,j+1]
              
    S = np.zeros(2*n)      
    for i in range(2*n):
        S[i] = 1/(i+1)
    
    k = 0
    while k<n and F >= Fmax[0]:
        F = 0
        K = n - k
        for i in range(1, K+1):
            F += 2*i*C[K,i]*S[i + K -1]
        for i in range(2*(K-1)):
            S[i] = p[K-1]*S[i+1] + (1 - p[K-1])*S[i]
            
        if F >= Fmax :
            Fmax[0], kmax[0]  = F, K
        k += 1
    

@cuda.jit(fastmath=True, nogil = True, parallel=True)
def process(x,c,a,SOL, PROBAS):
    S = len(SOL)
    N = 11255
    x,c,a = x[0],c[0],a[0]
    F1_top = 0.
    F1_avg = 0.
    F1 = 0.
    T = lambda x : x**a/(x**a + (1 - x)**a)
    PRO = PROBAS/(PROBAS + x*(1 - PROBAS))
    PRO = c*T(PRO)
    Plim = []
    kmax, Fmax, f1 = np.zeros(1, dtype = np.intp), np.zeros(1), np.zeros(1)
    
    th = np.sort(PRO.flatten())[-int(25*S)]
    for i in range(S):
        #print(i)
        output = PRO[i]
        tar = SOL[i]
        
        sort = np.argsort(-output)
        
        mask = np.where(output > 0)[0]
        decide(output[mask],kmax,Fmax)

        pred = np.zeros(N).astype(np.intp)
        pred[sort[:kmax]] = 1
        Plim.append(output[sort[kmax[0]-1]])
        f1_score(pred,tar,f1)
        F1[0] += f1[0]
        
        topk = np.zeros(N)
        topk[sort[:25]] = 1
        f1_score(tar,topk,f1)
        F1_top[0] += f1[0]

        avgk = np.zeros(N)
        for id in numba.prange(N):
            avgk[id] = output[id] > th
        f1_score(tar,avgk,f1)
        F1_avg[0] += f1[0]
        
    print(F1/S, F1_top/S, F1_avg/S)

def main(x = 10, c = 0.66, a= 0.85):    
    sol = p.read_csv('GLC24_SOLUTION_FILE.csv')['predictions'].to_numpy(dtype = str)
    pred_file = p.read_csv('predictions_test_dataset_pos.csv', delimiter = ',')
    
    probas = pred_file['probas'].to_numpy(dtype=str)
    preds = pred_file['predictions'].to_numpy(dtype=str)
    
    S = len(sol)
    N = 11255
    SOL = np.zeros((S, N), dtype = np.intp)
    for i in range(S):
        row = sol[i]
        specs = row.split(' ')
        for spec in specs :
            id = int(spec)
            SOL[i, id] = 1
            
    PROBAS = np.zeros((S,N), dtype = np.float32)
    for i in range(S) :
        r_preds = preds[i].split(' ')
        r_probas = probas[i].split(' ')
        for j in range(len(r_preds)):
            id = int(r_preds[j])
            PROBAS[i,id] = float(r_probas[j])
    start = perf_counter()
    process(x,c,a,SOL, PROBAS)
    print(perf_counter()-start)


def test():
    L = 100
    p = np.sort(np.random.uniform(0,1,L))[::-1]/100
    p[0] = 0.7
    K , _ = decide(p)
    tries = 1000
    F1 = np.zeros(L)
    F1_ref = 0
    for _ in tqdm(range(tries)):
        spec = np.array(np.random.uniform(0,1,L) < p)
        for i in range(1,L):
            pred = np.zeros(L)
            pred[:i] = 1
            F1[i] += f1_score(spec, pred)
    print(np.argmax(F1) + 1, K, np.max(F1)/tries)


def test_ctl():
    L = 100
    tries = 1000000
    SX = 0
    MX = 0
    VX = 0
    for _ in tqdm(range(tries)):
        p = np.sort(np.random.normal(0,1,L))[::-1]
        p[p>1] = 1
        p[p<0] = 0
        spec = np.array(np.random.uniform(0,1,L) < p)
        SX += np.sum(spec)
        MX += np.sum(p)
        VX += np.sum(p*(1-p))
    
    print(SX, MX, np.sqrt(VX))
    print((SX - MX)/np.sqrt(VX))
    
def calib_fig(x= 10, c = 0.66, a= 0.85):
    sol = p.read_csv('GLC24_SOLUTION_FILE.csv')['predictions'].values

    N = 11255
    S = len(sol)
    SOL = np.zeros((S, N))
    for i in range(S) :
        row = sol[i]
        specs = row.split(' ')
        for spec in specs :
            SOL[i, int(spec)] = 1
        
    pred_file = p.read_csv('predictions_test_dataset.csv')
    probas = pred_file['probas']
    preds = pred_file['predictions']

    PROBAS = np.zeros((S,N))

    for i in range(S) :
        r_preds = preds[i].split(' ')
        r_probas = probas[i].split(' ')
        for j in range(len(r_preds)):
            PROBAS[i,int(r_preds[j])] = float(r_probas[j])
            
    T = lambda x : x**a/(x**a + (1 - x)**a)
    PRO = PROBAS/(PROBAS + x*(1 - PROBAS))
    PRO = c*T(PRO)

    SX = np.sum(SOL)
    MX = np.sum(PRO)
    print(SX, MX)
    
    dp = 0.05
    bins = np.arange(0,1,dp)
    Y = []
    X = []
    sY = []
    for bin in bins:
        idx = np.where((PRO >= bin)*(PRO < bin+dp))
        if len(idx[0]!= 0) :
            Y.append(np.mean(SOL[idx]))
            sY.append(np.std(SOL[idx])/np.sqrt(len(idx[0])))
            X.append(np.mean(PRO[idx]))
    X, Y, sY = np.array(X), np.array(Y), np.array(sY)
    plt.scatter(X,Y)
    plt.plot(X,Y,c = 'blue')
    plt.plot(X,Y + 3*sY,'--',c = 'blue')
    plt.plot(X,Y - 3*sY,'--',c = 'blue')
    plt.plot(X,Y,c = 'blue')

    plt.plot(X,X, c = 'orange')
    plt.xlabel('Predicted probability')
    plt.ylabel('True accuracy')
    plt.gca().set_aspect('equal')

def temp(A):
    X = np.arange(0,1.001,0.01)
    fa = lambda a,x : x**a/(x**a + (1-x)**a)
    plt.plot(X,X)
    for alpha in A:
            plt.plot(X,fa(alpha,X))
    plt.gca().set_aspect('equal')
    plt.show()

main(x = 10, c = 1, a = 1)

'''
plt.figure()
calib_fig(x = 1)
plt.figure()
calib_fig()
plt.figure()
calib_fig(x = 10, c = 0.66)
plt.show()
main(x = 10, c = 0.66)

plt.show()
'''