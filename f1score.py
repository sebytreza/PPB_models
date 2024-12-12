
import numpy as np
import matplotlib.pyplot as plt
import pandas as p
import numba
from tqdm import tqdm
import math
from time import perf_counter
from scipy.stats import pearsonr

@numba.njit(fastmath=True, nogil=True, parallel = True)
def f1_score(survey1,survey2):
    VP = np.logical_and(survey1,survey2)
    return 2*np.sum(VP)/(np.sum(survey1) + np.sum(survey2))

@numba.njit(fastmath=True, nogil=True)
def f1(TP, FP, TN, FN):
    return (TN +TP) /(TP + FP + FN + TN)

@numba.njit(fastmath=True, nogil=True, parallel = True)
def decide(p):
    p = np.sort(p)[::-1]
    n = len(p)
    Fmax, kmax, F  = 0, 0, 0
    C = np.zeros((n+1,n+1))
    C[0,0] = 1
    for i in range(n):   #O(n²) terme à écrire, tft
        C[i+1,0] = (1 - p[i])*C[i,0]
        for j in range(i+1):
            C[i+1,j+1] = p[i]*C[i,j] + (1 - p[i])*C[i,j+1]
              
    S = np.zeros(2*n)      
    for i in range(2*n):
        S[i] = 1/(i+1)
    
    K = n
    while K > 0 and Fmax == F:
        F = 0
        for i in range(1, K+1):
            F += 2*i*C[K,i]*S[i + K -1]
            #print(C[K,i], S[i + K -1])
        for i in range(2*(K-1)):
            S[i] = p[K-1]*S[i+1] + (1 - p[K-1])*S[i]
            
        if F >= Fmax :
            Fmax, kmax  = F, K
        K -= 1
        
    return kmax, Fmax

@numba.njit(fastmath=True, nogil=True, parallel = True)
def D_func(p, func):
    p = np.sort(p)[::-1]
    n = len(p)
    Fmax, kmax, F  = 0, 0, 0
    C = np.zeros((n+1,n+1))
    C[0,0] = 1
    for i in range(n):
        C[i+1,0] = (1 - p[i])*C[i,0]
        for j in range(i+1):
            C[i+1,j+1] = p[i]*C[i,j] + (1 - p[i])*C[i,j+1]
              
    S = np.zeros((n+1,n+1))
    S[0,0] = 1
    for i in range(n): 
        S[i+1,0] = (1 - p[n-i-1])*S[i,0]
        for j in range(i+1):
            S[i+1,j+1] = p[n-i-1]*S[i,j] + (1 - p[n-i-1])*S[i,j+1]     

    K = 1
    while K < n and Fmax == F:
        F = 0
        for i in range(K+1):
            for j in range(n - K + 1):
                F += C[K,i]*S[n-K,j]*func(i, K -i, n-K-j, j) #func(tp,fp,tn, fn)
            
        if F >= Fmax :
            Fmax, kmax  = F, K
        K += 1
        
    return kmax, Fmax

@numba.njit(fastmath=True, nogil = True, parallel=True)
def process(x,c,a,SOL, PROBAS):
    S = len(SOL)
    N = 11255

    F1_top = 0.
    F1_avg = 0.
    F1 = 0.
    F1_sum = 0.
    F1_rich = 0.

    n = 1000
    #F_tresh = np.zeros(n).astype(np.float32)

    T = lambda x : x**a/(x**a + (1 - x)**a)
    PRO = PROBAS/(PROBAS + x*(1 - PROBAS))
    PRO = c*T(PRO)

    # T_probas = []
    # P_probas = []
    # L_probas = []

    th = np.sort(PRO.flatten())[-int(25*S)]
    for i in range(S):
        output = PRO[i]
        tar = SOL[i]
        
        sort = np.argsort(-output)
        
        mask = np.where(output > 0)[0]
        K, _ = decide(output[mask])
        pred = np.zeros(N).astype(np.intp)
        pred[sort[:K]] = 1
        F1 += f1_score(pred,tar)

        # P_probas.extend(list(output[sort[:K]]))
        # T_probas.extend(list(output[np.where(tar >0)]))
        # L_probas.append(output[sort[K-1]])
        
        topk = np.zeros(N)
        topk[sort[:25]] = 1
        F1_top += f1_score(tar,topk)
        
        avgk = np.zeros(N)
        for id in numba.prange(N):
            avgk[id] = output[id] > th
        F1_avg += f1_score(tar,avgk)
        
        '''
        for j in range(n):
            tresh = np.zeros(N)
            for id in numba.prange(N):
                tresh[id] = output[id] > j/n
            F_tresh[j] += f1_score(tresh,tar)
        '''

        sum_k = np.zeros(N)
        sum_k[sort[:int(sum(output))+1]] = 1
        F1_sum += f1_score(sum_k,tar)

        rich_k = np.zeros(N)
        rich_k[sort[:int(sum(tar))+1]] = 1
        F1_rich += f1_score(rich_k,tar)

    print(F1/S, F1_top/S, F1_avg/S, F1_sum/S, F1_rich/S)
    # return (T_probas, P_probas, L_probas)

def main(x = 10, c = 0.66, a= 0.85):

    sol = p.read_csv('GLC24_SOLUTION_FILE.csv')['predictions'].to_numpy(dtype = str)
    '''
    sol = p.read_csv('/home/sebgl/Documents/Python/Malpolon/dataset/geolifeclef-2024/GLC24_PA_metadata_train.csv')
    sol = sol.dropna(subset="speciesId").reset_index(drop=True).sort_values('surveyId')
    sol['speciesId'] = sol['speciesId'].astype(int)
    sol = sol.groupby('surveyId')['speciesId'].apply(lambda x: ' '.join(x.astype(str))).values[:10000]
    '''
    pred_file = p.read_csv('submission_probas.csv', delimiter = ',').sort_values('surveyId')
    
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
    #tprobas, pprobas, lprobas = process(x,c,a,SOL, PROBAS)
    process(x,c,a,SOL, PROBAS)
    print(perf_counter()-start)

    # plt.hist(PROBAS[np.where(PROBAS != 0)].flatten(), color = 'blue', bins  = np.arange(0,1.01,0.01))
    # plt.hist(tprobas,  alpha = 0.5, color = 'red', bins  = np.arange(0,1.01,0.01))

    # plt.figure()
    # plt.hist(PROBAS[np.where(PROBAS != 0)].flatten(), color = 'blue', bins  = np.arange(0,1.01,0.01))
    # plt.hist(pprobas, alpha = 0.5, color = 'green', bins  = np.arange(0,1.01,0.01))

    # plt.figure()
    # plt.hist(PROBAS[np.where(PROBAS != 0)].flatten(), color = 'blue', bins  = np.arange(0,1.01,0.01))
    # plt.hist(lprobas, alpha = 0.5, color = 'orange', bins  = np.arange(0,1.01,0.01))


    #print(pearsonr(L_ri, L_sum), pearsonr(L_ri, L_k), pearsonr(L_sum, L_k))
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
    
    sol = p.read_csv('GLC24_SOLUTION_FILE.csv')['predictions'].to_numpy(dtype = str)
    '''
    sol = p.read_csv('/home/sebgl/Documents/Python/Malpolon/dataset/geolifeclef-2024/GLC24_PA_metadata_train.csv')
    sol = sol.dropna(subset="speciesId").reset_index(drop=True).sort_values('surveyId')
    sol['speciesId'] = sol['speciesId'].astype(int)
    sol = sol.groupby('surveyId')['speciesId'].apply(lambda x: ' '.join(x.astype(str))).values[:5000]
    '''
    N = 11255
    S = len(sol)
    SOL = np.zeros((S, N))
    for i in range(S) :
        row = sol[i]
        specs = row.split(' ')
        for spec in specs :
            SOL[i, int(spec)] = 1
        
    pred_file = p.read_csv('submission_probas.csv', delimiter = ',').sort_values('surveyId')
    
    probas = pred_file['probas'].to_numpy(dtype=str)
    preds = pred_file['predictions'].to_numpy(dtype=str)

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
        if len(idx) :
            Y.append(np.mean(SOL[idx]))
            sY.append(np.std(SOL[idx])/np.sqrt(len(idx[0])))
            X.append(np.mean(PRO[idx]))
    X, Y, sY = np.array(X), np.array(Y), np.array(sY)
    plt.scatter(X,Y)
    plt.plot(X,Y, c = 'blue')
    plt.plot(X,Y + 3*sY,'--', c = 'blue')
    plt.plot(X,Y - 3*sY,'--', c = 'blue')
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

if __name__ == '__main__':
    main(x=1, c=1, a=1)
    #calib_fig(x = 1, c = 1, a = 1)
    plt.show()