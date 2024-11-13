import numpy as np
import matplotlib.pyplot as plt
import pandas as p
from tqdm import tqdm
from functions import f1_score
import math

def decide(p):
    p = np.sort(p)[::-1]
    n = len(p)
    Fmax, kmax  = 0, 0
    C = np.zeros((n+1,n+1))
    C[0,0] = 1
    
    for i in range(n):
        C[i+1,0] = (1 - p[i])*C[i,0]
        for j in range(i+1):
            C[i+1,j+1] = p[i]*C[i,j] + (1 - p[i])*C[i,j+1]
              
    S = np.zeros(2*n)      
    for i in range(2*n):
        S[i] = 1/(i+1)
    
    for k in range(n):
        F = 0
        K = n - k
        for i in range(1, K+1):
            F += 2*i*C[K,i]*S[i + K -1]
        for i in range(2*(K-1)):
            S[i] = p[K-1]*S[i+1] + (1 - p[K-1])*S[i]
            
        if F >= Fmax :
            Fmax, kmax  = F, K
        
    return kmax, Fmax

def main(x, c = 1):    
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

    F1_ref = np.zeros(50)
    F1 = 0
    a = 0.85
    T = lambda x : x**a/(x**a + (1 - x)**a)
    PRO = PROBAS/(PROBAS + x*(1 - PROBAS))
    PRO = c*T(PRO)
    Plim = []
    for i in tqdm(range(S)): 
        output = PRO[i]
        tar = SOL[i]
        
        sort = np.argsort(-output)
        
        mask = np.where(output > 0)[0]
        K, _ = decide(output[mask])
        '''
        topk = np.zeros(N)
        for j in range(50):
            topk[sort[j]] = 1
            f1_ref = f1_score(tar, topk)
            F1_ref[j] += f1_ref
        '''
        pred = np.zeros(N)
        pred[sort[:K]] = 1
        Plim.append(output[sort[K-1]])
        f1 = f1_score(pred,tar)
        F1 += f1
        #print(K, int(np.sum(tar)), f1 > f1_ref)
    plt.figure()
    calib_fig(x,a)
    plt.scatter(Plim, Plim, c = 'red',marker = '.', s = 10)
    print(F1/S, max(F1_ref)/S)
    return (F1/S)




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
    
def calib_fig(x= 10, c =1):
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
    a = 0.85
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
    X,Y, sY = np.array(X), np.array(Y), np.array(sY)
    plt.scatter(X,Y)
    plt.plot(X,Y,c = 'blue')
    plt.plot(X,Y + 3*sY,'--',c = 'blue')
    plt.plot(X,Y - 3*sY,'--',c = 'blue')
    plt.plot(X,Y,c = 'blue')

    plt.plot(X,X, c = 'orange')
    plt.gca().set_aspect('equal')

def temp(A):
    X = np.arange(0,1.001,0.01)
    fa = lambda a,x : x**a/(x**a + (1-x)**a)
    plt.plot(X,X)
    for alpha in A:
            plt.plot(X,fa(alpha,X))
    plt.gca().set_aspect('equal')
    plt.show()
    
#calib_fig(x = 10, c = 0.66)
main(x = 10, c = 0.66)

plt.show()
'''   
mini = 0.1
index = []
F1 = []
x = 0.1
while x != 0.001:
    for i in range(10):
        index.append(i*x + mini)
        F1.append(main(10,i*x + mini))
    sort = np.array(index)[np.argsort(F1)]
    if not math.isclose(abs(sort[-1] - sort[-2]),x):
        print("NON CONVEXE")
        break
    else:
        mini = min(sort[-1], sort[-2])
    x = x / 10

plt.show()
'''