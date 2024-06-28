from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import fclusterdata
import numpy as np
from tqdm import tqdm
from functions import f1_score
from sklearn_extra.cluster import KMedoids



class Clustering(MiniBatchKMeans):

    def __init__(self,n_clusters, n_init, verbose, batch_size, max_no_improvement):
        super().__init__(n_clusters, n_init= n_init,random_state= 42) #_clusters, n_init, verbose, batch_size, max_no_improvement)

    def predict(self,X) : 
        centers = self.cluster_centers_
        self.labels_ = np.zeros(len(X)).astype('int')
        for i in tqdm(range(len(X))):
            self.labels_[i] = int(np.argmax([f1_score(X[i], center) for center in centers]))
        return self.labels_
        

    def fit(self,X) :
        normalize(X,'l2', axis = 1, copy = False)
        super().fit(X)
        return self.predict(X)
        
    
    
class HClustering():
    def __init__(self, n_clusters, metric, method):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
    
    def fit(self, X):
        return fclusterdata(X, self.n_clusters, 'maxclust', self.metric, 2, self.method)
    

class MClustering(KMedoids):
    def __init__(self, n_clusters, metric, method):
        super().__init__(n_clusters, metric, method)

    def fit(self, X):
        epochs = 10
        batchsize = 5000
        for i in tqdm(range(epochs)):
            batch_id = np.random.randint(0, len(X), batchsize)
            super().fit(X[batch_id])
        
        return super().predict(X)
