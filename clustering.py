from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import fclusterdata

class Clustering(MiniBatchKMeans):

    def __init__(self,n_clusters, n_init, verbose, batch_size, max_no_improvement):
        super().__init__(n_clusters, n_init= n_init,random_state= 42) #_clusters, n_init, verbose, batch_size, max_no_improvement)


    def fit(self,X) :
        normalize(X,'l2', axis = 1, copy = False)
        return super().fit(X).labels_

    
class HClustering():
    def __init__(self, n_clusters, metric, method):
        self.n_clusters = n_clusters
        self.metric = metric
        self.method = method
    
    def fit(self, X):
        return fclusterdata(X, self.n_clusters, 'maxclust', self.metric, 2, self.method)