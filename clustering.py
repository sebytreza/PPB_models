from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize

class Clustering(MiniBatchKMeans):

    def __init__(self,n_clusters, n_init, verbose, batch_size, max_no_improvement):
        super().__init__(n_clusters, n_init= n_init,random_state= 42) #_clusters, n_init, verbose, batch_size, max_no_improvement)


    def normed_fit(self,X) :
        normalize(X,'l2', axis = 1, copy = False)
        return self.fit(X)
    
    def normed_predict(self,X) :
        pass
        return self.predict(X)
    
