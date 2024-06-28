
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

Sc_t = np.load('models/Score_test.npy')
Sc_t_f1  = np.load('models/Score_F1_test.npy')
Sc = np.load('models/Score.npy')
Sc_f1 = np.load('models/Score_F1.npy')

N_clusters = [10,50,100,200,500,1000]

plt.plot(N_clusters,np.mean(Sc_t_f1, axis = 1))
plt.plot(N_clusters, np.mean(Sc_t, axis = 1))
plt.plot(N_clusters, np.mean(Sc_f1, axis = 1))
plt.plot(N_clusters, np.mean(Sc,axis = 1))
plt.show()