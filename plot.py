
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib as mpl
from sklearn.metrics import r2_score
import pandas as pd

plt.style.use('tex')
c = {'cyan' : '#41a39e',
         'yellow' : '#e1ae36',
         'red' : '#ef4036',
         'lila' : '#a589ff',
         'green' : '#52b400',
         'pink' : '#f5766e',
         'blue' : '#00b7eb'}


Sc_t_f1  = np.load('models/Score_F1_test_nn.npy')
Sc_f1 = np.load('models/Score_F1_nn.npy')
N_clusters = np.array([10, 50, 100, 200,300, 400, 500,600, 700, 800, 900, 1000])
mask = [0,1,2,3,5,7, 11]


plt.plot(N_clusters, np.mean(Sc_f1, axis = 1), c = c['yellow'],linewidth = 2)
plt.plot(N_clusters,np.mean(Sc_t_f1, axis = 1), c = c['cyan'], linewidth = 2)


plt.scatter(Size, Score_cl, c = c['cyan'])

#plt.legend(['Score moyen train','Score moyen test' ],prop={'size': 24})
plt.ylabel(r'score $F_1$', fontsize =  36)
plt.xlabel(r'taille du cluster', fontsize =  36)
plt.tick_params(labelsize = 18)
plt.xlim(0,2900)
#plt.ylim(0,1)
plt.tight_layout()
plt.grid()
plt.show()

mask = ~(np.isnan(Score_cl))
print(r2_score(1 - np.array(Dist)[mask]**2/2, np.array(Score_cl)[mask]))

plt.show()

Norm = np.load('models/mc_norm.npy')
Uni = np.load('models/mc_uniform.npy')
Distr = np.load('models/mc_distri.npy')

Centr = np.load('models/centroid.npy')
Med = np.load('models/medoid.npy')
Bar = np.load('models/barycenter.npy')
Opti = np.load('models/opti.npy')
Centr_t = np.load('models/centroid_test.npy')
Med_t = np.load('models/medoid_test.npy')
Bar_t = np.load('models/barycenter_test.npy')
Opti_t = np.load('models/opti_test.npy')
T = np.arange(0,len(Norm[0]))

Size = np.bincount(train_cluster)
tot = sum(Size)

Centr = Size*Centr/tot
Med = Size*Med/tot
Bar = Size*Bar/tot
Opti = Size*Opti/tot
Norm = Size*Norm.T/tot
Distr = Size*Distr.T/tot
Uni = Size*Uni.T/tot


sb.lineplot(x = T, y = np.mean(Opti), c = c['red'], linewidth = 2)
# sb.lineplot(x = T, y = np.sum(Centr), c = c['pink'], linewidth = 2)
# sb.lineplot(x = T, y = np.sum(Bar), c = c['lila'], linewidth = 2)
# sb.lineplot(x = T, y = np.sum(Med), c = c['green'], linewidth = 2)
sb.lineplot(x = T, y = np.sum(Distr.T, axis = 0), c = c['blue'], linewidth = 2)
sb.lineplot(x = T, y = np.sum(Norm.T, axis = 0), c = c['cyan'], linewidth = 2)
sb.lineplot(x = T, y = np.sum(Uni.T, axis = 0), c = c['yellow'], linewidth = 2)

plt.legend(['Assemblage optimisé', '_no_legend',
            #'Assemblage itératif','_no_legend',
            #'Assemblage euclidien', '_no_legend',
            #'Assemblage médoïdes', '_no_legend',
            'Évolution distr. fréquentielle', '_no_legend',
              'Évolution distr. normalisée', '_no_legend',
              'Évolution distr. uniforme'],prop={'size': 24})
plt.ylabel(r'score $F_1$', fontsize =  36)
plt.xlabel(r'Nombre de réalisations aléatoires', fontsize =  36)
plt.xlim(0,1000)
plt.ylim(0.1,0.55)
plt.tick_params(labelsize = 18)
plt.tight_layout()
plt.grid()
plt.show()


Box = pd.DataFrame(np.concatenate((Med, Med_t,Bar, Bar_t, Centr,Centr_t,Opti, Opti_t)), columns= ["value"])
color = [c['yellow'], c['cyan']]
Num = len(Med) + len(Med_t)
Box['meth'] = pd.Series(['Medoïdes']*Num + ['Euclidien']*Num + ['Itératif']*Num + ['Optimal']*Num)
Box['hue'] = pd.Series(np.array([[False]*len(Med) + [True]*len(Med_t)]*4).flatten())


sb.boxplot(x = Box.meth, y = Box.value, hue = Box.hue, color = [[65, 163, 158],[225, 174, 54]])
plt.show()

