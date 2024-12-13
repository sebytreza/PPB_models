
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import matplotlib as mpl
from sklearn.metrics import r2_score
import pandas as pd
from scipy.stats import wilcoxon as wlw_test
from scipy.stats import f_oneway
from scipy.stats import spearmanr


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


palette = mpl.colors.LinearSegmentedColormap.from_list("",[c['cyan'],c['yellow']])
sb.scatterplot(x = Dist, y = Score_cl, c = c['cyan'])#, palette = palette, hue = Size)
sb.lineplot(x = [0,20], y = mean, c = 'grey')
plt.annotate(r'$\bar F_1 =$ '+ str(round(mean,2)), xy = [0.5, mean+0.02],size = 36, c= 'grey')

#plt.legend(['Score moyen train','Score moyen test' ],prop='size': 24})
plt.ylabel(r'Score $F_1$', fontsize =  36)
plt.xlabel(r'Distance $L^2$', fontsize =  36)
plt.tick_params(labelsize = 18)
plt.ylim(0,1)
plt.xlim(0,1)
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


def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


Box = pd.DataFrame(np.concatenate((Med, Med_t,Bar, Bar_t, Centr,Centr_t,Opti, Opti_t)), columns= ["value"])
color = [c['yellow'], c['cyan']]
Num = len(Med) + len(Med_t)
Box['meth'] = pd.Series(['Medoïdes']*Num + ['Euclidien']*Num + ['Itératif']*Num + ['Optimal']*Num)
Box['hue'] = pd.Series(np.array([['Train']*len(Med) + ['Test']*len(Med_t)]*4).flatten())


sb.boxplot(x = Box.meth, y = Box.value, hue = Box.hue, palette = color)
plt.xlabel(r'Méthode de construction', fontsize = 36)
plt.ylabel(r'score $F_1$', fontsize = 36)
plt.legend(prop={'size': 24})
plt.tick_params(labelsize = 18)
plt.tight_layout()
plt.grid(axis = 'y')
plt.show()



plt.scatter(f1_bar, f1, c = c['cyan'], s = 1)
plt.xlabel(r'Score $F_1$', fontsize = 36)
plt.ylabel(r'Score $\bar F_1$', fontsize = 36)
plt.tick_params(labelsize = 18)
plt.xlim(0,1)
plt.ylim(0,1)
plt.tight_layout()
plt.grid()
plt.show()

r2_score(f1,f1_bar)

sb.histplot(np.sum(cluster_dt, axis = 1), bins = 50, color = c['cyan'])
plt.xlim(0,100)
plt.xlabel(r"Nombre d'espèces", fontsize = 62)
plt.ylabel(r"Nombre d'observations", fontsize = 62)
plt.tick_params(labelsize = 28)
plt.show()


classif= np.load("loss/Evo_F1_classif.npy")
classif_soft = np.load("loss/Evo_F1_classif_soft.npy")
baseline_1 = np.load("loss/Evo_F1_baseline_ct.npy")
baseline_2 = np.load("loss/Evo_F1_baseline_proj.npy")
baseline_3 = np.load("loss/Evo_F1_baseline_topk.npy")

X = np.arange(1,len(baseline_1)+1)
plt.plot(X,classif)
plt.plot(X,classif_soft)
plt.plot(X,baseline_3)
plt.plot(X,baseline_2)


plt.legend(['Classification directe','Classification pondérée','Baseline top25', 'Baseline modifiée'],prop={'size': 24})
plt.xlabel(r"Epoques", fontsize = 36)
plt.xlim(1,len(X))
plt.ylabel(r"Score $F_1$", fontsize = 36)
plt.tick_params(labelsize = 18)
plt.xticks(X)
plt.grid()
plt.show()