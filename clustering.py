from sklearn.mixture import GaussianMixture
from sklearn import manifold
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import os


def cluster(file, cluster_num_list, save_dir):
  data=np.load(file)
  tsne2d = manifold.TSNE(n_components=2, init='pca', random_state=1000)
  vis2d = tsne2d.fit_transform(data)

  for num in cluster_num_list:
    gm = GaussianMixture(n_components=num).fit(data)
    pred = gm.predict(data)
    np.save(f'{save_dir}/{file}_{num}_pred.npy', pred)

    df2d = DataFrame({'x':vis2d[:,0], 'y':vis2d[:,1], 'label':pred})
    groups2d = df2d.groupby('label')

    ig, ax = plt.subplots()
    for name, group in groups2d:
      ax.scatter(group.x, group.y, s=1, label=name)

    ax.legend()
    plt.savefig(f'{save_dir}/{file}_{num}.png')
    plt.show()

cluster('zh_resid_2015-08.npy', [10,30,50,100], 'Gaussian_Mixture_result')
