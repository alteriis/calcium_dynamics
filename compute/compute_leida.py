#!/usr/bin/env python
# coding: utf-8

# this code computes, for a gien window: 1. leading eigenvector timeseries 2. first eigenvalue timeseries 3. reconf speed timeseries


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore 
from scipy.sparse.linalg import eigsh

from skimage.measure import block_reduce

import sys
sys.path.append('../utils')
sys.path.append('../dFC')
import widefield_utils
import connectivity_measures


# In[2]:


import numpy as np
import os


# In[3]:


final_mask = np.load(r"/home/k21208334/calcium_analyses/data/dowsampled_tight_mask.npy")
good_indices = np.ravel(final_mask)


# In[4]:


h=300
w=304
H = 50
W = 51
connectivity_thr = 0.20

window_size = 30
n_comp = 20 #number of components for ICA
n_comp_T = 20 #number of components for ICA transpose

start = 0
end = 25000
connectivity_thr = 0.2

names = ["NXAK22.1A","NXAK14.1A","NXAK7.1B","NXAK4.1B","NRXN78.1D","NRXN78.1A"]
names_knock = ["NXAK24.1C","NXAK20.1B","NXAK16.1B","NXAK10.1A","NXAK4.1A","NRXN71.2A"]


# In[18]:


path = '/home/k21208334/calcium_analyses/data/leading_eigenvectors/window_size=' + str(window_size) + '/'
if not os.path.exists(path):
   os.makedirs(path)

for name in names+names_knock: 

    print("\n starting: ",name)
    walking = np.load("/home/k21208334/calcium_analyses/data/walking/"+name+".npy")
    base_directory = r"/home/k21208334/calcium_analyses/data/" + name + "/"
    registered_directory = r"/home/k21208334/calcium_analyses/data/registration_data/" + name + "/"
    sample = widefield_utils.load_registered_sample(base_directory,registered_directory,start,end)
    coarse_sample = block_reduce(sample, block_size=(6,6,1), func=np.mean) 
    #  connectivity: flatten data
    H = np.shape(coarse_sample)[0]
    W = np.shape(coarse_sample)[1]
    all_signals = np.reshape(coarse_sample,(H*W,end))
    # remove zero signals based on mask!
    good_indices = np.ravel(final_mask)
    signals = all_signals[good_indices,:]
    
    # ok now I have the signals
    leading_eigenvectors = np.empty((0,signals.shape[0]))
    T = signals.shape[1]
    reconf_speed = []
    reconf_speed.append(0)
    eigenvalue_timeseries = []

    for i in range(T):
        if i!=0: previous_matrix = matrix
        matrix = connectivity_measures.get_instantaneous_matrix(window_size,signals,i)
        if i!=0: reconf_speed.append(np.corrcoef(matrix.reshape(-1),previous_matrix.reshape(-1))[0,1])
        eigenvalue, eigenvector = eigsh(matrix, k=1)
        eigenvalue_timeseries.append(eigenvalue)
        if np.sum(eigenvector)>0:
            eigenvector=-eigenvector;
        # normalize eigenvector
        eigenvector = eigenvector/(np.linalg.norm(eigenvector))
        leading_eigenvectors=np.vstack((leading_eigenvectors,eigenvector.T))
        

    np.save(path + name,leading_eigenvectors)
    np.save(path + name + '_reconf_speed',reconf_speed)
    np.save(path + name + '_eigenvalue_timeseries',eigenvalue_timeseries)



# In[5]:


eigenvectors_all = np.empty((0,np.sum(final_mask))) 

# load clustered data 

for name in names+names_knock:

    data = path + name + ".npy"
    centr = np.load(data)
    eigenvectors_all = np.vstack((eigenvectors_all,centr))


# In[ ]:


from sklearn.decomposition import FastICA

model = FastICA(n_components = n_comp,whiten='unit-variance')
S = model.fit_transform(eigenvectors_all) # started 10.04

components = model.mixing_
np.save(path + 'ICA_signals_n_comp=' + str(n_comp) + '.npy',S)
np.save(path + 'ICA_components_n_comp=' + str(n_comp) + '.npy',components)

model_T = FastICA(n_components = n_comp_T,whiten='unit-variance')
S = model_T.fit_transform(eigenvectors_all.T) 

components = model_T.mixing_
np.save(path + 'ICA_T_signals_n_comp=' + str(n_comp_T) + '.npy',S)
np.save(path + 'ICA_T_components_n_comp=' + str(n_comp_T) + '.npy',components)
