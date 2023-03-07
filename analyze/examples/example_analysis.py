#!/usr/bin/env python
# coding: utf-8

# #this notebook is an example of how to open the preprocessed data and show it

# In[4]:


import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.measure import block_reduce

base_directory = r"/home/k21208334/calcium_analyses/data/NXAK22.1A/"


# In[58]:


# Load SVT
svt = np.load(os.path.join(base_directory,"Corrected_SVT.npy"))
u = np.load(os.path.join(base_directory,"U.npy"))
print(np.shape(svt))
print(np.shape(u))

toshow = 1000; #i show only a subset of timepoints
window_size = 20 # this is window size for sliding window correlation!
sample = np.dot(u, svt[:, 0:toshow])
print(np.shape(sample))


# In[59]:


subsample = block_reduce(sample, block_size=(6, 6, 1), func=np.mean)


# In[60]:


# compute time varying connectivity: flatten data

signals = np.reshape(subsample,(np.shape(subsample)[0]*np.shape(subsample)[1],toshow))
np.shape(signals)
# remove zero signals
power = np.std(signals,axis=1) #standard dev of signal. I'll discard zero std
signals = signals[power!=0,:]


# In[66]:


plt.ion()

for x in range(toshow):
    left = int(max(x-window_size/2,0))
    right = int(min(x+window_size/2,toshow))
    #axs[0].imshow(subsample[:, :, x],cmap="viridis",vmin=np.min(subsample),vmax=np.max(subsample))
    plt.imshow(np.corrcoef(signals[:,left:right]),cmap="viridis",vmin = -1,vmax = 1)
    plt.draw()
    plt.pause(0.1)
    plt.clf()


# In[ ]:




