from datetime import datetime
import widefield_utils

import numpy as np
import matplotlib
import scipy

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.colors import Normalize
import h5py
import tables
from scipy import signal, ndimage, stats
from scipy.sparse.linalg import eigsh
import os
import cv2
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA, FactorAnalysis, TruncatedSVD
from skimage.transform import resize
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import time
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle
from skimage.measure import block_reduce

# functions for dynamic functional connectivity analyses


def create_connectivity_video(window_size,sample):
    
    n_time_points = np.shape(sample)[2]
    subsample = block_reduce(sample, block_size=(6,6,1), func=np.mean) 
    # compute time varying connectivity: flatten data

    signals = np.reshape(subsample,(np.shape(subsample)[0]*np.shape(subsample)[1],n_time_points))
    print('shape signals = ',np.shape(signals))
    # remove zero signals
    power = np.std(signals,axis=1) #standard dev of signal. I'll discard zero std
    signals = signals[power!=0,:]
    
    connectivity_evolution = np.zeros((np.shape(signals)[0],np.shape(signals)[0],n_time_points))
    
    for x in range(n_time_points):
        left = int(max(x-window_size/2,0))
        right = int(min(x+window_size/2,n_time_points))
        connectivity_evolution[:,:,x] = np.corrcoef(signals[:,left:right])
        
    return connectivity_evolution


def get_instantaneous_matrix(window_size,signals,timepoint):
    
    n_time_points = np.shape(signals)[1]
    left = int(max(timepoint-window_size/2,0))
    right = int(min(timepoint+window_size/2,n_time_points))
    instantaneous_matrix = np.corrcoef(signals[:,left:right])
        
    return instantaneous_matrix

def compute_eig1_trace(signals,window_size):
    
    n_time_points = np.shape(signals)[1]
    eig1_trace = np.zeros(n_time_points)
    for x in range(n_time_points):
        instantaneous_conn = get_instantaneous_matrix(window_size,signals,x)
        eigenvalue, eigenvector = eigsh(instantaneous_conn, k=1)
        eig1_trace[x] = eigenvalue
        
    return eig1_trace
    
def compute_eig5_trace(signals,window_size):
    
    n_time_points = np.shape(signals)[1]
    eig5_trace = np.zeros((5,n_time_points))
    for x in range(n_time_points):
        instantaneous_conn = get_instantaneous_matrix(window_size,signals,x)
        eigenvalues, eigenvector = eigsh(instantaneous_conn, k=5)
        for i in range(5):
            eig5_trace[i,x] = eigenvalues[i]
        
    return eig5_trace

def show_array(degree,H,W,good_indices,limits=None): # this fcn takes a linearized array of downsampled signals and re-plots it into an image, by filling with zero values outside. Good indices are the indices of the mask in the downsampled brain, H,W is the size of the downsampled image
    degree_all = np.zeros(H*W)
    degree_all[good_indices] = degree
    fig,ax = plt.subplots()
    if limits is not None:
        ax.imshow(np.reshape(degree_all,(H,W)),vmin=limits[0],vmax=limits[1],cmap='bwr')
        fig.colorbar(plt.cm.ScalarMappable(matplotlib.colors.Normalize(vmin=limits[0],vmax=limits[1]), cmap='bwr'), ax=ax)
    else:
        ax.imshow(np.reshape(degree_all,(H,W)))
    return fig
    
    
def create_sample_video_with_mousecam(delta_f_directory, bodycam_directory, output_directory,sample_start,sample_length,window_size):

    sample_end = sample_start + sample_length

    # Get Delta F Sample
    print("Getting Delta F Sample", datetime.now())
    delta_f_sample = widefield_utils.get_delta_f_sample_from_svd(delta_f_directory, sample_start, sample_end)
    print("Finished Getting Delta F Sample", datetime.now())
    
    # get "raw" delta F and create stacked signals
    sample = widefield_utils.get_delta_f_sample_from_svd_unprocessed(delta_f_directory,sample_start,sample_end)
    signals = widefield_utils.get_signals_from_svd_sample(sample)
    
    # get walking information
    walking = widefield_utils.get_walking_series(bodycam_directory,20,0.03)

    # Get Mousecam Sample
    print("Getting Mousecam Sample", datetime.now())
    bodycam_filename = widefield_utils.get_bodycam_filename(bodycam_directory)
    eyecam_filename = widefield_utils.get_eyecam_filename(bodycam_directory)
    bodycam_sample = widefield_utils.get_mousecam_sample(bodycam_directory, bodycam_filename, sample_start, sample_end)
    eyecam_sample = widefield_utils.get_mousecam_sample(bodycam_directory, eyecam_filename, sample_start, sample_end)
    print("Finished Getting Mousecam Sample", datetime.now())

    # Create Colourmaps
    widefield_colourmap = widefield_utils.get_musall_cmap()
    widefield_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.05, vmax=0.05), cmap=widefield_colourmap)
    mousecam_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=255), cmap=cm.get_cmap('Greys_r'))

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_generous_mask(bodycam_directory)
    background_pixels = widefield_utils.get_background_pixels(indicies, image_height, image_width)
    print("Loaded Mask", datetime.now())

    # Create Video File
    video_name = os.path.join(output_directory, "Brain_connect_Video.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(1500, 500), fps=30)  # 0, 12

    figure_1 = plt.figure(figsize=(15, 5))
    canvas = FigureCanvasAgg(figure_1)
    for frame_index in tqdm(range(sample_length)):

        rows = 1
        columns = 5
        brain_axis = figure_1.add_subplot(rows, columns, 1)
        body_axis = figure_1.add_subplot(rows, columns, 2)
        eye_axis = figure_1.add_subplot(rows, columns, 3)
        connectivity_axis = figure_1.add_subplot(rows,columns,4)
        histogram_axis = figure_1.add_subplot(rows,columns,5)

        # Extract Frames
        brain_frame = delta_f_sample[frame_index]
        body_frame = bodycam_sample[frame_index]
        eye_frame = eyecam_sample[frame_index]
        # create connectivity matrix and use as frame 
        connectivity_frame = get_instantaneous_matrix(window_size,signals,frame_index)
        # create histogram of eigenvalues
        eigenvalues, eigenvectors = eigsh(connectivity_frame, k=window_size+5)
        ordered_eigs = eigenvalues[::-1]


        # Set Colours
        brain_frame = widefield_colourmap.to_rgba(brain_frame)
        body_frame = mousecam_colourmap.to_rgba(body_frame)
        eye_frame = mousecam_colourmap.to_rgba(eye_frame)
        #brain_frame[background_pixels] = (1,1,1,1)

        # Display Images
        brain_axis.imshow(brain_frame)
        body_axis.imshow(body_frame)
        eye_axis.imshow(eye_frame)
        connectivity_axis.imshow(connectivity_frame,vmin=-1,vmax=1)
        histogram_axis.plot(ordered_eigs[0:window_size+4]/ordered_eigs[0])

        # Remove Axis
        brain_axis.axis('off')
        body_axis.axis('off')
        if walking[sample_start+frame_index]:
            histogram_axis.set_title('walking')
        eye_axis.axis('off')
   
        
        figure_1.canvas.draw()

        # Write To Video
        canvas.draw()
        buf = canvas.buffer_rgba()
        image_from_plot = np.asarray(buf)
        image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
        video.write(image_from_plot)


        plt.clf()

    cv2.destroyAllWindows()
    video.release()
    
def create_video_dFC(name,window_size,sample_start,sample_length,output_directory,n_clust,n_ica): # works with new ica t algo
    
    base_directory = r"/home/k21208334/calcium_analyses/data/" + name + "/"
    bodycam_directory = base_directory
    registered_directory = r"/home/k21208334/calcium_analyses/data/registration_data/" + name + "/"
    leida_path = '/home/k21208334/calcium_analyses/data/leading_eigenvectors/window_size=' + str(window_size) + '/'
    final_mask = np.load(r"/home/k21208334/calcium_analyses/data/dowsampled_tight_mask.npy")
    good_indices = np.ravel(final_mask)
    
    leading_eigen = np.load(leida_path + name + '.npy')
    labels = np.load(leida_path + name + '_k=' + str(n_clust) + '_ica_T=' + str(n_ica) + '_labels.npy')
    centroids = np.load(leida_path + 'k=' + str(n_clust) + '_ica_T=' + str(n_ica) + '_centroids.npy')
    
    sample_end = sample_start + sample_length
    leading_eigen = leading_eigen[sample_start:sample_end,:]
    labels = labels[sample_start:sample_end]

    # Get Delta F Sample
    print("Getting Delta F Sample", datetime.now())
    delta_f_sample = widefield_utils.get_delta_f_sample_from_svd(base_directory, sample_start, sample_end)
    print("Finished Getting Delta F Sample", datetime.now())
    
    # get raw signals 
    sample = widefield_utils.load_registered_sample(base_directory,registered_directory,sample_start,sample_end)
    coarse_sample = block_reduce(sample, block_size=(6,6,1), func=np.mean) 
    #  connectivity: flatten data
    H = np.shape(coarse_sample)[0]
    W = np.shape(coarse_sample)[1]
    all_signals = np.reshape(coarse_sample,(H*W,sample_end))
    # remove zero signals based on mask!
    signals = all_signals[good_indices,:]

    # Get Mousecam Sample
    print("Getting Mousecam Sample", datetime.now())
    bodycam_filename = widefield_utils.get_bodycam_filename(bodycam_directory);
    eyecam_filename = widefield_utils.get_eyecam_filename(bodycam_directory);
    bodycam_sample = widefield_utils.get_mousecam_sample(bodycam_directory, bodycam_filename, sample_start, sample_end);
    eyecam_sample = widefield_utils.get_mousecam_sample(bodycam_directory, eyecam_filename, sample_start, sample_end);
    print("Finished Getting Mousecam Sample", datetime.now())

    # Create Colourmaps
    widefield_colourmap = widefield_utils.get_musall_cmap();
    widefield_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=-0.05, vmax=0.05), cmap=widefield_colourmap)
    mousecam_colourmap = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=255), cmap=cm.get_cmap('Greys_r'))

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_generous_mask(bodycam_directory);
    background_pixels = widefield_utils.get_background_pixels(indicies, image_height, image_width);
    print("Loaded Mask", datetime.now())

    # Create Video File
    video_name = os.path.join(output_directory, name+"_Brain_connect_Video.avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(1500, 500), fps=30)  # 0, 12

    figure_1 = plt.figure(figsize=(15, 5))
    canvas = FigureCanvasAgg(figure_1)
    for frame_index in tqdm(range(sample_length)):

        rows = 2
        columns = 4
        brain_axis = figure_1.add_subplot(rows, columns, 2)
        body_axis = figure_1.add_subplot(rows, columns, 1)
        eye_axis = figure_1.add_subplot(rows, columns, 5)
        connectivity_axis = figure_1.add_subplot(rows,columns,4)
        leida_axis = figure_1.add_subplot(rows,columns,3)
        leida_matrix_axis = figure_1.add_subplot(rows,columns,8)
        current_cluster_axis = figure_1.add_subplot(rows,columns,7)
        
        # Extract Frames
        brain_frame = delta_f_sample[frame_index]
        body_frame = bodycam_sample[frame_index]
        eye_frame = eyecam_sample[frame_index]
        # create connectivity matrix and use as frame 
        connectivity_frame = get_instantaneous_matrix(window_size,signals,frame_index)
        leida_frame = np.zeros(H*W)
        leida_frame[good_indices] = leading_eigen[frame_index,:]
        leida_frame = np.reshape(leida_frame,(H,W))
        leida_frame = np.ma.masked_where(leida_frame == 0, leida_frame)
        leida_matrix_frame = np.outer(leading_eigen[frame_index,:],leading_eigen[frame_index,:])
        current_cluster_frame = np.zeros(H*W)
        current_cluster_frame[good_indices] = centroids.T[labels[frame_index],:]
        current_cluster_frame = np.reshape(current_cluster_frame,(H,W))
        current_cluster_frame = np.ma.masked_where(current_cluster_frame == 0, current_cluster_frame)
        
        # Set Colours
        brain_frame = widefield_colourmap.to_rgba(brain_frame)
        body_frame = mousecam_colourmap.to_rgba(body_frame)
        eye_frame = mousecam_colourmap.to_rgba(eye_frame)
        #brain_frame[background_pixels] = (1,1,1,1)

        # Display Images
        brain_axis.imshow(brain_frame)
        body_axis.imshow(body_frame)
        eye_axis.imshow(eye_frame)
        connectivity_axis.imshow(connectivity_frame,vmin=-1,vmax=1)
        leida_axis.imshow(leida_frame,cmap='bwr')
        leida_matrix_axis.imshow(leida_matrix_frame)
        current_cluster_axis.imshow(current_cluster_frame,cmap='bwr')
        
        # Remove Axis
        brain_axis.axis('off')
        body_axis.axis('off')
        eye_axis.axis('off')
        connectivity_axis.axis('off')
        leida_axis.axis('off')  
        leida_matrix_axis.axis('off')
        current_cluster_axis.axis('off')
   
        
        figure_1.canvas.draw()

        # Write To Video
        canvas.draw()
        buf = canvas.buffer_rgba()
        image_from_plot = np.asarray(buf)
        image_from_plot = cv2.cvtColor(image_from_plot, cv2.COLOR_RGB2BGR)
        video.write(image_from_plot)


        plt.clf()

    cv2.destroyAllWindows()
    video.release()

