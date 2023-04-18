import numpy as np
import tables
import os
import pandas as pd
from scipy import ndimage
from skimage.transform import resize, rescale
from skimage.feature import canny
from skimage.morphology import binary_opening, binary_erosion, binary_dilation
from matplotlib.colors import LinearSegmentedColormap
import pathlib
import pkg_resources
from bisect import bisect_left

import matplotlib.pyplot as plt
from skimage.measure import block_reduce
import scipy
import cv2

"""
def get_package_file_directory():
    cwd = pathlib.Path.cwd()
    package_root_directory = list(cwd.parts[:-1])
    package_root_directory.append("Files")
    package_file_directory = package_root_directory[0]
    for item in package_root_directory[1:]:
        package_file_directory = os.path.join(package_file_directory, item)

    return package_file_directory
"""

def get_session_folder_in_tensor_directory(base_directory, save_directory_root):

    # Create Output Directory
    split_base_directory = os.path.normpath(base_directory)
    split_base_directory = split_base_directory.split(os.sep)
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]

    # Check Mouse Directory Exists
    mouse_directory = os.path.join(save_directory_root, mouse_name)
    if not os.path.exists(mouse_directory):
        os.mkdir(mouse_directory)

    # Check Session Save Directory Exists
    output_directory = os.path.join(mouse_directory, session_name)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    return output_directory




def get_mask_edge_pixels(indicies, image_height, image_width):
    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    edges = canny(template)
    edge_indicies = np.nonzero(edges)
    return edge_indicies


def get_atlas_outline_pixels():

    # Load Atlas
    atlas_outline = np.load("/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Outlines.npy")

    # Load Atlas Transformation Dict
    transformation_dict = np.load(r"/home/matthew/Documents/Github_Code_Clean/Widefield_Analysis/Files/Atlas_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Atlas
    atlas_outline = transform_mask_or_atlas(atlas_outline, transformation_dict)

    atlas_pixels = np.nonzero(atlas_outline)
    return atlas_pixels



def get_background_pixels(indicies, image_height, image_width):
    template = np.ones(image_height * image_width)
    template[indicies] = 0
    template = np.reshape(template, (image_height, image_width))
    background_pixels = np.nonzero(template)
    return background_pixels


def get_best_grid(n_items):

    """ Return The Best Arrangement Of Rows and Columns To Display N Items In A Grid """

    # Take Square Root
    square_root = np.sqrt(n_items)
    print("Square Root", square_root)

    # Floor This
    floor_root = np.floor(square_root)
    print("Root Floor", floor_root)

    # Divide By This Floor
    divisor = np.divide(n_items, floor_root)
    print("Divisor", divisor)

    # Take Ceiling Of This
    divisor_ceiling = np.ceil(divisor)
    print("Divisor Celing", divisor_ceiling)

    return int(floor_root), int(divisor_ceiling)



def take_closest(myList, myNumber):

    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def get_mouse_name_and_session_name(base_directory):

    # Create Output Directory
    split_base_directory = os.path.normpath(base_directory)
    split_base_directory = split_base_directory.split(os.sep)
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]

    return mouse_name, session_name

def load_downsampled_mask(base_directory):
    mask_dict = np.load(os.path.join(base_directory, "Downsampled_mask_dict.npy"), allow_pickle=True)[()]
    indicies = mask_dict["indicies"]
    image_height = mask_dict["image_height"]
    image_width = mask_dict["image_width"]
    return indicies, image_height, image_width

"""
def load_within_mouse_aligned_mask(base_directory):

    # Load Downsampled Mask
    indicies, image_height, image_width = load_downsampled_mask(base_directory)

    # Recreate Image
    template = np.zeros((image_height * image_width))
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))

    # Load Alignment Dictionary
    wihin_mouse_alignment_dictionary = np.load(os.path.join(base_directory, "Within_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    # Transform Mask
    template = transform_image(template, wihin_mouse_alignment_dictionary)

    # Get New Indicies
    template = np.where(template > 0.1, 1, 0)
    template = np.reshape(template, (image_height * image_width))
    indicies = np.nonzero(template)

    return indicies, image_height, image_width
"""

def load_all_sessions_of_type(session_type):

    # This Is The Location of My Experimental Logbook
    logbook_file_location = r"/home/matthew/Documents/Experiment_Logbook.ods"

    #  Read Logbook As A Dataframe
    logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

    # Return A List Of the File Directories Of Sessions Matching The Mouse Name and Session Type
    selected_sessions = logbook_dataframe.loc[(logbook_dataframe["Session Type"] == session_type), ["Filepath"]].values.tolist()

    # Flatten The Subsequent Nested List
    selected_sessions = flatten(selected_sessions)

    return selected_sessions

def get_bodycam_filename(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        file_split = file.split('_')
        if file_split[-1] == '1.mp4' and file_split[-2] == 'cam':
            return file

def get_eyecam_filename(base_directory):

    file_list = os.listdir(base_directory)

    for file in file_list:
        file_split = file.split('_')
        if file_split[-1] == '2.mp4' and file_split[-2] == 'cam':
            return file



def load_across_mice_alignment_dictionary(base_directory):

    # Get Root Directory
    base_directory_parts = pathlib.Path(base_directory)
    base_directory_parts = list(base_directory_parts.parts)
    root_directory = base_directory_parts[0]
    for subfolder in base_directory_parts[1:-1]:
        root_directory = os.path.join(root_directory, subfolder)

    # Load Alignment Dictionary
    across_mouse_alignment_dictionary = np.load(os.path.join(root_directory, "Across_Mouse_Alignment_Dictionary.npy"), allow_pickle=True)[()]

    return across_mouse_alignment_dictionary



def downsample_mask_further(indicies, image_height, image_width, downsample_size=100):
    template = np.zeros((image_height*image_width))
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = resize(template, (downsample_size, downsample_size), anti_aliasing=True)
    template = np.reshape(template, (downsample_size * downsample_size))
    template = np.where(template > 0.5, 1, 0)
    template_indicies = np.nonzero(template)
    return template_indicies, downsample_size, downsample_size



def flatten(l):
    return [item for sublist in l for item in sublist]


def load_mouse_sessions(mouse_name, session_type):

    # This Is The Location of My Experimental Logbook
    logbook_file_location = r"/home/matthew/Documents/Experiment_Logbook.ods"

    #  Read Logbook As A Dataframe
    logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

    # Return A List Of the File Directories Of Sessions Matching The Mouse Name and Session Type
    selected_sessions = logbook_dataframe.loc[(logbook_dataframe["Mouse"] == mouse_name) & (logbook_dataframe["Session Type"] == session_type), ["Filepath"]].values.tolist()

    # Flatten The Subsequent Nested List
    selected_sessions = flatten(selected_sessions)

    return selected_sessions

def get_chunk_structure(chunk_size, array_size):
    number_of_chunks = int(np.ceil(array_size / chunk_size))
    remainder = array_size % chunk_size

    # Get Chunk Sizes
    chunk_sizes = []
    if remainder == 0:
        for x in range(number_of_chunks):
            chunk_sizes.append(chunk_size)

    else:
        for x in range(number_of_chunks - 1):
            chunk_sizes.append(chunk_size)
        chunk_sizes.append(remainder)

    # Get Chunk Starts
    chunk_starts = []
    chunk_start = 0
    for chunk_index in range(number_of_chunks):
        chunk_starts.append(chunk_size * chunk_index)

    # Get Chunk Stops
    chunk_stops = []
    chunk_stop = 0
    for chunk_index in range(number_of_chunks):
        chunk_stop += chunk_sizes[chunk_index]
        chunk_stops.append(chunk_stop)

    return number_of_chunks, chunk_sizes, chunk_starts, chunk_stops

def invert_dictionary(dictionary):
    inv_map = {v: k for k, v in dictionary.items()}
    return inv_map


def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Photodiode"        :0,
        "Reward"            :1,
        "Lick"              :2,
        "Visual 1"          :3,
        "Visual 2"          :4,
        "Odour 1"           :5,
        "Odour 2"           :6,
        "Irrelevance"       :7,
        "Running"           :8,
        "Trial End"         :9,
        "Camera Trigger"    :10,
        "Camera Frames"     :11,
        "LED 1"             :12,
        "LED 2"             :13,
        "Mousecam"          :14,
        "Optogenetics"      :15,
        }

    return channel_index_dictionary

def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))
    return image

def get_musall_cmap():
    cmap = LinearSegmentedColormap.from_list('mycmap', [

        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 1),

    ])

    return cmap



def transform_image(image, variable_dictionary, invert=False):

    # Settings
    background_size = 1000
    background_offset = 200
    origional_height, origional_width = np.shape(image)
    window_y_start = background_offset
    window_y_stop = window_y_start + origional_height
    window_x_start = background_offset
    window_x_stop = window_x_start + origional_width

    # Unpack Transformation Details
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    scale_factor = variable_dictionary['zoom']

    # Inverse
    if invert == True:
        angle = -1 * angle
        x_shift = -1 * x_shift
        y_shift = -1 * y_shift
        scale_factor = 1 - scale_factor
    else:
        scale_factor = 1 + scale_factor

    # Copy
    transformed_image = np.copy(image)

    # Scale
    transformed_image = rescale(transformed_image, scale_factor, anti_aliasing=False, preserve_range=True)

    # Rotate
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

    # Translate
    background = np.zeros((background_size, background_size))
    new_height, new_width = np.shape(transformed_image)

    y_start = background_offset + y_shift
    y_stop = y_start + new_height

    x_start = background_offset + x_shift
    x_stop = x_start + new_width

    background[y_start:y_stop, x_start:x_stop] = transformed_image

    # Get Normal Sized Window
    transformed_image = background[window_y_start:window_y_stop, window_x_start:window_x_stop]

    return transformed_image


def get_session_name(base_directory):
    # Create Output Directory
    split_base_directory = os.path.normpath(base_directory)
    split_base_directory = split_base_directory.split(os.sep)
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]
    return mouse_name + "_" + session_name

def check_save_directory(base_directory, save_directory_root):

    # Create Output Directory
    split_base_directory = os.path.normpath(base_directory)
    split_base_directory = split_base_directory.split(os.sep)
    mouse_name = split_base_directory[-2]
    session_name = split_base_directory[-1]

    # Check Mouse Directory Exists
    mouse_directory = os.path.join(save_directory_root, mouse_name)
    if not os.path.exists(mouse_directory):
        os.mkdir(mouse_directory)

    # Check Session Save Directory Exists
    output_directory = os.path.join(mouse_directory, session_name)
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    return output_directory


# Load Generous Mask
def load_generous_mask(base_directory):

    # Loads the mask for a video, returns a list of which pixels are included, as well as the original image height and width
    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width





def transform_atlas_regions(image, variable_dictionary):


    unique_values = list(set(np.unique(image)))

    transformed_mask = np.zeros(np.shape(image))

    for value in unique_values:
        value_mask = np.where(image == value, 1, 0)
        value_mask = transform_mask_or_atlas_300(value_mask, variable_dictionary)
        value_indicies = np.nonzero(value_mask)
        transformed_mask[value_indicies] = value

    return transformed_mask



def transform_mask_or_atlas_300(image, variable_dictionary):

    image_height = 300
    image_width = 304

    # Unpack Dictionary
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    x_scale = variable_dictionary['x_scale']
    y_scale = variable_dictionary['y_scale']

    # Copy
    transformed_image = np.copy(image)

    # Scale
    original_height, original_width = np.shape(transformed_image)
    new_height = int(original_height * y_scale)
    new_width = int(original_width * x_scale)
    transformed_image = resize(transformed_image, (new_height, new_width), preserve_range=True)

    # Rotate
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

    # Insert Into Background
    mask_height, mask_width = np.shape(transformed_image)
    centre_x = 200
    centre_y = 200
    background_array = np.zeros((1000, 1000))
    x_start = centre_x + x_shift
    x_stop = x_start + mask_width

    y_start = centre_y + y_shift
    y_stop = y_start + mask_height

    background_array[y_start:y_stop, x_start:x_stop] = transformed_image

    # Take Chunk
    transformed_image = background_array[centre_y:centre_y + image_height, centre_x:centre_x + image_width]

    # Rebinarize
    transformed_image = np.where(transformed_image > 0.5, 1, 0)

    return transformed_image


def transform_mask_or_atlas(image, variable_dictionary):

    image_height = 600
    image_width = 608

    # Unpack Dictionary
    angle = variable_dictionary['rotation']
    x_shift = variable_dictionary['x_shift']
    y_shift = variable_dictionary['y_shift']
    x_scale = variable_dictionary['x_scale']
    y_scale = variable_dictionary['y_scale']

    # Copy
    transformed_image = np.copy(image)

    # Scale
    original_height, original_width = np.shape(transformed_image)
    new_height = int(original_height * y_scale)
    new_width = int(original_width * x_scale)
    transformed_image = resize(transformed_image, (new_height, new_width), preserve_range=True)

    # Rotate
    transformed_image = ndimage.rotate(transformed_image, angle, reshape=False, prefilter=True)

    # Insert Into Background
    mask_height, mask_width = np.shape(transformed_image)
    centre_x = 200
    centre_y = 200
    background_array = np.zeros((1000, 1000))
    x_start = centre_x + x_shift
    x_stop = x_start + mask_width

    y_start = centre_y + y_shift
    y_stop = y_start + mask_height

    background_array[y_start:y_stop, x_start:x_stop] = transformed_image

    # Take Chunk
    transformed_image = background_array[centre_y:centre_y + image_height, centre_x:centre_x + image_width]

    # Rebinarize
    transformed_image = np.where(transformed_image > 0.5, 1, 0)

    return transformed_image


def get_activity_tensor(base_directory, onset_file, start_window, stop_window, save_directory_root, start_cutoff=3000):

    # Load Activity Matrix
    delta_f_file = os.path.join(base_directory, "Delta_F.h5")
    delta_f_container = tables.open_file(delta_f_file, "r")
    activity_matrix = delta_f_container.root.Data
    number_of_timepoints, number_of_pixels = np.shape(activity_matrix)

    # Open Onset File
    onset_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", onset_file))
    number_of_trials = len(onset_list)

    # Create Activity Tensor
    activity_tensor = []
    for trial_index in range(number_of_trials):

        # Get Trial Activity
        trial_start = onset_list[trial_index] + start_window
        trial_stop = onset_list[trial_index] + stop_window
        print("Start", trial_start,  "stop", trial_stop)


        if trial_start > start_cutoff and trial_stop < number_of_timepoints:
            trial_activity = activity_matrix[trial_start:trial_stop]
            activity_tensor.append(trial_activity)

    activity_tensor = np.array(activity_tensor)

    # Save Activity Tensor
    output_directory = check_save_directory(base_directory, save_directory_root)
    save_file = os.path.join(output_directory, onset_file.replace("_onsets.npy", "") + "_" + "Activity_Tensor.npy")
    np.save(save_file, activity_tensor)

    # Close Delta F File
    delta_f_container.close()

    return activity_tensor


def load_tight_mask():

    #package_file_directory = get_package_file_directory()
    #tight_mask_file = os.path.join(package_file_directory, "Tight_Mask_Dict.npy")

    tight_mask_file = pkg_resources.resource_stream('Files', 'Tight_Mask_Dict.npy')

    tight_mask_dict = np.load(tight_mask_file, allow_pickle=True)[()]
    indicies = tight_mask_dict["indicies"]
    image_height = tight_mask_dict["image_height"]
    image_width = tight_mask_dict["image_width"]
    return indicies, image_height, image_width


def load_tight_mask_downsized():

    mask = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Mask_Array.npy")
    mask_alignment_dictionary = np.load(r"/media/matthew/Expansion/Widefield_Analysis/Consensus_Clustering/Tight_Mask_Alignment_Dictionary.npy", allow_pickle=True)[()]

    # Transform Mask
    mask = transform_mask_or_atlas(mask, mask_alignment_dictionary)
    mask = resize(mask, (100, 100), preserve_range=True, order=0, anti_aliasing=False)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask>0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width



def get_ai_filename(base_directory):

    #Get List of all files
    file_list = os.listdir(base_directory)
    ai_filename = None

    #Get .h5 files
    h5_file_list = []
    for file in file_list:
        if file[-3:] == ".h5":
            h5_file_list.append(file)

    #File the H5 file which is two dates seperated by a dash
    for h5_file in h5_file_list:
        original_filename = h5_file

        #Remove Ending
        h5_file = h5_file[0:-3]

        #Split By Dashes
        h5_file = h5_file.split("-")

        if len(h5_file) == 2 and h5_file[0].isnumeric() and h5_file[1].isnumeric():
            return original_filename

def load_ai_recorder_file(base_directory):

    ai_filename = get_ai_filename(base_directory)
    ai_recorder_file_location = os.path.join(base_directory, ai_filename)

    table = tables.open_file(ai_recorder_file_location, mode='r')
    data = table.root.Data

    number_of_seconds = np.shape(data)[0]
    number_of_channels = np.shape(data)[1]
    sampling_rate = np.shape(data)[2]

    data_matrix = np.zeros((number_of_channels, number_of_seconds * sampling_rate))

    for second in range(number_of_seconds):
        data_window = data[second]
        start_point = second * sampling_rate

        for channel in range(number_of_channels):
            data_matrix[channel, start_point:start_point + sampling_rate] = data_window[channel]

    data_matrix = np.clip(data_matrix, a_min=0, a_max=None)
    return data_matrix


def load_analysis_container(analysis_name):

    # This Is The Location of My Analysis Logbook
    logbook_file_location = r"/home/matthew/Documents/Github_Code/Workflows/Analysis_Containers.ods"

    #  Read Logbook As A Dataframe
    logbook_dataframe = pd.read_excel(logbook_file_location, engine="odf")

    # Return First Analysis Container With This Name
    selected_analysis = logbook_dataframe.loc[(logbook_dataframe["Analysis_Name"] == analysis_name)].values[0]

    # Unpack Container
    start_window = selected_analysis[1]
    stop_window = selected_analysis[2]
    onset_files = selected_analysis[3].replace(' ', '')
    tensor_names = selected_analysis[4].replace(' ', '')
    behaviour_traces = selected_analysis[5] #.replace(' ', '')
    difference_conditions = selected_analysis[6].replace(' ', '')

    onset_files = onset_files.split(',')
    tensor_names = tensor_names.split(',')
    behaviour_traces = behaviour_traces.split(',')
    difference_conditions = difference_conditions.split(',')

    if difference_conditions[0] == 'None':
        difference_conditions = None

    else:
        for x in range(2):
            difference_conditions[x] = int(difference_conditions[x])

    # Return Container
    return [start_window, stop_window, onset_files, tensor_names, behaviour_traces, difference_conditions]


def check_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
        
def load_downsampled_mask(base_directory):

    mask = np.load(os.path.join(base_directory, "Generous_Mask.npy"))

    # Transform Mask
    mask = resize(mask, (300, 304), preserve_range=True, order=0, anti_aliasing=True)

    image_height = np.shape(mask)[0]
    image_width = np.shape(mask)[1]

    mask = np.where(mask > 0.1, 1, 0)
    mask = mask.astype(int)
    flat_mask = np.ndarray.flatten(mask)
    indicies = np.argwhere(flat_mask)
    indicies = np.ndarray.astype(indicies, int)
    indicies = np.ndarray.flatten(indicies)

    return indicies, image_height, image_width

def get_delta_f_sample_from_svd_unprocessed(base_directory, sample_start, sample_end):

    # Extract Raw Delta F
    corrected_svt = np.load(os.path.join(base_directory, "Corrected_SVT.npy"))
    print("SVT Shape", np.shape(corrected_svt))
    u = np.load(os.path.join(base_directory, "U.npy"))
    print("U Shape", np.shape(u))
    delta_f_sample = np.dot(u, corrected_svt[:, sample_start:sample_end])
    print("Delta F Sample", np.shape(delta_f_sample))

    return delta_f_sample



def get_delta_f_sample_from_svd(base_directory, sample_start, sample_end, window_size=3):

    # Extract Raw Delta F
    corrected_svt = np.load(os.path.join(base_directory, "Corrected_SVT.npy"))
    print("SVT Shape", np.shape(corrected_svt))
    u = np.load(os.path.join(base_directory, "U.npy"))
    print("U Shape", np.shape(u))
    delta_f_sample = np.dot(u, corrected_svt[:, sample_start:sample_end])
    print("Delta F Sample", np.shape(delta_f_sample))
    delta_f_sample = np.moveaxis(delta_f_sample, 2, 0)
    print("Delta F Sample", np.shape(delta_f_sample))

    # Load Mask
    indicies, image_height, image_width = load_generous_mask(base_directory)

    # Reconstruct Data
    reconstructed_delta_f = []
    #plt.ion()
    #colourmap = widefield_utils.get_musall_cmap()
    number_of_frames = (sample_end - sample_start) + window_size
    for frame_index in range(number_of_frames):
        frame_data = delta_f_sample[frame_index:frame_index + window_size]
        frame_data = np.mean(frame_data, axis=0)
        template = frame_data
        #template = np.zeros(image_height * image_width)
        #template[indicies] = frame_data
        #template = np.reshape(template, (image_height, image_width))
        template = ndimage.gaussian_filter(template, sigma=1)

        reconstructed_delta_f.append(template)

        #plt.imshow(template, vmin=-0.05, vmax=0.05, cmap=colourmap)
        #plt.draw()
        #plt.pause(0.1)
        #plt.clf()

    reconstructed_delta_f = np.array(reconstructed_delta_f)

    return reconstructed_delta_f

def get_delta_f_sample(base_directory, sample_start, sample_end, window_size=3):

    # Extract Raw Delta F
    delta_f_file = os.path.join(base_directory, "Downsampled_Delta_F.h5")
    delta_f_file_container = tables.open_file(delta_f_file, mode="r")
    delta_f_matrix  = delta_f_file_container.root["Data"]
    delta_f_sample = delta_f_matrix[sample_start-window_size:sample_end]
    delta_f_sample = np.nan_to_num(delta_f_sample)

    # Denoise with dimensionality reduction
    model = PCA(n_components=150)
    transformed_data = model.fit_transform(delta_f_sample)
    delta_f_sample = model.inverse_transform(transformed_data)

    # Load Mask
    indicies, image_height, image_width = load_downsampled_mask(base_directory)

    # Reconstruct Data
    reconstructed_delta_f = []
    number_of_frames = (sample_end - sample_start) + window_size
    for frame_index in range(number_of_frames):
        frame_data = delta_f_sample[frame_index :frame_index + window_size]
        frame_data = np.mean(frame_data, axis=0)
        template = np.zeros(image_height * image_width)
        template[indicies] = frame_data
        template = np.reshape(template, (image_height, image_width))
        template = ndimage.gaussian_filter(template, sigma=1)

        reconstructed_delta_f.append(template)

    reconstructed_delta_f = np.array(reconstructed_delta_f)

    delta_f_file_container.close()
    return reconstructed_delta_f


def extract_mousecam_data(video_file, frame_list):

    # Open Video File
    cap = cv2.VideoCapture(video_file)

    # Extract Selected Frames
    extracted_data = []
    for frame in frame_list:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
        ret, frame = cap.read()
        frame = frame[:, :, 0]
        extracted_data.append(frame)

    cap.release()
    extracted_data = np.array(extracted_data)

    return extracted_data




def get_mousecam_sample(base_directory, mousecam_filename, sample_start, sample_end):

    # Load Widefield Frame Dict
    widefield_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    print("Widefield Frame Dict", widefield_frame_dict)

    # Get Mousecam Frames
    mousecam_frames = []
    for widefield_frame in range(sample_start, sample_end):
        corresponding_mousecam_frame = widefield_frame_dict[widefield_frame]
        mousecam_frames.append(corresponding_mousecam_frame)

    # Extract Mousecam Data
    mousecam_data = extract_mousecam_data(os.path.join(base_directory, mousecam_filename), mousecam_frames)

    return mousecam_data


def get_signals_from_svd_sample(sample):
    
    n_time_points = np.shape(sample)[2]
    subsample = block_reduce(sample, block_size=(6,6,1), func=np.mean) 
    # compute time varying connectivity: flatten data

    signals = np.reshape(subsample,(np.shape(subsample)[0]*np.shape(subsample)[1],n_time_points))
    # remove zero signals
    power = np.std(signals,axis=1) #standard dev of signal. I'll discard zero std
    signals = signals[power!=0,:]
    return signals

    
def get_walking_series(base_directory,kernel_std,threshold):
    
    ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))
    abs_speed = np.abs(ai_matrix[8,:]-np.mean(ai_matrix[8,:]))
    filtered_speed = scipy.ndimage.gaussian_filter(abs_speed,sigma = kernel_std)
    walking = filtered_speed > threshold
    return walking


def load_registered_sample(base_directory,registered_directory,start,end):
    corrected_svt = np.load(os.path.join(base_directory, "Corrected_SVT.npy"))
    u = np.load(os.path.join(registered_directory, "Registered_U.npy"))
    delta_f_sample = np.dot(u, corrected_svt[:, start:end])
    return delta_f_sample