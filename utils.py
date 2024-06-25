import os
import numpy as np
from scipy.signal import argrelmax


def listdirNHF(input_directory):
    """
    creates a list of files in the input directory, avoiding hidden files. Hidden files are identified by the fact that they start with a .
    input: directory of the folder whose elements have to be listed.
    output: list of elements in the input-folder, except hidden files. 
    """
    return [f for f in os.listdir(input_directory) if not f.startswith(".")]


def get_intensity_values_from_histogram(image4hist, perc_v=None, multiplication_f=None, hist_bins=100):
    """
    Given and input image (image4hist), returns, order:
    - position 0; mean intensity value
    - position 1; median intensity value
    - position 2; mode of the histogram of the intensity distribution (the number of bins to calculate the histogram distribution can be specified in the hist_bins, default is 100)
    - position 3; the standard deviation of the histogram of intensities distribution.
    - position 4; per each perc_v provided as input, it returns intensity value of the corresponding histogram of intensities distribution percentile. If perc_v is a single value, a single value is returned, if it is a list, a list is returned.
    - position 5; per each multiplication_f provided it returns an intensity value of the histogram of intensities distribution calculated as output_value=mode+(standard_deviation*multiplication_f). if multiplication_f is single value, a single value is returned, if it is a list, a list is returned

    perc_v -> single value or list
    multiplication_f -> single value or list

    """
    #Initialize an output_list
    output_list = []

    #Get the mean intensity and append it to the output list
    mean_intensity = np.mean(image4hist)
    output_list.append(mean_intensity)

    #Get the median intensity and append it to the output list
    median_intensity = np.median(image4hist)
    output_list.append(median_intensity)

    #Get picture histogram distribution
    image4hist_hist_counts, image4hist__hist_edges = np.histogram(image4hist, bins=hist_bins)

    #Get the histogram mode
    image4hist_hist_mode_pos = np.argmax(image4hist_hist_counts)
    image4hist_hist_mode_val = image4hist__hist_edges[image4hist_hist_mode_pos]

    #Append mode value to output list
    output_list.append(image4hist_hist_mode_val)

    #Get standard deviation of the intensity values histogram distribution
    histogram_ddistr_std_dv = np.std(image4hist)

    #Append standard deviation value to the output list
    output_list.append(histogram_ddistr_std_dv)

    #Get the itensity values corresponding to each percentil indicated in perc_v
    if perc_v != None:
        #If perc_v is a single value, calculate the corresponding intensity value and add it to the output list
        if not isinstance(perc_v, list):
            intensity_percentil = np.percentile(image4hist, perc_v)
            output_list.append(intensity_percentil)
        #If perc_v is a list, caluclate the intensity value corresponding to each indicated percentil
        else:
            intensity_percentils_list = []
            for pv in perc_v:
                intensity_percentil_1 = np.percentile(image4hist, pv)
                intensity_percentils_list.append(intensity_percentil_1)
            output_list.append(intensity_percentils_list)

    #Get the intensity values corresponding to multiplication_f
    if multiplication_f != None:
        #If multiplication_f is a single value, calculate the corresponding intensity value and add it to the output list
        if not isinstance(multiplication_f, list):
            intensity_dist_val = image4hist_hist_mode_val+(histogram_ddistr_std_dv*multiplication_f)
            output_list.append(intensity_dist_val)
        #If multiplication_f is a list, calculate the intensity value corresponding to each indicated value
        else:
            intensity_distances_list = []
            for dv in multiplication_f:
                intensity_dist_val_1 = image4hist_hist_mode_val+(histogram_ddistr_std_dv*dv)
                intensity_distances_list.append(intensity_dist_val_1)
            output_list.append(intensity_distances_list)

    return output_list


def set_boarder_to_value(img_2changeborder, boarder_value=0, boarder_size=8):
    """
    sets input image boarders to a specified value.
    The size of what is considered boarder, in pixels, can be indicated with the boarder_size parameter (default is 8).
    The value boarders are set to can be indicate with the boarder_value parameter (default is 0).
    NOTE: the function only supports 2D images
    """
    copy_img_2changeborder = img_2changeborder.copy()
    copy_img_2changeborder[:boarder_size+1,:] = boarder_value
    copy_img_2changeborder[-boarder_size-1:,:] = boarder_value
    copy_img_2changeborder[:,:boarder_size+1] = boarder_value
    copy_img_2changeborder[:,-boarder_size-1:] = boarder_value
    return copy_img_2changeborder


def detect_maxima_in_hist_distribution(input_ima_ge, target_maxima_position, hist_bins=100, initial_max_order=10, final_max_order=3):
    #Copy input image
    input_ima_ge_copy = input_ima_ge.copy()
        
    #Calculate the histogram distribution of the intensities of the input image - separate the counts per histogram bin, and the edges of the bins on the histogram
    img_hist_counts, img_hist_edges = np.histogram(input_ima_ge_copy.flatten(), bins=hist_bins)
        
    #Detect at less then final_max_number maxima in the histogram distribution
    maxima_starting_order = initial_max_order #Inialize a value to be used for initial maxima detection
    max_of_hist_counts = argrelmax(img_hist_counts, order=maxima_starting_order) #Initialize maxima detection
    while len(max_of_hist_counts[0])<final_max_order: #Iterate detection until less then final_max_number maxima are detected, by progressively decreasing the order of maxima detection
        maxima_starting_order = maxima_starting_order-1
        max_of_hist_counts = argrelmax(img_hist_counts, order=maxima_starting_order)
        
    #Get the maxima in the targeted maxima position
    position_of_target_max = max_of_hist_counts[0][target_maxima_position]
        
    #get the intensity value which correspond to maxima in the targeted maxima position
    target_max_intensity_val = img_hist_edges[position_of_target_max]

    return target_max_intensity_val


