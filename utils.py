import os
import numpy as np
from scipy.signal import argrelmax
from image_filtering import blur_image


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


def detect_maxima_in_hist_distribution(input_ima_ge, target_maxima_position, initial_max_order=10, final_max_order=3, hist_bins=100):
    #Copy input image
    input_ima_ge_copy = input_ima_ge.copy()
        
    #Calculate the histogram distribution of the intensities of the input image - separate the counts per histogram bin, and the edges of the bins on the histogram
    img_hist_counts, img_hist_edges = np.histogram(input_ima_ge_copy.flatten(), bins=hist_bins)
        
    #Detect less than final_max_number maxima in the histogram distribution
    maxima_starting_order = initial_max_order #Initialize a value to be used for initial maxima detection
    max_of_hist_counts = argrelmax(img_hist_counts, order=maxima_starting_order) #Initialize maxima detection
    while len(max_of_hist_counts[0])<final_max_order: #Iterate detection until less than final_max_number maxima are detected, by progressively decreasing the order of maxima detection
        maxima_starting_order = maxima_starting_order-1
        max_of_hist_counts = argrelmax(img_hist_counts, order=maxima_starting_order)
        
    #Get the maxima in the targeted maxima position
    position_of_target_max = max_of_hist_counts[0][target_maxima_position]
        
    #get the intensity value which corresponds to maxima in the targeted maxima position
    target_max_intensity_val = img_hist_edges[position_of_target_max]

    return target_max_intensity_val


def get_minima_in_hist_distribution(input_imga_ge, smooth_img=False, n=None, ny=None):
    if smooth_img:
        assert n != None, "indicate the size of the gaussian kernel (n) to use for smoothing"

    #Copy the input image
    input_imga_ge_copy = input_imga_ge.copy()

    #Smooth input image if selected
    if smooth_img:
        gau_input_imga_ge = blur_image(input_imga_ge_copy, n=None, ny=None)
    else:
        gau_input_imga_ge = input_imga_ge_copy

    return
        # #Get the histogram distribution of the flatten image
        # gauss_tp_hist_counts, gauss_tp_hist_edges = np.histogram(gauss_tp_fl, bins=int_hist_bins)
        
        # #Get the position of the mode value of the histogram distribution in the guassian smoothed image (the position of the intensity value with minimum counts)
        # tp_mode_position = np.argmax(gauss_tp_hist_counts)
    
        # #Sub-slice the histogram values to consider only the part after the mode
        # gauss_tp_hist_counts_overmode = gauss_tp_hist_counts[tp_mode_position:]
    
        # #Initialize an order counter
        # order_count__min = 5
    
        # #Get the position/index of the minima values in the sliced (over the mode) histogram distribution of the gaussian smoothed image (the position of the intensity values with minimum counts)
        # min_val_hist_scipy_i = scipy.signal.argrelmin(gauss_tp_hist_counts_overmode, order=order_count__min)
        
        # #while loop, decreasing the order counter until at least a minimum value is found
        # while min_val_hist_scipy_i[0].shape[0]<1:
        #     order_count__min = order_count__min-1
        #     min_val_hist_scipy_i = scipy.signal.argrelmin(gauss_tp_hist_counts_overmode, order=order_count__min)
        
        # # Sum the mode position to the position/index of the first (from the x axis origin) minima value in the sliced (over the mode) histogram distribution
        # min_val_hist_scipy = tp_mode_position + min_val_hist_scipy_i[0][0]

        # #Initialize an order counter
        # order_count__max = 5
    
        # #Get the position/index of the maxima values in the sliced (over the mode) histogram distribution of the gaussian smoothed image (the position of the intensity values with maximum counts)
        # max_val_hist_scipy_i = scipy.signal.argrelmax(gauss_tp_hist_counts_overmode, order=order_count__max)
    
        # #while loop, decreasing the order counter until at least a maxima value is found
        # while max_val_hist_scipy_i[0].shape[0]<1:
        #     order_count__max = order_count__max-1
        #     max_val_hist_scipy_i = scipy.signal.argrelmax(gauss_tp_hist_counts_overmode, order=order_count__max)
        
        # # Sum the mode position to the position/index of the first (from the x axis origin) maxima value in the sliced (over the mode) histogram distribution
        # max_val_hist_scipy = tp_mode_position + max_val_hist_scipy_i[0][0]
    
        # #Get the position/index of the minimum value in the histogram distribution of the gaussian smoothed image, when considering only the values in between the mode and the first (from x axis origin) maxima after the mode. Then sum to it the position of the mode
        # min_val_hist_minbwnmaxs = tp_mode_position + np.argmin(gauss_tp_hist_counts[tp_mode_position:max_val_hist_scipy])
    
        # #Get the intensity value of the first histogram minima (min_val_hist_scipy has all the positions of the minima of the histogram distribution. We take the first from the left - from the x axis origin - and check which intensity value is in that position)
        # putative_threshold_min_scipy = gauss_tp_hist_edges[min_val_hist_scipy]
        
        # #Get the intensity value of the histogram-minimum-in-between-the-first-two-histogram-maxima (hmbfthm)
        # putative_threshold_min_minbwnmaxs = gauss_tp_hist_edges[min_val_hist_minbwnmaxs]
    
        # #Get mode intensity value
        # putative_mode_threshold = gauss_tp_hist_edges[tp_mode_position]
    
        # # #Get the intensity value of the first (from x axis origin) sliced (over the mode) histogram maxima
        # putative_max_val = gauss_tp_hist_edges[max_val_hist_scipy]


