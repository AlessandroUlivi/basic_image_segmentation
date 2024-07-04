import os
import numpy as np
from scipy.signal import argrelmax, argrelmin
from skimage.draw import polygon
from image_filtering import blur_image
from roifile import ImagejRoi


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
    """
    given an input image (input_ima_ge) the function does the following:
    - first, it detectes a certain number of the maxima of the histogram distribution of intensities values. This certain number is => to final_max_order (input value, default 3)
      and <= to initial_max_order (input value, default 10). These maxima are ordered according to their position in the histogram distribution, such that the position in the
      list correspond to the position in the histogram distribution, from left to right.
    - second, given a target position (input value target_maxima_position), the function returns the intensity value correponding to the maxima in the target position of the list
      of detected maxima.
    """
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


def get_minima_after_mode_in_hist_distribution(input_ima_ge, roi__mask=None, smooth_img=True, n=5, ny=None, bins_of_hist=100, i_order_count__min=5, i_order_count__max=5, return_mode_max=False):
    """
    Given an input image (input_ima_ge), it returns the following minima of the histogram distributions of the intensity values:
    - output position 0, the first minima after the mode of value. First is considered from the x axis origin.
    - output position 1, the mimima value in the range of values between the mode and the first maxima value after the mode.

    Note that by default the input image is smoothed using a gaussian kernel of size 5 pixels before calculating the histogram distribution of intensity values.
    
    If return_mode_max parameter (bool, True/False, default False) is set to True, the following values are alse returned:
    - oputput position 2, the mode value of the histogram distribution of intensity values.
    - output position 3, the first maxima value of the histogram distribution of intensity values, calculated by taking into account only value after the mode value.

    If an roi__mask is present, the whole analysis is restricted to this region (positive values in the mask are assumed to be the pixels of interest).
    """

    if smooth_img:
        assert n != None, "indicate the size of the gaussian kernel (n) to use for smoothing"

    #Copy the input image
    input_ima_ge_copy = input_ima_ge.copy()

    #Smooth input image if selected
    if smooth_img:
        gau_input_ima_ge = blur_image(input_ima_ge_copy, n=n, ny=ny)
    else:
        gau_input_ima_ge = input_ima_ge_copy

    #Restrict the analysis to an roi if a mask is provided
    if hasattr(roi__mask, "__len__"):
        roi_gau_input_ima_ge = gau_input_ima_ge[roi__mask>0]
    else:
        roi_gau_input_ima_ge = gau_input_ima_ge.flatten()

    #Get the histogram distribution of the flatten image
    img_hist_counts, img_hist_edges = np.histogram(roi_gau_input_ima_ge, bins=bins_of_hist)

    #Get the position of the mode value of the histogram distribution (the position of the intensity value with maximum counts)
    img_mode_position = np.argmax(img_hist_counts)

    #Sub-slice the histogram values to consider only the part after the mode
    img_hist_counts_overmode = img_hist_counts[img_mode_position:]
    
    #Initialize an order counter
    order_count__min = i_order_count__min

    #Get the position/index of the minima values in the sliced (over the mode) histogram distribution (the position of the intensity values with minimum counts)
    min_val_hist_posistion_i = argrelmin(img_hist_counts_overmode, order=order_count__min)
        
    #while loop, decreasing the order counter until at least a minimum value is found
    while min_val_hist_posistion_i[0].shape[0]<1:
        order_count__min = order_count__min-1
        min_val_hist_posistion_i = argrelmin(img_hist_counts_overmode, order=order_count__min)

    # Sum the mode position to the position/index of the first (from the x axis origin) minima value in the sliced (over the mode) histogram distribution
    min_val_hist_posistion = img_mode_position + min_val_hist_posistion_i[0][0]

    # #Initialize an order counter
    order_count__max = i_order_count__max

    #Get the position/index of the maxima values in the sliced (over the mode) histogram distribution of the gaussian smoothed image (the position of the intensity values with maximum counts)
    max_val_hist_posistion_i = argrelmax(img_hist_counts_overmode, order=order_count__max)

    #while loop, decreasing the order counter until at least a maxima value is found
    while max_val_hist_posistion_i[0].shape[0]<1:
        order_count__max = order_count__max-1
        max_val_hist_posistion_i = argrelmax(img_hist_counts_overmode, order=order_count__max)

    # Sum the mode position to the position/index of the first (from the x axis origin) maxima value in the sliced (over the mode) histogram distribution
    max_val_hist_posistion = img_mode_position + max_val_hist_posistion_i[0][0]

    #Get the position/index of the minimum value in the histogram distribution of the image, when considering only the values in between the mode and the first (from x axis origin) maxima after the mode. Then sum to it the position of the mode
    min_val_hist_posistion_minbwnmaxs = img_mode_position + np.argmin(img_hist_counts[img_mode_position:max_val_hist_posistion])

    #Get the intensity value of the first histogram minima (from the left - from the x axis origin -) in the sliced (over the mode) histogram distribution
    min_intensity_val_overmode = img_hist_edges[min_val_hist_posistion]

    #Get the intensity value of the histogram minima, , when considering only the values in between the mode and the first (from x axis origin) maxima after the mode
    min_intensity_val_minbwnmaxs = img_hist_edges[min_val_hist_posistion_minbwnmaxs]

    #For plotting purposes it could be convenient to visualize the mode and the first maxima after the mode which was detected. Return these values if the option is selected
    #Else only return the minima values
    if return_mode_max:

        #Get the value corresponding to the mode of the histogram of intensity distribution
        mode_intensity_val = img_hist_edges[img_mode_position]

        #Get the value corresponding to the first maxima of the histogram of intensity distribution when considering only the part after the mode
        max_overmode_intensity_val = img_hist_edges[max_val_hist_posistion]

        return min_intensity_val_overmode, min_intensity_val_minbwnmaxs, mode_intensity_val, max_overmode_intensity_val
    else:
        return min_intensity_val_overmode, min_intensity_val_minbwnmaxs


def join_masks(image_mask_1, image_mask_2, low_binary_val=0, high_binary_val=255, output_dtype=np.uint8):
    """
    Returns a union of 2 input binary masks. Each input mask is interpreted as: positive pixels are pixels of interest.
    The default binary output has values 0 or 255 and it is dtype uint8.
    """
    #Copy input masks
    image_mask_1_copy = image_mask_1.copy()
    image_mask_2_copy = image_mask_2.copy()

    #sum the two binary masks
    summed_mask = image_mask_1_copy + image_mask_2_copy

    #binarize the summed masks, set the range of values as 0 or 255 and the dtype as uint8
    unit8_binarized_combined_img = np.where(summed_mask>0, high_binary_val, low_binary_val).astype(output_dtype)

    return unit8_binarized_combined_img


def subtract_arrays(array_1_to_sub_from, array_2_to_be_subtract, threshold=0, low_binary_val=0, high_binary_val=255, output_dtype=np.uint8):
    """
    Returns the subtraction of array_2_to_be_subtract from array_1_to_sub_from. The output of the subtraction is then binarized. By default the binarizations sets
    values higher than 0 to 255 and values equal or lower than 0 to 0.
    The default binary output has values 0 or 255 and it is dtype uint8.
    """

    #Copy input arrays
    array_1_copy = array_1_to_sub_from.copy()
    array_2_copy = array_2_to_be_subtract.copy()

    #Subtract the arrays
    array_1_minus_array_2 = array_1_copy - array_2_copy

    #Rescale in the wanted range
    rescaled_subtraction = np.where(array_1_minus_array_2>threshold, high_binary_val, low_binary_val).astype(output_dtype)

    return rescaled_subtraction


def form_mask_from_roi(roi_file_path, reference_img, return_coordinates=False, roi_pixel_value=255, background_pixel_val=0, output_dtype=np.uint8):
    """
    Given:
    - The directory of a .roi or .zip file containing one or multiple ROIs saved from Fiji/ImageJ.
    - A reference image to be used for shape.

    The function returns a binary array of the same shape as reference, where pixels of the ROIs are assinged value roi_pixel_value (default 255) and the rest of the pixels are
    assigned value background_pixel_val (default 0). The default output dtype is uint8.

    If return_coordinates=True (default is False) the function returns the coordinates of the .roi/.zip file. If the file is a signle roi (extension .roi) the output is a numpy array
    of sub-numpy-arrays. Each sub-numpy-array is the col_x, row_y coordinates of a pixel of the roi. If the roi file contains multiple rois (extension .zip) the output is a list where
    each element is a numpy array of sub-numpy-arrays. Each numpy-array corresponds to an individual roi. Each sub-numpy-array of each numpy array is the col_x, row_y coordinates of a
    pixel of the individual roi.

    NOTES: the function is tested for Imagej/Fiji -generated roi files (extensions .roi or .zip). The function was only tested on 2D images.
    """

    #Open roi file
    roi_file = ImagejRoi.fromfile(roi_file_path)

    #Initialize the output array as a zero array of the same shape of reference_img
    out_img = np.zeros((reference_img.shape[0], reference_img.shape[1])).astype(np.uint8)

    #Iterate through the coordinates of the roi file and collect col_x and row_y pixels coordinates in separate lists.
    # Note: if file is a signle roi (extension .roi) roi_file_copy is a list of tuples. Each tuple is the col_x, row_y coordinates of a pixel of the roi.
    # If the roi file contains multiple rois (extension .zip) the file is a list where each element is a sub-list of tuples. Each sub-list corresponds to an individual roi.
    # Each tuple of each sub-list is the col_x, row_y coordinates of a pixel of the individual roi.

    #If the input file is a collection of rois
    if roi_file_path[-4:]==".zip":

        #Initialize an outputlist to collect coordinates of different rois, if return_coordinates is set to True
        if return_coordinates:
            output_coords_coll_list = []

        #Iterate through the individual sub-roi
        for sub_roi in roi_file:

            #Initialize collection lists for col_x and row_y coordinates
            col_x_list = []
            row_y_list = []

            #Get roi_file_coordinates
            sub_roi_file_coords = sub_roi.coordinates()

            #Add coordinates to output collection list if return_coordinates is set to True
            if return_coordinates:
                output_coords_coll_list.append(sub_roi_file_coords)

            #Iterate through the coordinates of the individual sub-roi
            for c in sub_roi_file_coords:
                col_x_list.append(c[0]-1) #Note: because roi have been generate in ImageJ their coordinate numeration starts from 0, while python starts from 0
                row_y_list.append(c[1]-1) #Note: because roi have been generate in ImageJ their coordinate numeration starts from 0, while python starts from 0

            #Transform the coordinates collection lists in numpy arrays
            col_x_array = np.asarray(col_x_list)
            row_y_array = np.asarray(row_y_list)

            #Use the array coordinates to form a polygon
            yy_rr, xx_cc = polygon(row_y_array, col_x_array)

            #Modify the output array
            out_img[yy_rr, xx_cc]=255

    #If the input file is a single rois
    else:
        #Initialize collection lists for col_x and row_y coordinates
        col_x_list_1 = []
        row_y_list_1 = []
        
        #Get roi_file_coordinates
        roi_file_coords = roi_file.coordinates()
        
        #Iterate through the coordinates of the roi
        for c1 in roi_file_coords:
            col_x_list_1.append(c1[0]-1) #Note: because roi have been generate in ImageJ their coordinate numeration starts from 0, while python starts from 0
            row_y_list_1.append(c1[1]-1) #Note: because roi have been generate in ImageJ their coordinate numeration starts from 0, while python starts from 0
    
        #Transform the coordinates collection lists in numpy arrays
        col_x_array_1 = np.asarray(col_x_list_1)
        row_y_array_1 = np.asarray(row_y_list_1)

        #Use the array coordinates to form a polygon
        yy_rr_1, xx_cc_1 = polygon(row_y_array_1, col_x_array_1)

        #Modify the output array
        out_img[yy_rr_1, xx_cc_1]=255

    #Rescale the output array in the desired range
    rescaled_out_img = np.where(out_img>0, roi_pixel_value, background_pixel_val).astype(output_dtype)

    if return_coordinates:
        if roi_file_path[-4:]==".zip":
            return rescaled_out_img, output_coords_coll_list
        else:
            return rescaled_out_img, roi_file_coords
    else:
        return rescaled_out_img

