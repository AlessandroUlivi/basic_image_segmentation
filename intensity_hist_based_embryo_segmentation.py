import numpy as np
from utils import get_minima_after_mode_in_hist_distribution
from image_filtering import blur_image, highpass_area_filter
from image_segmentation import get_highpass_based_segmentaion_img, get_highpass_based_segmentaion_img


def mask_embryo(timecourse_file, area_threshold4embryo, sigma_gaussian_smt=5, int_hist_bins=100, roi_mask=None, time_axis=0,
                output_lowval=0, output_highval=255, output_dtype=np.uint8):
    """
    Function only supports 3D files
    """
    
    #Copy timecourse file
    timecourse_file_copy = timecourse_file.copy()

    #Form a list of timepoints from the timecourse_file
    if time_axis==0:
        timepoints_list = [timecourse_file_copy[tp,...] for tp in range(timecourse_file_copy.shape[0])]
    elif time_axis==1:
        timepoints_list = [timecourse_file_copy[:,tp1,:] for tp1 in range(timecourse_file_copy.shape[1])]
    elif time_axis==2:
        timepoints_list = [timecourse_file_copy[...,tp2] for tp2 in range(timecourse_file_copy.shape[2])]

    #Initialize:
    # a dictionary to map each timepoint to different intensity values - tp_thre_dict -
    # a list to collect all thresholds calculated - tp_min_thresh_collection -
    # a list to collect all the gaussian smoothed timepoints - gaus_tp_collection -
    tp_val_dict = {}
    tp_min_val_collection = []
    gaus_tp_collection = []

    #Iterate through the timepoints
    for tpc, timepoint in enumerate(timepoints_list):
        # print("="*20, tpc)
        # print(tp.shape)

        #Gaussian smooth the timepoint
        gau_timepoint = blur_image(timepoint, n=sigma_gaussian_smt)

        #Add the gaussian smoothed timepoint to relative collection list
        gaus_tp_collection.append(gau_timepoint)
        # print(gau_timepoint.shape)

        #Get, for the timepoint, the following intensity values:
        #- The intensity value corresponding to the first minima of the histogram distribution of intensity values, when considering only the part of the histrogram after the mode.
        #- The intensity value corresponding to the minima of the histogram distribution of intensity values, when considering the part of the histrogram in between the mode and
        #the first maxima after the mode.
        #- The intensity value corresponding to the mode of the the histogram distribution of intensity values.
        #- The intensity value corresponding to the first maxima of the histogram distribution of intensity values, when considering only the part of the histogram after the mode.
        timepoint_minima = get_minima_after_mode_in_hist_distribution(gau_timepoint,
                                                                        roi__mask=roi_mask,
                                                                        smooth_img=False,
                                                                        bins_of_hist=int_hist_bins,
                                                                        i_order_count__min=5,
                                                                        i_order_count__max=5,
                                                                        return_mode_max=True)
        
        over_mode_first_min = timepoint_minima[0]
        first_min_between_max = timepoint_minima[1]
        hist_mode_val = timepoint_minima[2]
        hist_max_val = timepoint_minima[3]
        
        #link timepoint to the calculated intensity values
        tp_val_dict[tpc] = timepoint_minima

    #     #Add:
        #- The intensity value corresponding to the first minima of the histogram distribution of intensity values, when considering only the part of the histrogram after the mode.
        #- The intensity value corresponding to the minima of the histogram distribution of intensity values, when considering the part of the histrogram in between the mode and
        #the first maxima after the mode.
        #tp_min_val_collection
        tp_min_val_collection.append(over_mode_first_min)
        tp_min_val_collection.append(first_min_between_max)

    # print(tp_min_val_collection)
    
    #Calculate the median and standard deviation of all values the following values (collected in tp_min_val_collection):
    #- The intensity value corresponding to the first minima of the histogram distribution of intensity values, when considering only the part of the histrogram after the mode.
    #- The intensity value corresponding to the minima of the histogram distribution of intensity values, when considering the part of the histrogram in between the mode and
    #the first maxima after the mode.
    median_min_values, std_dev_min_values = np.median(tp_min_val_collection), np.std(tp_min_val_collection)
    # print(median_min_values, std_dev_min_values)

    # #Initialize an output array
    output_array = np.zeros(timecourse_file_copy.shape)
    # print(output_array.shape)
    #Iterate through the timepoints in the dictionary linking timepoints to their calculated thresholds - note: use range() for iteration,
    # to make sure timepoints will be samples in ascending order
    for tp1 in range(len(tp_val_dict)):
        # print("==="*3, tp1)
        #Collect the intensity value corresponding to the first minima of the histogram distribution of intensity values, when considering only the part of
        # the histrogram after the mode.
        put_threshold_first_min = tp_val_dict[tp1][0]
    
        #Collect the intensity value corresponding to the minima of the histogram distribution of intensity values, when considering the part of the histrogram in between
        # the mode and the first maxima after the mode.
        put_threshold_minbwnmaxs = tp_val_dict[tp1][1]
        # print(put_threshold_first_min, put_threshold_minbwnmaxs)
        #First option for highpass threshold of the embryo in the timepoint:
        # if put_threshold_first_min is not further than half a standard deviation from the median of minima values - use it to threshold the timepoint image
        if ((put_threshold_first_min>(median_min_values-(std_dev_min_values/2))) and (put_threshold_first_min<(median_min_values+(std_dev_min_values/2)))):
            threshold_to_use = put_threshold_first_min
            # print("SCIPY")
    
        #Second option for highpass threshold of the embryo in the timepoint:
        #if the threshold value based on histogram-minimum-in-between-the-first-two-histogram-maxima-position is not further than half a standard deviation
        # median of minima values - use it to threshold the timepoint image
        elif ((put_threshold_minbwnmaxs>(median_min_values-(std_dev_min_values/2))) and (put_threshold_minbwnmaxs<(median_min_values+(std_dev_min_values/2)))):
            threshold_to_use = put_threshold_minbwnmaxs
            # print("MBTM")
    
        #As a last option: simply use the median of minima values to threshold the timepoint image
        else:
            threshold_to_use = median_min_values
            # print("MEDIAN")
        # print(threshold_to_use)
        #Recollect the gaussian smoothed image from the collection list rather than re-doing the smoothing
        guass_tp_img = gaus_tp_collection[tp1]
        # print(guass_tp_img.shape)
        #Threshold the image
        thresholded_guass_tp_img = get_highpass_based_segmentaion_img(guass_tp_img,
                                                                        threshold_to_use,
                                                                        roi__ma_s_k=roi_mask)
        # print(thresholded_guass_tp_img.shape)
        # print(thresholded_guass_tp_img.dtype)
        # print(np.unique(thresholded_guass_tp_img))
        #Filter the thresholded image for areas which are bigger than area_threshold 4embryo
        areafilt_thresholded_guass_tp_img = highpass_area_filter(thresholded_guass_tp_img,area_threshold4embryo)
        # print(areafilt_thresholded_guass_tp_img.shape)
        # print(areafilt_thresholded_guass_tp_img.dtype)
        # print(np.unique(areafilt_thresholded_guass_tp_img))
        #Modify the output array
        if time_axis==0:
            output_array[tp1,...] += areafilt_thresholded_guass_tp_img
        elif time_axis==1:
            output_array[:,tp1,:] += areafilt_thresholded_guass_tp_img
        elif time_axis==2:
            output_array[...,tp1] += areafilt_thresholded_guass_tp_img
    # print(output_array.shape)
    # print(output_array.dtype)
    # print(np.unique(output_array))

    #Rescale the output array to be in the desired output range
    rescaled_output_array = np.where(output_array>0, output_highval, output_lowval).astype(output_dtype)

    return rescaled_output_array


