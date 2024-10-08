import numpy as np
from utils import get_intensity_values_from_histogram, set_boarder_to_value, detect_maxima_in_hist_distribution
from skimage.filters import apply_hysteresis_threshold
from image_filtering import frangi_filter


def get_hysteresis_based_segmentation(input_img, hyst_filt_bot_perc, hyst_filt_top_perc, hyst_filt_bot_stdfactor=None, hyst_filt_top_stdfactor=None, filter_choice_logic='strict',
                                      roi_mask=None, img_4_histogram=None, output_lowval=0, output_highval=255, output_dtype=np.uint8):
    """
    Returns a binary mask with positive pixels selected based on an hystereris-filter. I will call the low and high values inputed to the hysteresis filtering as
    low_hyst and high_hyst in this description.
    
    By default low_hyst and high_hyst are calculated as percentiles of the histogram distribution of intensity values. These percentiles must be inputed by the
    user (hyst_filt_bot_perc, hyst_filt_top_perc). I will call the low_hyst and high_hyst values calculated by this method low_perc_v and high_perc_v in this descrition.
    
    The function also allows to indicate multiplication factors (hyst_filt_bot_stdfactor and hyst_filt_top_stdfactor) to be used for calculating low_hyst and high_hyst.
    It is not required that both hyst_filt_bot_stdfactor and hyst_filt_top_stdfactor are provided. It is possible to only provide one of the two.
    When hyst_filt_bot_stdfactor and hyst_filt_top_stdfactor are provided. The corresponding low_hyst and high_hyst are calculated using
    formula-1: v=mode+(multiplication_factor*standard_deviation)
    Where v is low_hyst or high_hyst, mode is the mode value of the histogram distribution of image intensity values.
    multiplication_factor is the corresponding hyst_filt_bot_stdfactor ore hyst_filt_top_stdfactor. Standard_deviation is the standard deviation of the histogram distribution
    of image intensity values.
    I will call low_hyst and high_hyst calculated by this method low_dist_v and high_dist_v in this descrition.

    If low_dist_v and high_dist_v are calculted, the low_hyst and high_hyst which will ultimately be used for the hysteresis filtering will be chosen either from
    low_perc_v and high_perc_v or from low_dist_v and high_dist_v, depending on the filter_choice_logic which is chosen.

    The default filter_choice_logic is "strict". In this case the highest value between low_perc_v and low_dist_v will be used for low_hyst and
    the highest value between high_perc_v and high_dist_v will be used for high_hyst.

    When filter_choice_logic is "loose" the lowest value between low_perc_v and low_dist_v will be used for low_hyst and
    the lowest value between high_perc_v and high_dist_v will be used for high_hyst.

    Custom filter_choice_logic can be created.

    When a binary mask is passed to roi_mask, the process will be restricted to the roi specified in the mask (positive values are interpreted as pixels of interest). The mask
    must have the same shape of the input image.

    When an image is passed to img_4_histogram, this image, instead of input_img will be used to calculate the histogram as well as the low_hyst and high_hyst values. This allows, for
    example to use a smoothed image for the histogram and low_hyst and high_hyst values calculation, while still applying the segmentation to the raw image.

    Output: binary segmentation of putative puncta structures as numpy array of the same shape of the input image. NOTE: the default output is of type np.uint8 with values 0 and 255.
    """
    #Copy input image
    input_img_copy = input_img.copy()

    #If img_4_histogram is provided, copy it
    if hasattr(img_4_histogram, "__len__"):
        img_4_histogram_copy = img_4_histogram.copy()
    
    #Select pixels of roi_mask, if it is provided. Use img_4_histogram instead of input_img if it is provided
    if hasattr(roi_mask, "__len__"):

        if hasattr(img_4_histogram, "__len__"):
            image_2_process = img_4_histogram_copy[roi_mask>0]
        else:
            image_2_process = input_img_copy[roi_mask>0]
    else:
        full_image_zeroes_array = np.ones(input_img_copy.shape) #Note: I should check if it is necessary to pass a zero array instead of just working with the input image
        if hasattr(img_4_histogram, "__len__"):
            image_2_process = img_4_histogram_copy[full_image_zeroes_array>0]
        else:
            image_2_process = input_img_copy[full_image_zeroes_array>0]
    
    #If no values are provided to calculate hysteresis-based filtering by summing the standard deviantion of histogram intensity to the mode of the distribution
    if hyst_filt_bot_stdfactor == None and hyst_filt_top_stdfactor == None:
        #Get mode, high, low values to apply to hysteresis-based filtering based on percentiles
        histogram_based_intensity_values = get_intensity_values_from_histogram(image_2_process, perc_v=[hyst_filt_bot_perc, hyst_filt_top_perc])
        low_hyst_threshold = histogram_based_intensity_values[4][0]
        high_hyst_threshold = histogram_based_intensity_values[4][1]
        #Apply hysteresis-based filtering
        hysteresis_filt_img = apply_hysteresis_threshold(input_img_copy, low=low_hyst_threshold, high=high_hyst_threshold)

    #If a values is provided to calculate the low value of the hysteresis-based filtering by summing the standard deviantion of histogram intensity to the mode of the distribution
    elif hyst_filt_bot_stdfactor != None and hyst_filt_top_stdfactor == None:
        #Get mode, high, low values to apply to hysteresis-based filtering based on percentiles and based of the mode of the histogram intensity distribution plus the
        #  standard deviation
        histogram_based_intensity_values = get_intensity_values_from_histogram(image_2_process, perc_v=[hyst_filt_bot_perc, hyst_filt_top_perc], multiplication_f=hyst_filt_bot_stdfactor)
        low_hyst_threshold_perc = histogram_based_intensity_values[4][0]
        high_hyst_threshold = histogram_based_intensity_values[4][1]
        low_hyst_threshold_dist = histogram_based_intensity_values[5]
        #If filter_choice_logic is set to 'strict'
        #Use as low value for hysteresis -based filtering the highest between that calculated using the percentile and the mode plus standard deviation of histogram intensity
        #  distribution
        if filter_choice_logic == 'strict':
            if low_hyst_threshold_dist>low_hyst_threshold_perc:
                low_hyst_threshold = low_hyst_threshold_dist
            else:
                low_hyst_threshold = low_hyst_threshold_perc
        
        #If filter_choice_logic is set to 'loose'
        # Use as low value for hysteresis -based filtering the smallest between that calculated using the percentile and the mode plus standard deviation of histogram intensity
        #  distribution
        elif filter_choice_logic == 'loose':
            if low_hyst_threshold_dist<low_hyst_threshold_perc:
                low_hyst_threshold = low_hyst_threshold_dist
            else:
                low_hyst_threshold = low_hyst_threshold_perc
        else:
            print("because you provide a value for the low_hyst_threshold_dist you are asked to choose between a 'strict' and 'loose' as filter_choice_logic")

        #Apply hysteresis-based filtering
        hysteresis_filt_img = apply_hysteresis_threshold(input_img_copy, low=low_hyst_threshold, high=high_hyst_threshold)
    
    #If a values is provided to calculate the high value of the hysteresis-based filtering by summing the standard deviantion of histogram intensity to the mode of the
    #  distribution
    elif hyst_filt_bot_stdfactor == None and hyst_filt_top_stdfactor != None:
        #Get mode, high, low values to apply to hysteresis-based filtering based on percentiles and based of the mode of the histogram intensity distribution plus the 
        # standard deviation
        histogram_based_intensity_values = get_intensity_values_from_histogram(image_2_process, perc_v=[hyst_filt_bot_perc, hyst_filt_top_perc], multiplication_f=hyst_filt_top_stdfactor)
        low_hyst_threshold = histogram_based_intensity_values[4][0]
        high_hyst_threshold_perc = histogram_based_intensity_values[4][1]
        high_hyst_threshold_dist = histogram_based_intensity_values[5]
        #If filter_choice_logic is set to 'strict'
        # Use as high value for hysteresis -based filtering the highest between that calculated using the percentile and the mode plus standard deviation of histogram intensity
        #  distribution
        if filter_choice_logic == 'strict':
            if high_hyst_threshold_dist>high_hyst_threshold_perc:
                high_hyst_threshold = high_hyst_threshold_dist
            else:
                high_hyst_threshold = high_hyst_threshold_perc
        #If filter_choice_logic is set to 'loose'
        # Use as high value for hysteresis -based filtering the smallest between that calculated using the percentile and the mode plus standard deviation of histogram intensity
        #  distribution
        elif filter_choice_logic == 'loose':
            if high_hyst_threshold_dist<high_hyst_threshold_perc:
                high_hyst_threshold = high_hyst_threshold_dist
            else:
                high_hyst_threshold = high_hyst_threshold_perc
        else:
            print("because you provide a value for the high_hyst_threshold_dist you are asked to choose between a 'strict' and 'loose' as filter_choice_logic")

        #Apply hysteresis-based filtering
        hysteresis_filt_img = apply_hysteresis_threshold(input_img_copy, low=low_hyst_threshold, high=high_hyst_threshold)
    
    #If a values is provided both to calculate the high value and to calculate the low value of the hysteresis-based filtering by summing the standard deviantion
    # of histogram intensity to the mode of the distribution
    else:
        #Get mode, high, low values to apply to hysteresis-based filtering based on percentiles and based of the mode of the histogram intensity distribution plus the standard deviation
        histogram_based_intensity_values = get_intensity_values_from_histogram(image_2_process, perc_v=[hyst_filt_bot_perc, hyst_filt_top_perc], multiplication_f=[hyst_filt_bot_stdfactor, hyst_filt_top_stdfactor])
        low_hyst_threshold_perc = histogram_based_intensity_values[4][0]
        high_hyst_threshold_perc = histogram_based_intensity_values[4][1]
        low_hyst_threshold_dist = histogram_based_intensity_values[5][0]
        high_hyst_threshold_dist = histogram_based_intensity_values[5][1]

        #If filter_choice_logic is set to 'strict'
        # Use as low and high values for hysteresis -based filtering the highest between that calculated using the percentile and the mode plus standard deviation of histogram
        #  intensity distribution
        if filter_choice_logic == 'strict':
            if low_hyst_threshold_dist>low_hyst_threshold_perc:
                low_hyst_threshold = low_hyst_threshold_dist
            else:
                low_hyst_threshold = low_hyst_threshold_perc
            if high_hyst_threshold_dist>high_hyst_threshold_perc:
                high_hyst_threshold = high_hyst_threshold_dist
            else:
                high_hyst_threshold = high_hyst_threshold_perc
        
        #If filter_choice_logic is set to 'loose'
        # Use as low and high values for hysteresis -based filtering the smallest between that calculated using the percentile and the mode plus standard deviation of histogram
        #  intensity distribution
        elif filter_choice_logic == 'loose':
            if low_hyst_threshold_dist<low_hyst_threshold_perc:
                low_hyst_threshold = low_hyst_threshold_dist
            else:
                low_hyst_threshold = low_hyst_threshold_perc
            
            if high_hyst_threshold_dist<high_hyst_threshold_perc:
                high_hyst_threshold = high_hyst_threshold_dist
            else:
                high_hyst_threshold = high_hyst_threshold_perc

        #Follow the specific logic for the choice of high low values of hysteresis-based segmentation if provided
        elif filter_choice_logic == 'custom_1':
            if high_hyst_threshold_perc>=high_hyst_threshold_dist:
                high_hyst_threshold = high_hyst_threshold_perc
            else:
                high_hyst_threshold = high_hyst_threshold_dist

            if low_hyst_threshold_dist > high_hyst_threshold_perc:
                low_hyst_threshold = low_hyst_threshold_dist
            else:
                if low_hyst_threshold_perc>=low_hyst_threshold_dist:
                    low_hyst_threshold = low_hyst_threshold_dist
                else:
                    low_hyst_threshold = low_hyst_threshold_perc

        else:
            print("because you provide values for the low_hyst_threshold_dist and high_hyst_threshold_dist you are asked to also indicate a filter_choice_logic. Choose between 'strict', 'loose' or other custom logics available")

        #Apply hysteresis-based filtering
        hysteresis_filt_img = apply_hysteresis_threshold(input_img_copy, low=low_hyst_threshold, high=high_hyst_threshold)
    
    #Rescale hysteresis-based binary image in the chosen value range
    rescaled_hysteresis_filt_img = np.where(hysteresis_filt_img>0, output_highval, output_lowval).astype(output_dtype)

    #Further remove every puncta which is detected outside the roi_mask if an roi_mask is provided
    if hasattr(roi_mask, "__len__"):
        final_filtered_img = np.where(roi_mask>0, rescaled_hysteresis_filt_img, output_lowval).astype(output_dtype)
    else:
        final_filtered_img = rescaled_hysteresis_filt_img.copy()
    
    return final_filtered_img


def get_highpass_based_segmentaion_img(image_to_threshold, high_pass__threshold, roi__ma_s_k=None,
                                       output_lowval=0, output_highval=255, output_dtype=np.uint8):
    """
    Given an input image (image_to_threshold) and a intensity value threshold (high_pass__threshold), returns a binary mask of the input image where pixels with intensity value
    higher than high_pass__threshold are set to output_highval and pixels with intensity value lower than high_pass__threshold are set to output_lowval.
    The default output image is of dtype uint8 and values 0 and 255.
    If an roi mask is provided (roi__ma_s_k), the pixels outside the roi are set to output_lowval. The roi mask is interpreted as positive pixels are the pixels of interest.
    """
    #Copy the input image
    image_to_threshold_copy = image_to_threshold.copy()

    # Binarize the image using the highpass threshold calculated threshold
    high__pass__thresh_img = np.where(image_to_threshold_copy>high_pass__threshold, 1, 0)

    #Set binary values outside roi__ma_s_k to 0 if roi__ma_s_k is provided
    if hasattr(roi__ma_s_k, "__len__"):
        roi_high__pass__thresh_img = np.where(roi__ma_s_k>0, high__pass__thresh_img, 0)

    else:
        roi_high__pass__thresh_img = high__pass__thresh_img.copy()
    
    #Rescale image in the unit8 range
    rescaled_roi_high__pass__thresh_img = np.where(roi_high__pass__thresh_img>0, output_highval, output_lowval).astype(output_dtype)

    return rescaled_roi_high__pass__thresh_img



def get_maxima_based_segmentation_img(img__2_segment, maxima__position, i_nitial_max_order=10, f_inal_max_order=3, h_ist_bins=100, ro_i__mask=None,
                                      output_lowval=0, output_highval=255, output_dtype=np.uint8):
    """
    Returns a binary mask of an input image (img__2_segment) where the intesity threshold is calculated as follow:
    1) the maxima of the histogram distribution of intensity values is calculated.
    2) A position (maxim__position) is inputed indicating the position of the maxima to use as threshold within the maxima calculated in point 1. The position counting assumes
       starting from the origin of the histogram distribution x axis.
    3) Pixels with intensity values higher than the threshold calculated in point 2 are set to output_highval, the others to output_lowval.

    The default output image is of dtype uint8 and values 0 and 255.

    If an roi mask is provided (ro_i__mask), the analysis is restricted to this regeion and pixels outside the roi are set to 0.
    The roi mask is interpreted as positive pixels are the pixels of interest.
    """
    #Copy input image
    img__2_segment_copy = img__2_segment.copy()
    
    #If roi_mask is provided, get the array of pixels in the input image which are in the roi_mask. Else use the entire image
    if hasattr(ro_i__mask, "__len__"):
        image_2__process = img__2_segment_copy[ro_i__mask>0]
    else:
        full_img_zeroes_array = np.ones(img__2_segment_copy.shape) #Note: I should check if it is necessary to pass a zero array instead of just working with the input image
        image_2__process = img__2_segment_copy[full_img_zeroes_array>0]

    #Calculate the highpass threshold by selecting the maxima corresponding to the inputed position
    highpass__threshold = detect_maxima_in_hist_distribution(image_2__process, maxima__position, initial_max_order=i_nitial_max_order, final_max_order=f_inal_max_order, hist_bins=h_ist_bins)

    # Binarize the image using get_highpass_based_segmentaion_img
    high__pass_thresh_img = get_highpass_based_segmentaion_img(img__2_segment_copy, highpass__threshold, roi__ma_s_k=ro_i__mask,
                                                               output_lowval=output_lowval, output_highval=output_highval, output_dtype=output_dtype)

    return high__pass_thresh_img



def get_frangi_based_segmentation_img(img_2_segment, maxima_position, i_initial_max_order=10, i_final_max_order=3,
                                      i_hist_bins=100, i_boarder_value=0, i_boarder_size=8, roi__mask=None,
                                      output_lowval=0, output_highval=255, output_dtype=np.uint8, **kwargs):
    """
    Given an input image (img_2_segment), the function:
    1) filters the image using Frangi filtering
    2) Binarizes the Frangi filtered image using get_maxima_based_segmentation_img. The position of the maxima to use as highpass threshold is inputed (maxima_position)
    3) Return the binary mask.

    The default output image is of dtype uint8 and has values 0 and 255.

    If an roi mask is provided (roi__mask). The Frangi filtering and initial thresholding using get_maxima_based_segmentation_img are performed on the full image.
    However after obtaining the thresholded mask, pixels outside the roi are set to 0. The roi mask is interpreted as positive pixels are the pixels of interest.
    The choice of performing filtering and thresholding on the entire image is to avoid artifacts due to edges generated by mask-based segmentation.
    """

    #Copy input image
    img_2_segment_copy = img_2_segment.copy()

    #Filter the input image using Frangi filter - NOTE: Frangi filtering is not restricted to the roi__mask provided
    frangi_filtered_img = frangi_filter(img_2_segment_copy, **kwargs)
    
    #Get image segmentation based on maxima position - NOTE: Maxima detection is not restricted to the roi__mask provided
    maxima_based_segmantion_img = get_maxima_based_segmentation_img(frangi_filtered_img,
                                                                    maxima_position,
                                                                    i_nitial_max_order=i_initial_max_order,
                                                                    f_inal_max_order=i_final_max_order,
                                                                    h_ist_bins=i_hist_bins,
                                                                    )
    
    #Further set the area outside roi__mask to 0 if provided
    if hasattr(roi__mask, "__len__"):
        roi_segmented_frangi_filtered_img = np.where(roi__mask>0, maxima_based_segmantion_img, 0)
    else:
        roi_segmented_frangi_filtered_img = maxima_based_segmantion_img.copy()

    #Further set the boarder values to 0. NOTE: this is required because Frangi filtering could lead to bright boarders.
    #In principle if a roi is passed to the get_maxima based function, boarders could aready be 0, but this is, of course, not guaranteed
    no_boarders_roi_segmented_frangi_filtered_img = set_boarder_to_value(roi_segmented_frangi_filtered_img, boarder_value=i_boarder_value, boarder_size=i_boarder_size)

    #Rescale image in the desired output range
    rescaled_no_boarders_roi_segmented_frangi_filtered_img = np.where(no_boarders_roi_segmented_frangi_filtered_img>0, output_highval, output_lowval).astype(output_dtype)

    return rescaled_no_boarders_roi_segmented_frangi_filtered_img


def hysteresis_segmentation_over_axis(input_ima_ge, hyst_filt_low_percentile, hyst_filt_high_percentile,
                                      hyst_filt_low_stdfactor=None, hyst_filt_high_stdfactor=None, filtering_logic='strict',
                                      roi_mas_k=None, img__4__histogram=None, output_low_val=0, output_high_val=255, output_d_type=np.uint8, iteration_axis=0):
    """
    Given a 3D array as input_ima_ge, the function iterates get_hysteresis_based_segmentation on one of the axes (parameter iteration_axis, default axis 0).
    roi_mas_k and img__4__histogram are respectively passed to roi_mask and img_4_histogram. In both cases, either a 3D array of same shape of input_ima_ge or
    a 2D array can be passed. If a 2D array is passed, it will be used for all the 2D arrays of input_ima_ge along the iteration_axis. If a 2D array is passed, 
    the shape must match the shape of the 2D arrays obtained from input_ima_ge along the iteration_axis.
    """

    #Make sure that input image is the correct dimension
    assert len(input_ima_ge.shape)==3, "the function currently only supports 3D arrays"

    #Copy input image
    input_ima_ge_copy = input_ima_ge.copy()

    #Form a list of 2D images based on the axis to use for iteration.
    if iteration_axis==0:
        input_ima_ge_list = [input_ima_ge_copy[tp,...] for tp in range(input_ima_ge_copy.shape[0])]
    elif iteration_axis==1:
        input_ima_ge_list = [input_ima_ge_copy[:,tp1,:] for tp1 in range(input_ima_ge_copy.shape[1])]
    elif iteration_axis==2:
        input_ima_ge_list = [input_ima_ge_copy[...,tp2] for tp2 in range(input_ima_ge_copy.shape[2])]

    #If roi mask is provided
    if hasattr(roi_mas_k, "__len__"):

        #make sure it is the correct dimension
        if not (len(roi_mas_k.shape)>1 and len(roi_mas_k.shape)<4):
            print("roi_mas_k mus't be a 2D or 3D image")
            return
        
        #Copy it
        roi_mas_k_copy = roi_mas_k.copy()
        
        #Form a list of 2D images based on the axis to use for iteration.
        #NOTE: if it is a 2D array. Iterate the array along iteration_axis to match the lengh of iteration_axis in input_ima_ge
        if len(roi_mas_k.shape)==2:
            roi_mas_k_list = [roi_mas_k_copy for i in range(len(input_ima_ge_list))]
        else:
            if iteration_axis==0:
                roi_mas_k_list = [roi_mas_k_copy[tp3,...] for tp3 in range(roi_mas_k_copy.shape[0])]
            elif iteration_axis==1:
                roi_mas_k_list = [roi_mas_k_copy[:,tp4,:] for tp4 in range(roi_mas_k_copy.shape[1])]
            elif iteration_axis==2:
                roi_mas_k_list = [roi_mas_k_copy[...,tp5] for tp5 in range(roi_mas_k_copy.shape[2])]
    
    #If img__4__histogram is provided
    if hasattr(img__4__histogram, "__len__"):
        
        #make sure it is the correct dimension
        if not (len(img__4__histogram.shape)>1 and len(img__4__histogram.shape)<4):
            print("img__4__histogram mus't be a 2D or 3D image")
            return

        #Copy it
        img__4__histogram_copy = img__4__histogram.copy()

        #Form a list of 2D images based on the axis to use for iteration.
        #NOTE: if it is a 2D array. Iterate the array along iteration_axis to match the lengh of iteration_axis in input_ima_ge
        if len(img__4__histogram.shape)==2:
            img__4__histogram_list = [img__4__histogram_copy for i in range(len(input_ima_ge_list))]
        else:
            if iteration_axis==0:
                img__4__histogram_list = [img__4__histogram_copy[tp6,...] for tp6 in range(img__4__histogram_copy.shape[0])]
            elif iteration_axis==1:
                img__4__histogram_list = [img__4__histogram_copy[:,tp7,:] for tp7 in range(img__4__histogram_copy.shape[1])]
            elif iteration_axis==2:
                img__4__histogram_list = [img__4__histogram_copy[...,tp8] for tp8 in range(img__4__histogram_copy.shape[2])]

    #Initialize an output array
    output_3D_array = np.zeros(input_ima_ge_copy.shape)

    #Iterate through the 2D arrays of the input image along the iteration axis
    for pos_counter, arr_2D in enumerate(input_ima_ge_list):

        #Get the 2D array for roi mask if it is provided, otherwise set the variable to None
        if hasattr(roi_mas_k, "__len__"):
            arr_2D_roi = roi_mas_k_list[pos_counter]
        else:
            arr_2D_roi = None
        
        #Get the 2D array for img__4__histogram if it is provided, otherwise set the variable to None
        if hasattr(img__4__histogram, "__len__"):
            arr_2D_img4hist = img__4__histogram_list[pos_counter]
        else:
            arr_2D_img4hist = None


        #Obtain the hysteresis based segmentation of the 2D array
        hyst_based_segmented_img = get_hysteresis_based_segmentation(arr_2D,
                                                                     hyst_filt_bot_perc=hyst_filt_low_percentile,
                                                                     hyst_filt_top_perc=hyst_filt_high_percentile,
                                                                     hyst_filt_bot_stdfactor=hyst_filt_low_stdfactor,
                                                                     hyst_filt_top_stdfactor=hyst_filt_high_stdfactor,
                                                                     filter_choice_logic=filtering_logic,
                                                                     roi_mask=arr_2D_roi,
                                                                     img_4_histogram=arr_2D_img4hist,
                                                                     output_lowval=0,
                                                                     output_highval=255,
                                                                     output_dtype=np.uint8)
        
        #Modify the output array according to the iteration axis
        if iteration_axis==0:
            output_3D_array[pos_counter,...] += hyst_based_segmented_img
        elif iteration_axis==1:
            output_3D_array[:,pos_counter,:] += hyst_based_segmented_img
        elif iteration_axis==2:
            output_3D_array[...,pos_counter] += hyst_based_segmented_img

    #Rescale the output array in the desired value range and dtype
    rescaled_output_3D_array = np.where(output_3D_array>0, output_high_val, output_low_val).astype(output_d_type)

    return rescaled_output_3D_array

