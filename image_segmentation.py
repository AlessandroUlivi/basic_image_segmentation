import numpy as np
from utils import get_intensity_values_from_histogram, set_boarder_to_value, detect_maxima_in_hist_distribution
from skimage.filters import apply_hysteresis_threshold
from image_filtering import frangi_filter


def get_hysteresis_based_segmentation(input_img, hyst_filt_bot_perc, hyst_filt_top_perc, hyst_filt_bot_stdfactor=None, hyst_filt_top_stdfactor=None, filter_choice_logic='strict', roi_mask=None):
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

    Output: binary segmentation of putative puncta structures as numpy array of the same shape of the input image. NOTE: the output is, by default of type np.uint8 with values 0 and 255.
    """
    #Copy input image
    input_img_copy = input_img.copy()
    
    #If roi_mask is provided, get the array of pixels in the input image which are in the roi_mask. Else use the entire image
    if hasattr(roi_mask, "__len__"):
        image_2_process = input_img_copy[roi_mask>0]
    else:
        full_image_zeroes_array = np.ones(input_img_copy.shape) #Note: I should check if it is necessary to pass a zero array instead of just working with the input image
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
    
    #Rescale hysteresis-based binary image on the uint8 value range
    uint8_hysteresis_filt_img = np.where(hysteresis_filt_img>0, 255, 0).astype(np.uint8)

    #Further remove every puncta which is detected outside the roi_mask if an roi_mask is provided
    if hasattr(roi_mask, "__len__"):
        final_filtered_img = np.where(roi_mask>0, uint8_hysteresis_filt_img, 0).astype(np.uint8)
    else:
        final_filtered_img = uint8_hysteresis_filt_img.copy()
    
    return final_filtered_img


def get_frangi_based_segmentation_img(img_2_segment, maxima_position, i_initial_max_order=10, i_final_max_order=3, i_hist_bins=100, roi__mask=None, **kwargs):
    #Copy input image
    img_2_segment_copy = img_2_segment.copy()

    #Filter the input image using Frangi filter
    frangi_filtered_img = frangi_filter(img_2_segment_copy, **kwargs)
    
    #If roi_mask is provided, get the array of pixels in the input image which are in the roi_mask. Else use the entire image
    if hasattr(roi__mask, "__len__"):
        image_2__process = frangi_filtered_img[roi__mask>0]
    else:
        full_img_zeroes_array = np.ones(img_2_segment_copy.shape) #Note: I should check if it is necessary to pass a zero array instead of just working with the input image
        image_2__process = frangi_filtered_img[full_img_zeroes_array>0]

    #Calculate the highpass threshold by selecting the maxima corresponding to the inputed position
    highpass_threshold = detect_maxima_in_hist_distribution(image_2__process, maxima_position, initial_max_order=i_initial_max_order, final_max_order=i_final_max_order, hist_bins=i_hist_bins)

    return frangi_filtered_img

    # # Binarize the image using the Frangi-based highpass calculated threshold
    # high__pass_thresh_frangi_filtered_img = np.where(frangifiltered_input_pict_ure>threshold_frangi_val, 1000, 0)
    
    # # Set binary values outside the embryo to 0
    # only_embryo_frangi_binary_img = np.where(thresholded_embryo_input>0, high__pass_thresh_frangi_filtered_img, 0)

    # #Set border values to 0 - NOTE: this is required because Frangi-based filtering leads to the detection of the picture borders. I thouht that the step just above (Set binary values outside the embryo to 0) would have taken care of it, but it doesn't in cases where some tissue is at the border
    # no_borders_only_embryo_frangi_binary_img = set_border_to_0(only_embryo_frangi_binary_img)
    
    # #Rescale image in the unit8 range
    # uint8_only_embryo_frangi_binary_img = np.where(no_borders_only_embryo_frangi_binary_img>0, 255, 0).astype(np.uint8)

    # return uint8_only_embryo_frangi_binary_img
