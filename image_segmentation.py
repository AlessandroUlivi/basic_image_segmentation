import numpy as np
from utils import get_intensity_values_from_histogram
from skimage.filters import apply_hysteresis_threshold


def get_hysteresis_based_segmentation(input_img, hyst_filt_bot_perc, hyst_filt_top_perc, hyst_filt_bot_stdfactor=None, hyst_filt_top_stdfactor=None, filter_choice_logic='strict', roi_mask=None):
    """
    returns a binary mask with brightest pixels selected based on an hystereris-filter. The top and bottom values of the hysteresis process are both, independentely, defined as follow: first it is calculated the value to use, based on an histogram percentile.
    Secondly a value is calculated by taking the mode intensity value and summing it the standard deviation of intensity values distribution, multiplied for an indicated factor. The highest of these two values is used. This process is quite robust to images with
    low signal-to-noise (the throush signal is dim or bleach and the image is almost all background).
    
    inputs: image to process as 2D numpy array (timelapse_with_puncta_tmpt). Embryo mask (binary segmentation of the embryo cytosol) as numpy array of the same shape of timelapse_with_puncta_tmpt. Percentile to use to calculate the high value of the hysteresis filtering (hyst_filt_top_perc). 
    Percentile to use to calculate the low value of the hysteresis filtering (hyst_filt_bot_perc). Number of standard deviation to sum to mode value for calculating the high value of the hysteresis filtering (hyst_filt_top_stdfactor).
    Number of standard deviation to sum to mode value for calculating the low value of the hysteresis filtering (hyst_filt_bot_stdfactor).

    output: binary segmentation of putative puncta structures as 2D numpy array of the same shape of the input image.
    """

    #If roi_mask is provided, get the array of pixels in the input image which are in the roi_mask. Else use the entire image
    if roi_mask != None:
        image_2_process = input_img[roi_mask>0]
    else:
        full_image_zeroes_array = np.ones(input_img.shape) #Note: I should check if it is necessary to pass a zero array instead of just working with the input image
        image_2_process = input_img[full_image_zeroes_array>0]
    
    #If no values are provided to calculate hysteresis-based filtering by summing the standard deviantion of histogram intensity to the mode of the distribution
    if hyst_filt_bot_stdfactor == None and hyst_filt_top_stdfactor == None:
        #Get mode, high, low values to apply to hysteresis-based filtering based on percentiles
        histogram_based_intensity_values = get_intensity_values_from_histogram(image_2_process, perc_v=[hyst_filt_bot_perc, hyst_filt_top_perc])
        low_hyst_threshold = histogram_based_intensity_values[4][0]
        high_hyst_threshold = histogram_based_intensity_values[4][1]
        #Apply hysteresis-based filtering
        hysteresis_filt_img = apply_hysteresis_threshold(input_img, low=low_hyst_threshold, high=high_hyst_threshold)

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
        hysteresis_filt_img = apply_hysteresis_threshold(image_2_process, low=low_hyst_threshold, high=high_hyst_threshold)
    
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
        hysteresis_filt_img = apply_hysteresis_threshold(input_img, low=low_hyst_threshold, high=high_hyst_threshold)
    
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
        hysteresis_filt_img = apply_hysteresis_threshold(input_img, low=low_hyst_threshold, high=high_hyst_threshold)
    
    #Rescale hysteresis-based binary image on the uint8 value range
    uint8_hysteresis_filt_img = np.where(hysteresis_filt_img>0, 255, 0).astype(np.uint8)

    #Further remove every puncta which is detected outside the roi_mask if an roi_mask is provided
    if roi_mask != None:
        final_filtered_img = np.where(roi_mask>0, uint8_hysteresis_filt_img, 0).astype(np.uint8)
    else:
        final_filtered_img = uint8_hysteresis_filt_img.copy()
    
    return final_filtered_img

