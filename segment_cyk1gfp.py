import os
import tifffile
import numpy as np
from skimage.util import img_as_int
from utils import listdirNHF, get_minima_after_mode_in_hist_distribution, detect_maxima_in_hist_distribution, set_boarder_to_value, join_masks
from image_filtering import blur_image, highpass_area_filter, bilateral_filter_image, frangi_filter, filter_mask1_on_mask2, filter_mask1_by_centroid_distance_from_mask2, median_blur_image
from image_segmentation import get_highpass_based_segmentaion_img, get_hysteresis_based_segmentation, get_highpass_based_segmentaion_img, hysteresis_segmentation_over_axis


def organize_2_channels_files(input_dir, ch1_trgt_name, ch2_trgt_name, file_extension='.tif', prefix_channel='ch1', reduce_1_pixel=False, axis_2_reduce=2):
    
    #Initialize a dictionary which links channels to their files
    organized_dict_out = {ch1_trgt_name:{}, ch2_trgt_name:{}}
    
    #Get a list of the the files in input_dir
    input_files_l = listdirNHF(input_dir)

    #If reduce 1 pixel is True, initialize a dictionary linking the size of the axis_2_reduce to the name of the target channel names
    if reduce_1_pixel:
        size_name_dict = {}
    
    #Iterate through the input files
    for f in input_files_l:
    
        #Double-check that file has the correct file extension
        if file_extension in f:
            
            #Open the file
            f_ile = tifffile.imread(os.path.join(input_dir, f))
            
            #Link target channel names to their files. If the channel is selected to be used for the prefix name, link the
            # part of the name before the target string, to the 'prefix_name' string in the output dictionary
            if ch1_trgt_name in f:
                # print("CH1", tc)
                organized_dict_out[ch1_trgt_name] = f_ile
                if prefix_channel=='ch1':
                    organized_dict_out['prefix_name'] = f[:f.index(ch1_trgt_name)] + "_"
                #If reduce_1_pixel is True, link the the size of the axis_2_reduce to the name of the target channel names in size_name_dict
                if reduce_1_pixel:
                    size_name_dict[f_ile.shape[axis_2_reduce]]=ch1_trgt_name

            elif ch2_trgt_name in f:
                # print("CH2", tc)
                organized_dict_out[ch2_trgt_name] = f_ile
                if prefix_channel=='ch2':
                    organized_dict_out['prefix_name'] = f[:f.index(ch2_trgt_name)] + "_"

                #If reduce_1_pixel is True, link the the size of the axis_2_reduce to the name of the target channel names in size_name_dict
                if reduce_1_pixel:
                    size_name_dict[f_ile.shape[axis_2_reduce]]=ch2_trgt_name
            else:
                print("some problem here")
    
    if reduce_1_pixel:
        #Get the file which has the maximum size for the dimension axis_2_reduce
        file_2_reduce = organized_dict_out[size_name_dict[max(list(size_name_dict))]]
        #Crop the file of 1 pixel on the axis_2_reduce dimension
        if axis_2_reduce==0:
            cropped_file_2_reduce = file_2_reduce[0:file_2_reduce.shape[0]-1,:,:]
        elif axis_2_reduce==1:
            cropped_file_2_reduce = file_2_reduce[:,0:file_2_reduce.shape[1]-1,:]
        elif axis_2_reduce==2:
            cropped_file_2_reduce = file_2_reduce[:,:,0:file_2_reduce.shape[2]-1]
        else:
            print("for the moment the reduction of 1 pixel is implemented only on 2 or 3D files")
        
        #link the cropped file to its target name in organized_dict_out
        organized_dict_out[size_name_dict[max(list(size_name_dict))]]=cropped_file_2_reduce
        
        return organized_dict_out

    else:
        return organized_dict_out


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



def segment_pip2_enriched_domains(raw_timelapse_pip2, embryomask_timelapse, sigma_gaussian_smoothing, diameter_bilateral_smoothing, sigmacol_bilateral_smoothing,
                                  sigmaspac_bilateral_smoothing, toppercent_hyster, botpercent_hyster,topstdv_hyster, botstdv_hyster, peakposition_frangi,
                                  area_highpass_threshold, distance_highpass_threshold, axis_to_use=0, return_processing_steps=False, output_arr_lowval=0,
                                  output_arr_highval=255, output_arr_dtype=np.uint8, **kwargs):
    """
    Process all timepoints of a pip2-mCherry timecourse to obtain a segmentation mask for pip2-enriched domains.
    The function only works with 3D arrays. The default time-axis is set to axis 0.
    If return_processing_steps is set to True, a dictionary linking each timepoint to its processed steps is returned at outout position 1. Processed steps are in a list object,
    in the following order:
    - position 0, raw_timepoint.
    - position 1, gaussian blurred timepoint.
    - position 2, bilateral filtered timepoint.
    - position 3, embryo segmentation mask for the timepoint.
    - position 4, hysteresis -based segmented timepoint.
    - position 5, frangi filtered timepoint.
    - position 6, intensity value corresponding to the highest maxima value (from x axis orgigin) detected in frangi-filtered histogram distribution of intensity values
    - position 7, binary segmentation mask obtained by using value of position 6 as highpass intensity threshold of frangi filtered image.
    - position 8, binary segmentation mask obtained by keeping the areas of hysteresis-based segmentation (position 4) if at least 1 pixel of these areas is positive in the frangi-based segmentation mask (position 7).
    - position 9, binary segmentation mask obtained by keeping the areas of frangi-based segmentation (position 7) if the minimum distance between their centroid and the background pixels of the embryo segmentation mask (position 3) further apart than distance_highpass_threshold.
    - position 10, dictionary linking centroids coordinates to the coordinates of the closest pixels in the background pixels of the embryo segmentation mask.
    - position 11, binary segmentation mask obtained by joining frangi-filtered hysteresis-based segmentation mask (point position 8) and distance-filtered frangi-based segmentation mask (position 9)
    - position 12, binary segmentation mask obtained by keeping areas of joint segmentation masks (position 11) if their number of pixels is higher than area_highpass_threshold (NOTE: this is the output image for the timepoint)
    """

    #Copy the input images
    raw_timelapse_pip2_copy = raw_timelapse_pip2.copy()
    embryomask_timelapse_copy = embryomask_timelapse.copy()

    #For a list of 2D arrays based on the axis to use
    if axis_to_use==0:
        raw_timelapse_pip2_list = [raw_timelapse_pip2_copy[tp,...] for tp in range(raw_timelapse_pip2_copy.shape[0])]
        embryomask_timelapse_list = [embryomask_timelapse_copy[tp1,...] for tp1 in range(embryomask_timelapse_copy.shape[0])]
    elif axis_to_use==1:
        raw_timelapse_pip2_list = [raw_timelapse_pip2_copy[:,tp2,:] for tp2 in range(raw_timelapse_pip2_copy.shape[1])]
        embryomask_timelapse_list = [embryomask_timelapse_copy[:,tp3,:] for tp3 in range(embryomask_timelapse_copy.shape[1])]
    elif axis_to_use==2:
        raw_timelapse_pip2_list = [raw_timelapse_pip2_copy[...,tp4] for tp4 in range(raw_timelapse_pip2_copy.shape[2])]
        embryomask_timelapse_list = [embryomask_timelapse_copy[...,tp5] for tp5 in range(embryomask_timelapse_copy.shape[2])]
    
    #Initialize a dictionary linking indivual timepoints to their processed images, this will save time in the second part of the script
    dict_tmpnt_to_processed_img_and_frangithresh = {}

    #Initialize a list to collect all Frangi thresholds
    frangi_thresh_collect_l = []
    
    #Iterate through timepoints
    # for tmpnt__2ww_c, tmpnt__2ww in enumerate(list(raw_timelapse_pip2_copy)):
    for tmpnt__2ww_c, tmpnt__2ww in enumerate(raw_timelapse_pip2_list):
        # print("===",tmpnt__2ww_c)
        # print("raw shape", tmpnt__2ww.shape)
        #Use extract_timepoint_2ww_wp to get the raw image, the gaussian smoothed image, the bilateral filtered image, the embryo thresholded mask, the values of the pixels within the embryo mask of the gaussian smoothed and bilateral filtered images, for the current timepoint
        # tmpnt___im_raw, tmpnt___im_gaussian, tmpnt___im_bilateral, tmpnt___im_embryo_thresh, ovth_tmpnt___im_gauss, ovth_tmpnt___im_bilat = extract_smooth_timepoint_2ww(raw_timelapse_ch3, sigma_gaussian_smoothing, (diameter_bilateral_smoothing, sigmacol_bilateral_smoothing, sigmaspac_bilateral_smoothing), embryomask_timelapse, tmpnt__2ww_c)
        #Denoise a bit the timepoint image by gaussian smoothing
        tmpnt___im_gaussian = blur_image(tmpnt__2ww, n=sigma_gaussian_smoothing)
        # print("gaussian shape", tmpnt___im_gaussian.shape)
        #Denoise a bit the timepoint image, while preserving edges by bilateral filtering
        tmpnt___im_bilateral = bilateral_filter_image(tmpnt__2ww, diameter_bilateral_smoothing, sigmacol_bilateral_smoothing,sigmaspac_bilateral_smoothing)
        # print("bilateral shape", tmpnt___im_bilateral.shape)
        #Get the embryo binary mask for the timepoint
        tmpnt___im_embryo_thresh = embryomask_timelapse_list[tmpnt__2ww_c]
        # print("embryo mask shape", tmpnt___im_embryo_thresh.shape)
        # print("embryo mask values", np.unique(tmpnt___im_embryo_thresh))

        #Use hysteresis-based segmentation to obtain a binary mask of the gaussian smoothed timepoint - NOTE: restrict the process to the segmented embryo mask
        #NOTE2: filtering method is set to "custom_1", refer to get_hysteresis_based_segmentation function to understand what it does.
        tmpnt___im_hyst = get_hysteresis_based_segmentation(tmpnt___im_gaussian,
                                                            hyst_filt_bot_perc=botpercent_hyster,
                                                            hyst_filt_top_perc=toppercent_hyster,
                                                            hyst_filt_bot_stdfactor=botstdv_hyster,
                                                            hyst_filt_top_stdfactor=topstdv_hyster,
                                                            filter_choice_logic='custom_1',
                                                            roi_mask=tmpnt___im_embryo_thresh,
                                                            img_4_histogram=None,
                                                            output_lowval=0,
                                                            output_highval=255,
                                                            output_dtype=np.uint8)
        
        # print("hysteresis mask shape", tmpnt___im_hyst.shape)
        # print("hysteresis mask values", np.unique(tmpnt___im_hyst))

        #Filter the bilateral smoothed image using Frangi filter - NOTE: Don't restrict the process to the segmented embryo mask
        frangi_filtered_tmpnt = frangi_filter(img_as_int(tmpnt___im_bilateral), **kwargs)
        # print("frangi shape", frangi_filtered_tmpnt.shape)

        #Get the intensity value in the frangi-filtered image corresponding to the highest maxima value (from the x axis orgin).
        # This can also be seen as the first maxima from the end of x axis.
        frangi_based_threshold = detect_maxima_in_hist_distribution(frangi_filtered_tmpnt, peakposition_frangi)
        # print("frangi threshold", frangi_based_threshold)

        #Link timepoint to its processing steps and to the intensity value corresponding to the highest maxima value of frangi-filtered histogram distribution of intensity values
        # dict_tmpnt_to_processed_img_and_frangithresh[tmpnt__2ww_c] = [tmpnt__2ww, tmpnt___im_gaussian, tmpnt___im_bilateral, tmpnt___im_embryo_thresh,ovth_tmpnt___im_gauss, ovth_tmpnt___im_bilat,tmpnt___im_hyst, frangi_based_filtered_img, frangi_based_threshold]
        dict_tmpnt_to_processed_img_and_frangithresh[tmpnt__2ww_c] = [tmpnt__2ww,
                                                                      tmpnt___im_gaussian,
                                                                      tmpnt___im_bilateral,
                                                                      tmpnt___im_embryo_thresh,
                                                                      tmpnt___im_hyst,
                                                                      frangi_filtered_tmpnt,
                                                                      frangi_based_threshold]
        
        #Collect the intensity value corresponding to the highest maxima value of frangi-filtered histogram distribution of intensity values in frangi_thresh_collect_l list
        frangi_thresh_collect_l.append(frangi_based_threshold)
        
        #Print the progress
        if tmpnt__2ww_c in [len(list(raw_timelapse_pip2_list))//8, len(list(raw_timelapse_pip2_list))//4, (len(list(raw_timelapse_pip2_list))//8)*3,
                            len(list(raw_timelapse_pip2_list))//2, (len(list(raw_timelapse_pip2_list))//8)*5, (len(list(raw_timelapse_pip2_list))//4)*3,
                            (len(list(raw_timelapse_pip2_list))//8)*7, len(list(raw_timelapse_pip2_list))]:
            print("="*3, tmpnt__2ww_c, "/",len(list(raw_timelapse_pip2_list)))
    
        
    print("=== finished initial filtering ===")
    # return dict_tmpnt_to_processed_img_and_frangithresh

    #Calculate the median and standard deviations of calculated intensity values corresponding to the highest maxima values of the histogram distributions of intensity values
    # for each frangi-filtered timepoint
    frangi_thresh_med, frangi_thresh_stdv = np.median(frangi_thresh_collect_l), np.std(frangi_thresh_collect_l)
    
    #Initialize the output array
    output_array = np.zeros(raw_timelapse_pip2_copy.shape)

    #Iterate through the timepoints of the dict_tmpnt_to_frangi_img_thresh dictionary
    for tm_pnt in dict_tmpnt_to_processed_img_and_frangithresh:
             
        #Get the raw image
        tmpnt__img_raw = dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt][0]
    
        #Get the gaussian smoothed image
        tmpnt__img_gaussian = dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt][1]

        #Get the bilateral smoothed image
        tmpnt__img_bilateral = dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt][2]
        
        #Get the embryo binary mask
        tmpnt__img_embryo_thresh = dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt][3]
    
        # #Get the array of the over embryo threshold pixels in the gaussian smoothed picture
        # tmpnt_ovth_embryo = tmpnt__img_gaussian[tmpnt__img_embryo_thresh>0]
    
        #Get hysteresis-based segmentation mask
        hysteresis_based_binary_img = dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt][-3]
        
        #Get Frangi-filtered image
        frangi_filtered__img = dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt][-2]
        
        #Get intensity value corresponding to the highest maxima value of the histogram distributions of intensity values for the current frangi-filtered timepoint
        #This value is the initial guess for the binarization of frangi-filtered image
        i_frangi_based_thresh = dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt][-1]
        
        #use i_frangi_based_thresh as threshold to binarize frangi-filtered image if it is within 1 a standard deviation from the global median of frangi based threshold,
        # otherwise use the globa median
        if ((i_frangi_based_thresh>(frangi_thresh_med-(1*frangi_thresh_stdv))) and (i_frangi_based_thresh<(frangi_thresh_med+(1*frangi_thresh_stdv)))):
            frangi_based_thresh = i_frangi_based_thresh
        else:
            frangi_based_thresh = frangi_thresh_med
    
        #Get Frangi-based binary image
        frangi_based_binary_img_i = get_highpass_based_segmentaion_img(frangi_filtered__img,
                                                                     frangi_based_thresh,
                                                                     roi__ma_s_k=tmpnt__img_embryo_thresh,
                                                                     output_lowval=0,
                                                                     output_highval=255,
                                                                     output_dtype=np.uint8)

        #Set boarders of frangi_based_binary_img to 0 - this is required because Frangi filtering leads to bright boarders which might pass the highpass segmentation threshold.
        #In principle if having passed the embryo binary mask to get_highpass_based_segmentaion_img boarders could aready be 0, but this is not guaranteed.
        frangi_based_binary_img = set_boarder_to_value(frangi_based_binary_img_i, boarder_value=0, boarder_size=8)

        #If return_processing_steps is selected, append frangi-based segmentation mask to the list linked to the present timepoint within dict_tmpnt_to_processed_img_and_frangithresh
        if return_processing_steps:
            dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt].append(frangi_based_binary_img)
        
        #Filter Hysteresis-based binary image to remove structures which don't overlap at all with Frangi detected structures
        frangi_filtered_hyst_based_binary_img = filter_mask1_on_mask2(hysteresis_based_binary_img,
                                                                      frangi_based_binary_img,
                                                                      pixels_highpass_threshold=0,
                                                                      output_lowval=0,
                                                                      output_highval=255,
                                                                      output_dtype=np.uint8)
        
        #If return_processing_steps is selected, append frangi_filtered_hyst_based_binary_img to the list linked to the present timepoint within dict_tmpnt_to_processed_img_and_frangithresh
        if return_processing_steps:
            dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt].append(frangi_filtered_hyst_based_binary_img)

        #Filter frangi-based binary image to remove structures very close to the embryo boarder
        #Invert the embryo mask first
        inverted_tmpnt__img_embryo_thresh = np.where(tmpnt__img_embryo_thresh>0, 0, 255).astype(np.uint8)

        #If return_processing_steps is selected, append the output binary mask and the dictionary linking centroid coordinates to the coordinates of their closest points outside
        #the embryo, to the list linked to the present timepoint within dict_tmpnt_to_processed_img_and_frangithresh
        if return_processing_steps:
            dist_fitered_frangi_based_binary_img, coordinates_link_dict = filter_mask1_by_centroid_distance_from_mask2(frangi_based_binary_img,
                                                                                                                        inverted_tmpnt__img_embryo_thresh,
                                                                                                                        distance_thr=distance_highpass_threshold,
                                                                                                                        filtering_modality='highpass',
                                                                                                                        n_distances=2,
                                                                                                                        return_coordinates=True,
                                                                                                                        output_low_value=0,
                                                                                                                        output_high_value=255,
                                                                                                                        output_dtype=np.uint8)
            dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt].append(dist_fitered_frangi_based_binary_img)
            dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt].append(coordinates_link_dict)
        
        else:
            dist_fitered_frangi_based_binary_img = filter_mask1_by_centroid_distance_from_mask2(frangi_based_binary_img,
                                                                                                inverted_tmpnt__img_embryo_thresh,
                                                                                                distance_thr=distance_highpass_threshold,
                                                                                                filtering_modality='highpass',
                                                                                                n_distances=2,
                                                                                                return_coordinates=False,
                                                                                                output_low_value=0,
                                                                                                output_high_value=255,
                                                                                                output_dtype=np.uint8)
            
        
        #Combined [hysteresis-based segmentation mask filtered on frangi-based segmentation mask] AND
        # [frangi-based segmentation mask filtered on distance from the outside of the embryo segmentation mask]
        combined_hysteresis_frangibased_binary_img = join_masks(frangi_filtered_hyst_based_binary_img,
                                                                dist_fitered_frangi_based_binary_img,
                                                                low_binary_val=0,
                                                                high_binary_val=255,
                                                                output_dtype=np.uint8)
        
        #If return_processing_steps is selected, append combined_hysteresis_frangibased_binary_img to the list linked to the present timepoint within dict_tmpnt_to_processed_img_and_frangithresh
        if return_processing_steps:
            dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt].append(combined_hysteresis_frangibased_binary_img)

        # Filter combined image to remove very small region
        region_by_area_filtered_img = highpass_area_filter(combined_hysteresis_frangibased_binary_img,
                                                            area_highpass_threshold,
                                                            return_area_list=False,
                                                            input_is_label_image=False,
                                                            output_lowval=0,
                                                            output_highval=255,
                                                            output_dtype=np.uint8)
        
        #If return_processing_steps is selected, append region_by_area_filtered_img to the list linked to the present timepoint within dict_tmpnt_to_processed_img_and_frangithresh
        if return_processing_steps:
            dict_tmpnt_to_processed_img_and_frangithresh[tm_pnt].append(region_by_area_filtered_img)

        #Modify the output array
        if axis_to_use==0:
            output_array[tm_pnt,...] += region_by_area_filtered_img
        elif axis_to_use==1:
            output_array[:,tm_pnt,:] += region_by_area_filtered_img
        elif axis_to_use==2:
            output_array[...,tm_pnt] += region_by_area_filtered_img
        
        #Print the progress
        if tm_pnt in [len(dict_tmpnt_to_processed_img_and_frangithresh)//8,
                      len(dict_tmpnt_to_processed_img_and_frangithresh)//4,
                      (len(dict_tmpnt_to_processed_img_and_frangithresh)//8)*3,
                      len(dict_tmpnt_to_processed_img_and_frangithresh)//2,
                      (len(dict_tmpnt_to_processed_img_and_frangithresh)//8)*5,
                      (len(dict_tmpnt_to_processed_img_and_frangithresh)//4)*3,
                      (len(dict_tmpnt_to_processed_img_and_frangithresh)//8)*7,
                      len(dict_tmpnt_to_processed_img_and_frangithresh)]:
            print("="*3, tm_pnt, "/",len(dict_tmpnt_to_processed_img_and_frangithresh))


    # Rescale the output array in the desired value range and dtype
    rescaled_output_array = np.where(output_array>0, output_arr_highval, output_arr_lowval).astype(output_arr_dtype)
    
    #If return_processing_steps is selected return the rescaled output array and the dictionary linking each timepoint with all the individual processing steps.
    #Else return only the rescaled output array
    if return_processing_steps:
        return rescaled_output_array, dict_tmpnt_to_processed_img_and_frangithresh
    else:
        return rescaled_output_array


def segment_cyk1_enriched_domains(raw_timecourse_cyk1, embryo_mask_timecourse, hyst_filt_low_percentil_e, hyst_filt_high_percentil_e,
                                    hyst_filt_low_stdfacto_r=None, hyst_filt_high_stdfacto_r=None,
                                    output_low_va_l=0, output_high_va_l=255, output_d_typ_e=np.uint8, iteration_axi_s=0):
    """
    this function only supports 3D arrays
    """

    #Copy the input timecourses
    raw_timecourse_cyk1_copy = raw_timecourse_cyk1.copy()
    embryo_mask_timecourse_copy = embryo_mask_timecourse.copy()
    
    #Preprocess the raw timecourse using median filtering
    if iteration_axi_s==0:
        median_filt_timecourse_cyk1_list = [median_blur_image(raw_timecourse_cyk1_copy[a,...]) for a in range(raw_timecourse_cyk1_copy.shape[0])]
    elif iteration_axi_s==1:
        median_filt_timecourse_cyk1_list = [median_blur_image(raw_timecourse_cyk1_copy[:,b,:]) for b in range(raw_timecourse_cyk1_copy.shape[1])]
    elif iteration_axi_s==2:
        median_filt_timecourse_cyk1_list = [median_blur_image(raw_timecourse_cyk1_copy[...,c]) for c in range(raw_timecourse_cyk1_copy.shape[2])]
    
    median_filt_timecourse_cyk1 = np.asarray(median_filt_timecourse_cyk1_list)

    #Apply hysteresis-based segmentation on the timecourse - NOTE: the function is applied to the median-filtered timecourse
    hysteresis_segmented_timecourse = hysteresis_segmentation_over_axis(median_filt_timecourse_cyk1,
                                                                        hyst_filt_low_percentile=hyst_filt_low_percentil_e,
                                                                        hyst_filt_high_percentile=hyst_filt_high_percentil_e,
                                                                        hyst_filt_low_stdfactor=hyst_filt_low_stdfacto_r,
                                                                        hyst_filt_high_stdfactor=hyst_filt_high_stdfacto_r,
                                                                        filtering_logic='strict',
                                                                        roi_mas_k=embryo_mask_timecourse_copy,
                                                                        img__4__histogram=None,
                                                                        output_low_val=output_low_va_l,
                                                                        output_high_val=output_high_va_l,
                                                                        output_d_type=output_d_typ_e,
                                                                        iteration_axis=0)
    
    return hysteresis_segmented_timecourse



