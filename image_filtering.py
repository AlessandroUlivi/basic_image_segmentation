import numpy as np
from scipy.signal import convolve
from skimage.filters import median as medianfilter
from skimage.filters import frangi
from skimage.util import img_as_float32
from skimage.measure import label, regionprops
import cv2


def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions - this code is taken from scipy documentation https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html"""
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size)+y**2/float(sizey)))
    return g / g.sum()


def blur_image(im, n, ny=None):
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
        This code modified from scipy documentation https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """
    #Copy the input image
    im_copy = im.copy()
    g = gauss_kern(n, sizey=ny)
    improc = convolve(im_copy,g, mode='same')
    return(improc)

def median_blur_image(imm, **kwargs):
    """
    returns a smoothed copy of input image using a median filter
    """
    #Copy input image
    imm_copy = imm.copy()
    return medianfilter(imm_copy, **kwargs)


def bilateral_filter_image(img, smooth_diameter, smooth_sigma_color, smooth_sigma_space):
    """
    returns a smoothed copy of the input image using a bilateral filter
    """
    #Copy input image
    img_copy = img.copy()

    #Transform input image to type float32 before applying bilateral filtering
    img_f32 = img_as_float32(img_copy, )
    #Apply bilateral filtering
    img_bilat = cv2.bilateralFilter(img_f32, smooth_diameter, smooth_sigma_color, smooth_sigma_space)
    return img_bilat

def frangi_filter(immg, **kwargs):
    """
    returns a copy of the filtered input image after applying a frangi filtering
    """
    #Copy input image
    immg_copy = immg.copy()

    return frangi(immg_copy, **kwargs)


def highpass_area_filter(input__binary__imag_e, area_highpass_thr, return_area_list=False, input_is_label_image=False):
    """
    Given a binary input array and a threshold for area (the threshold is in number of pixels), it filters the structures of the binary image and only keep those whose area is
    higher than the threshold.
    
    By default input__binary__imag_e must be a binary mask. It is however possible to alternatively provide a label image.
    If a label image is provided for input__binary__imag_e, the parameter input_is_label_image must be set to True.

    NOTE: the option of providing input__binary__imag_e as a label image hasn't been properly tested.
    """
    #Copy the input image
    input__binary__imag_e_copy = input__binary__imag_e.copy()

    # label image regions if input image is not already a label image
    if input_is_label_image:
        label__im_g = input__binary__imag_e.copy()
    else:
        label__im_g = label(input__binary__imag_e_copy)
    
    #measure the properties of the region
    label__im_g_properties = regionprops(label__im_g)

    #Initialize a zero array to be modified as output array
    out_put_arr_ay = np.zeros((input__binary__imag_e_copy.shape[0], input__binary__imag_e_copy.shape[1]))

    #Initialize a collection list for the areas
    areas_cl = []
    
    #Iterate through the regions of the labelled image, identified using measure.regionprops
    for re_gi_on in label__im_g_properties:
        
        #Get the area of the region
        re_gion_area = re_gi_on.area

        #Add area to collection list
        areas_cl.append(re_gion_area)

        #If the region area is higher than the highpass area threshold, modify the output array
        if re_gion_area >= area_highpass_thr:
    
            #Get region coordinates
            re_gi_on_coordinates = re_gi_on.coords
    
            #Unzip the coordinates in individual lists
            unzipped_re_gi_on_coordinates = [list(t) for t in zip(*re_gi_on_coordinates)]
            
            #Set output array values at region coordinates to 255
            out_put_arr_ay[unzipped_re_gi_on_coordinates[0], unzipped_re_gi_on_coordinates[1]] = 1
    
    #Return output array and area list if return_area_list is selected, else only return the output array
    if return_area_list:
        return out_put_arr_ay, areas_cl
    else:
        return out_put_arr_ay



def filter_mask1_on_mask2(mask_1, mask_2, pixels_highpass_threshold=1):
    """
    Given a binary mask_1 and second binary mask_2, the function:
    1) iterates through individual regions of mask_1.
    2) keeps the region if at least a number of pixels higher than pixels_highpass_threshold (default 1) overlaps with pixels in mask_2
    
    For both mask_1 and mask_2 positive pixels are assumed to be the pixels of interest

    The output is a binary mask of values 0, 255 and dtype uint8.

    """
    #Copy mask_1 and mask_2
    mask_1_copy = mask_1.copy()
    mask_2_copy = mask_2.copy()

    #Label regions in mask_1
    label_mask_1 = label(mask_1_copy)
    
    #measure the properties of the regions in mask_1
    regionprops_mask_1 = regionprops(label_mask_1)

    #Get coordinates of mask_2 positive pixels - reorganize them to be a list of tuples
    coord_mask_2 = np.argwhere(mask_2_copy>0)
    coord_mask_2_as_list_of_tuples = [(cr_dnt[0], cr_dnt[1]) for cr_dnt in list(coord_mask_2)]

    #Initialize a zero array to be modified as output array
    output_array_filtered_img = np.zeros((mask_1_copy.shape[0], mask_1_copy.shape[1])).astype(np.uint8)

    #Iterate through the regions of mask_1, identified using regionprops
    for m1_reg_i_on in regionprops_mask_1:
    
        #Get region coordinates - reorganize them to be a list of tuples
        m1_reg_i_on_coordinates = m1_reg_i_on.coords
        m1_reg_i_on_coordinates_as_listoftupl = [(cr_dnt1[0], cr_dnt1[1]) for cr_dnt1 in list(m1_reg_i_on_coordinates)]
    
        #Get intersection of region coordinates and mask_2 positive-pixels coordinates
        m1_reg_i_on_interescion_with_m2 = list(set(m1_reg_i_on_coordinates_as_listoftupl).intersection(set(coord_mask_2_as_list_of_tuples)))

        #If there is an intersection, add the region to the output array
        if len(m1_reg_i_on_interescion_with_m2)>0:
            #Unzip the coordinates in individual lists
            unzipped_m1_reg_i_on_coordinates = [list(tt33) for tt33 in zip(*m1_reg_i_on_coordinates)]
        
            #Set output array values at region coordinates to 255
            output_array_filtered_img[unzipped_m1_reg_i_on_coordinates[0], unzipped_m1_reg_i_on_coordinates[1]] = 255

    return output_array_filtered_img


