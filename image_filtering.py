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
