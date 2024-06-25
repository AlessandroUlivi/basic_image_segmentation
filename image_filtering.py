import numpy as np
from scipy.signal import convolve
from skimage.filters import median as medianfilter
from skimage.filters import frangi
from skimage.util import img_as_float32
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

