# preproc.py
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage.filters import threshold_otsu


def rgbgrey(img):
    # convert rgb to grayscale
    greyimg = np.zeros((img.shape[0], img.shape[1]))
    for row in range(len(img)):
        for col in range(len(img[row])):
            greyimg[row][col] = np.average(img[row][col])
    return greyimg


def greybin(img):
    # converts grayscale to binary
    blur_radius = 0.8
    img = ndimage.gaussian_filter(img, blur_radius)  # to remove small components or noise
#     img = ndimage.binary_erosion(img).astype(img.dtype)
    thres = threshold_otsu(img)
    binimg = img > thres
    binimg = np.logical_not(binimg)
    return binimg


def preproc(path, img=None, display=True):
    if img is None:
        img = mpimg.imread(path)
    if display:
        plt.imshow(img)
        plt.show()
    grey = rgbgrey(img)
    if display:
        plt.imshow(grey, cmap=matplotlib.cm.Greys_r)
        plt.show()
    binimg = greybin(grey)
    if display:
        plt.imshow(binimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
    r, c = np.where(binimg == 1)
    # now we will make a bounding box with the boundary as the position of pixels on extreme.
    # thus we will get a cropped image with only the signature part.
    signimg = binimg[r.min(): r.max(), c.min(): c.max()]
    if display:
        plt.imshow(signimg, cmap=matplotlib.cm.Greys_r)
        plt.show()

    signimg = 255 * signimg
    signimg = signimg.astype('uint8')

    return signimg


def preproc_image(img, display=False):
    """
    Process image from PIL Image object instead of file path
    
    Args:
        img: PIL Image object
        display: whether to display intermediate results
        
    Returns:
        Preprocessed image as numpy array
    """
    # convert PIL image to numpy array
    img_array = np.array(img)
    
    if display:
        plt.imshow(img_array)
        plt.show()
        
    grey = rgbgrey(img_array)
    if display:
        plt.imshow(grey, cmap=matplotlib.cm.Greys_r)
        plt.show()
        
    binimg = greybin(grey)
    if display:
        plt.imshow(binimg, cmap=matplotlib.cm.Greys_r)
        plt.show()
        
    r, c = np.where(binimg == 1)
    
    # handle empty image case
    if len(r) == 0 or len(c) == 0:
        return None
        
    # crop to signature part
    signimg = binimg[r.min(): r.max(), c.min(): c.max()]
    if display:
        plt.imshow(signimg, cmap=matplotlib.cm.Greys_r)
        plt.show()

    signimg = 255 * signimg
    signimg = signimg.astype('uint8')

    return signimg

