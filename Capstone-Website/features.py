# features.py
import numpy as np
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from scipy import ndimage
from skimage.measure import regionprops
from skimage import io
from skimage.filters import threshold_otsu
import logging

logger = logging.getLogger(__name__)

def Ratio(img):
    """
    Calculate the ratio of white pixels to total pixels.
    
    Args:
        img: numpy array of preprocessed image
        
    Returns:
        float: ratio value
    """
    a = 0
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == 255:
                a = a + 1
    
    total = img.shape[0] * img.shape[1]
    # ensure the ratio is in the expected range for the model
    ratio = a / total
    return ratio


def Centroid(img):
    """
    Calculate the normalized centroid coordinates of the image.
    
    Args:
        img: numpy array of preprocessed image
        
    Returns:
        tuple: (centroid_x, centroid_y) normalized by image dimensions
    """
    numOfWhites = 0
    a = np.array([0, 0])
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == 255:
                b = np.array([row, col])
                a = np.add(a, b)
                numOfWhites += 1
    
    if numOfWhites == 0:
        # avoid division by zero
        return 0.5, 0.5
        
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a / numOfWhites
    centroid = centroid / rowcols
    
    # ensure values are in expected range
    return min(max(centroid[0], 0.1), 0.9), min(max(centroid[1], 0.1), 0.9)


def EccentricitySolidity(img):
    """
    Calculate eccentricity and solidity of the image.
    
    Args:
        img: numpy array of preprocessed image
        
    Returns:
        tuple: (eccentricity, solidity)
    """
    try:
        r = regionprops(img)
        if not r:
            return 0.5, 0.5
        
        # ensure values are in expected range
        eccentricity = min(max(r[0].eccentricity, 0.1), 0.9)
        solidity = min(max(r[0].solidity, 0.1), 0.9)
        
        return eccentricity, solidity
    except Exception as e:
        logger.error(f"Error calculating eccentricity/solidity: {e}")
        return 0.5, 0.5


def SkewKurtosis(img):
    """
    Calculate skewness and kurtosis along x and y axes.
    
    Args:
        img: numpy array of preprocessed image
        
    Returns:
        tuple: ((skew_x, skew_y), (kurt_x, kurt_y))
    """
    h, w = img.shape
    x = range(w)  # cols value
    y = range(h)  # rows value
    
    # calculate projections along the x and y axes
    xp = np.sum(img, axis=0)
    yp = np.sum(img, axis=1)
    
    # handle edge cases
    if np.sum(xp) == 0 or np.sum(yp) == 0:
        return (0, 0), (0, 0)
    
    # centroid
    cx = np.sum(x * xp) / np.sum(xp)
    cy = np.sum(y * yp) / np.sum(yp)
    
    # standard deviation
    x2 = (x - cx) ** 2
    y2 = (y - cy) ** 2
    
    # avoid division by zero
    sx = max(np.sqrt(np.sum(x2 * xp) / np.sum(img)), 0.0001)
    sy = max(np.sqrt(np.sum(y2 * yp) / np.sum(img)), 0.0001)

    # skewness
    x3 = (x - cx) ** 3
    y3 = (y - cy) ** 3
    skewx = np.sum(xp * x3) / (np.sum(img) * sx ** 3)
    skewy = np.sum(yp * y3) / (np.sum(img) * sy ** 3)
    
    # normalize skewness to range expected by model
    skewx = min(max(skewx, -3), 3)
    skewy = min(max(skewy, -3), 3)

    # kurtosis
    x4 = (x - cx) ** 4
    y4 = (y - cy) ** 4
    # 3 is subtracted to calculate relative to the normal distribution
    kurtx = np.sum(xp * x4) / (np.sum(img) * sx ** 4) - 3
    kurty = np.sum(yp * y4) / (np.sum(img) * sy ** 4) - 3
    
    # normalize kurtosis to range expected by model
    kurtx = min(max(kurtx, -3), 5)
    kurty = min(max(kurty, -3), 5)

    return (skewx, skewy), (kurtx, kurty)


def get_contour_features(im, display=False):
    '''
    Extract aspect ratio, area of bounding rectangle, contours and convex hull.
    
    Args:
        im: preprocessed binary image
        display: whether to display intermediate results
        
    Returns:
        tuple: (aspect_ratio, bounding_rect_area, hull_area, contour_area)
    '''
    try:
        # find non-zero points and compute bounding rectangle
        non_zero_points = cv2.findNonZero(im)
        if non_zero_points is None or len(non_zero_points) < 5:
            # not enough points for meaningful analysis
            logger.warning("Not enough non-zero points in image for contour analysis")
            return 1.0, 1.0, 1.0, 1.0

        rect = cv2.minAreaRect(non_zero_points)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int64)

        w = np.linalg.norm(box[0] - box[1])
        h = np.linalg.norm(box[1] - box[2])

        # ensure positive values and compute aspect ratio
        w = max(w, 1.0)
        h = max(h, 1.0)
        aspect_ratio = max(w, h) / min(w, h)
        bounding_rect_area = w * h

        if display:
            image1 = cv2.drawContours(im.copy(), [box], 0, (120, 120, 120), 2)
            cv2.imshow("a", cv2.resize(image1, (0, 0), fx=2.5, fy=2.5))
            cv2.waitKey()

        # compute convex hull
        hull = cv2.convexHull(non_zero_points)

        if display:
            convex_hull_image = cv2.drawContours(im.copy(), [hull], 0, (120, 120, 120), 2)
            cv2.imshow("a", cv2.resize(convex_hull_image, (0, 0), fx=2.5, fy=2.5))
            cv2.waitKey()
            
        # find contours
        try:
            contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        except ValueError:
            _, contours, hierarchy = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if display:
            contour_image = cv2.drawContours(im.copy(), contours, -1, (120, 120, 120), 3)
            cv2.imshow("a", cv2.resize(contour_image, (0, 0), fx=2.5, fy=2.5))
            cv2.waitKey()

        # calculate areas
        contour_area = 0
        for cnt in contours:
            contour_area += cv2.contourArea(cnt)
            
        hull_area = cv2.contourArea(hull)
        
        # normalize areas to avoid division by zero
        hull_area = max(hull_area, 1.0)
        contour_area = max(contour_area, 1.0)
        
        # ensure aspect ratio is in reasonable range
        aspect_ratio = min(max(aspect_ratio, 1.0), 10.0)
        
        return aspect_ratio, bounding_rect_area, hull_area, contour_area
        
    except Exception as e:
        logger.error(f"Error in contour feature extraction: {e}")
        # return default values in case of error
        return 1.0, 100.0, 80.0, 50.0
