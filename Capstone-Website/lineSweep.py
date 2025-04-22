# lineSweep.py
from PIL import Image
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

def process_image(img):
    """
    Process image through line sweep algorithm using in-memory image.
    
    Args:
        img: PIL Image object
        
    Returns:
        PIL Image: Processed image with signature isolated
    """
    logger.info("Processing image through line sweep algo...")
    
    # convert PIL image to numpy array
    temp = np.array(img)
    
    # convert to grayscale
    grayscale = img.convert("L")
    _, thresh = cv2.threshold(
        np.array(grayscale), 128, 255, cv2.THRESH_BINARY_INV
    )

    rows = thresh.shape[0]
    cols = thresh.shape[1]

    # find starting and ending X coordinates
    flagx = 0
    indexStartX = 0
    indexEndX = 0

    for i in range(rows):
        line = thresh[i, :]

        if flagx == 0:
            ele = [255]
            mask = np.isin(ele, line)

            if True in mask:
                indexStartX = i
                flagx = 1

        elif flagx == 1:
            ele = [255]
            mask = np.isin(ele, line)

            if True in mask:
                indexEndX = i
            elif indexStartX + 5 > indexEndX:
                indexStartX = 0
                flagx = 0
            else:
                break

    # find starting and ending Y coordinates
    flagy = 0
    indexStartY = 0
    indexEndY = 0

    for i in range(cols):
        line = thresh[:, i]

        if flagy == 0:
            ele = [255]
            mask = np.isin(ele, line)

            if True in mask:
                indexStartY = i
                flagy = 1

        elif flagy == 1:
            ele = [255]
            mask = np.isin(ele, line)

            if True in mask:
                indexEndY = i
            elif indexStartY + 5 > indexEndY:
                indexStartY = 0
                flagy = 0
            else:
                break

    # draw boundary lines for visualization
    cv2.line(
        thresh,
        (indexStartY, indexStartX),
        (indexEndY, indexStartX),
        (255, 0, 0),
        1,
    )

    cv2.line(
        thresh,
        (indexStartY, indexEndX),
        (indexEndY, indexEndX),
        (255, 0, 0),
        1,
    )

    cv2.line(
        thresh,
        (indexStartY, indexStartX),
        (indexStartY, indexEndX),
        (255, 0, 0),
        1,
    )
    cv2.line(
        thresh,
        (indexEndY, indexStartX),
        (indexEndY, indexEndX),
        (255, 0, 0),
        1,
    )
    
    # crop to the detected signature area
    temp_np = temp[
        indexStartX : indexEndX + 1, indexStartY : indexEndY + 1
    ]
    
    # convert numpy array back to PIL image
    if temp_np.size == 0:
        logger.warning("LineSweep resulted in empty image, returning original")
        return img
        
    processed_img = Image.fromarray(temp_np)
    logger.info("LineSweep processing complete.")
    
    return processed_img

