import logging
import numpy as np
import cv2
from PIL import Image
import pytesseract
import io
import base64

logger = logging.getLogger(__name__)

# configure tesseract path
pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

def process_image(img):
    """
    Process image through OCR algorithm using in-memory image.
    
    Args:
        img: PIL Image object
        
    Returns:
        tuple: (ifsc_code, processed_image)
    """
    logger.info("Processing image through OCR algo...")
    
    # convert PIL Image to OpenCV format
    img_np = np.array(img.convert('RGB'))
    img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    h, w, _ = img_cv.shape  # assumes color image

    # convert to HSV and apply mask
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    lower = np.array([103, 79, 60])
    upper = np.array([129, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    
    # process small contours
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 10:
            cv2.drawContours(mask, [c], -1, (0, 0, 0), -1)

    mask = 255 - mask
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # run OCR on the image
    data = pytesseract.image_to_data(img)

    pleaseCd = [0, 0, 0, 0]
    aboveCd = [0, 0, 0, 0]
    result = ""

    for d in data.splitlines():
        d = d.split("\t")
        
        if len(d) == 12:
            # check for IFSC code
            if(len(d[11]) == 11):
                s = d[11][:4]
                temp = d[11]
                if(s == "SYNB" or s == "SBIN" or s == "HDFC" or s == "CNRB" or s == "HDFC" or s == "PUNB" or
                        s == "UTIB" or s == "ICIC"):
                    result = d[11]
                    logger.info(f"IFSC CODE: {d[11]}")

                if(s == "1C1C"):
                    str1 = temp
                    list1 = list(str1)
                    list1[0] = 'I'
                    list1[2] = 'I'
                    str1 = ''.join(list1)
                    logger.info(f"IFSC CODE: {str1}")
                    result = str1
                    
            # extract coordinates for "please" and "above" text
            if d[11].lower() == "please":
                pleaseCd[0] = int(d[6])
                pleaseCd[1] = int(d[7])
                pleaseCd[2] = int(d[8])
                pleaseCd[3] = int(d[9])
                
            if d[11].lower() == "above":
                aboveCd[0] = int(d[6])
                aboveCd[1] = int(d[7])
                aboveCd[2] = int(d[8])
                aboveCd[3] = int(d[9])

    # calculate signature bounding box
    if pleaseCd[0] > 0 and aboveCd[0] > 0:
        lengthSign = aboveCd[0] + aboveCd[3] - pleaseCd[0]
        scaleY = 2
        scaleXL = 2.5
        scaleXR = 0.5

        lengthSignCd = [0, 0, 0, 0]
        lengthSignCd[0] = int(pleaseCd[0] - lengthSign * 2.5)
        lengthSignCd[1] = int(pleaseCd[1] - lengthSign * 2)

        # draw rectangle around signature area
        cv2.rectangle(
            img_cv,
            (lengthSignCd[0], lengthSignCd[1]),
            (
                lengthSignCd[0] + int((scaleXL + scaleXR + 1) * lengthSign),
                lengthSignCd[1] + int(scaleY * lengthSign),
            ),
            (255, 255, 255),
            2,
        )
        
        # crop the signature area
        cropImg = img_cv[
            lengthSignCd[1]: lengthSignCd[1] + int(scaleY * lengthSign),
            lengthSignCd[0]: lengthSignCd[0] + int((scaleXL + scaleXR + 1) * lengthSign),
        ]
        
        # convert cropped image back to PIL format
        if cropImg.size != 0:
            cropImg_rgb = cv2.cvtColor(cropImg, cv2.COLOR_BGR2RGB)
            processed_img = Image.fromarray(cropImg_rgb)
            logger.info("OCR processing complete.")
            return result, processed_img
    
    # if processing fails or bounding box not found, return original image
    logger.info("OCR processing failed or no signature area detected.")
    return result, img
