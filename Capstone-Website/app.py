from flask import Flask, request, jsonify, send_from_directory
import pickle
import os
import pytesseract
import preproc
import features
import imagehash
import numpy as np
from PIL import Image
import io
import base64
from werkzeug.utils import secure_filename
from flask_cors import CORS
import lineSweep
import svm
import ocr
import atexit
import shutil
import logging


app = Flask(__name__)

CORS(app)

logger = logging.getLogger(__name__)

# configure tesseract path
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# store the last processed image for use in process_ocr
# last_processed_image = None

# loading svm
try:
    model = pickle.load(open("../Code_Directory/Verification_Phase/SVM/model.pkl", "rb"))
    scaler = pickle.load(open("../Code_Directory/Verification_Phase/SVM/scaler.pkl", "rb"))
    logger.info("model and scaler loaded.")
except FileNotFoundError:
    logger.error("model or scaler file not found.")
    model, scaler = None, None


def allowed_file(filename):
    """check if the file type is allowed."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features_from_image(img):
    """extract features for prediction from in-memory image."""
    try:
        logger.info("extracting features from image")
        preprocessed_image = preproc.preproc_image(img, display=False)
        if preprocessed_image is None:
            logger.info("processed image not found.")
            return None

        features_result = features.get_contour_features(preprocessed_image.copy(), display=False)
        if features_result is None:
            logger.info("features result not found.")
            return None

        logger.info(f"features result: {features_result}")

        # extract features
        aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = features_result
        hash_val = int(str(imagehash.phash(img)), 16)
        ratio = features.Ratio(preprocessed_image.copy())
        centroid_0, centroid_1 = features.Centroid(preprocessed_image.copy())
        eccentricity, solidity = features.EccentricitySolidity(preprocessed_image.copy())
        (skew_0, skew_1), (kurt_0, kurt_1) = features.SkewKurtosis(preprocessed_image.copy())
        
        features_final_result = np.array([[aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area, ratio,
                          centroid_0, centroid_1, eccentricity, solidity, skew_0, skew_1, kurt_0, kurt_1, hash_val]])
        
        logger.info(features_final_result)
        return features_final_result
    except Exception as e:
        logger.error(f"error fetching features: {e}")
        return None


def pil_image_to_base64(img):
    """Convert PIL image to base64 string."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{img_str}"


@app.route('/process_ocr', methods=['POST'])
def process_image():
    """Processes the image through OCR, Line Sweep, and SVM Verification."""
    
    logger.info("Processing OCR and LineSweep...")
    
    global last_processed_image
    
    # get the image either from the request or use the last processed image
    img = None
    
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            img_bytes = file.read()
            img = Image.open(io.BytesIO(img_bytes))
    elif 'image_data' in request.form:
        # get base64 encoded image data
        img_data = request.form['image_data']
        if img_data and img_data.startswith('data:image'):
            img_data = img_data.split(',')[1]
            img = Image.open(io.BytesIO(base64.b64decode(img_data)))
    
    # if no image in request, use the last processed image
    if img is None:
        if last_processed_image is None:
            return jsonify({"error": "No image provided and no previously processed image available"}), 400
        img = last_processed_image
    
    # process through OCR
    ocr_result, ocr_processed_img = ocr.process_image(img)
    
    # process through LineSweep
    line_sweep_img = lineSweep.process_image(ocr_processed_img)
    
    # process through SVM
    result, accuracy = svm.process_image(line_sweep_img)

    # classify the signature
    final_result = "Genuine Signature" if result == "Genuine" else "Forged Signature"

    # convert all images to base64
    uploaded_image_b64 = pil_image_to_base64(img)
    ocr_processed_image_b64 = pil_image_to_base64(ocr_processed_img)
    line_sweep_image_b64 = pil_image_to_base64(line_sweep_img)

    response = {
        "process_image": "success",
        "uploaded_image": uploaded_image_b64,
        "ocr_processed_image": ocr_processed_image_b64,
        "line_sweep_image": line_sweep_image_b64,
        "final_result": final_result,
        "accuracy": f"{accuracy:.2f}%"
    }

    logger.info("OCR processing complete")
    return jsonify(response)


@app.route('/predict', methods=['POST'])
def predict():
    """handles image upload and starts prediction."""
    global last_processed_image
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    if file and allowed_file(file.filename):
        # read the image directly into memory
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # store the image for later use
        last_processed_image = img.copy()

        # extract features
        features_data = extract_features_from_image(img)
        if features_data is None:
            return jsonify({"error": "Feature extraction failed"}), 400

        if model is None or scaler is None:
            return jsonify({"error": "Model is not loaded"}), 500

        # scale features and predict
        try:
            logger.info("scaling features and making prediction...")
            features_scaled = scaler.transform(features_data)
            prediction = model.predict(features_scaled)
            result = "Genuine Signature" if prediction[0] == 1 else "Forged Signature"

            logger.info(f"prediction result: {result}")

            # convert image to base64
            img_base64 = pil_image_to_base64(img)

            response = {
                "uploaded_image": img_base64,
                "prediction": result
            }
            return jsonify(response)
        except Exception as e:
            logger.error(f"prediction error: {e}")
            return jsonify({"error": f"Prediction error: {str(e)}"}), 500
    else:
        return jsonify({"error": "Allowed image types are png, jpg, jpeg, pdf"}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)
