from flask import Flask, request, jsonify, send_from_directory
import pickle
import os
import pytesseract
import preproc
import features
import imagehash
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename
from flask_cors import CORS
import lineSweep
import svm
import ocr

# Initialize Flask App
app = Flask(__name__)
CORS(app)

# Define directories
UPLOAD_FOLDER = 'static/uploads'
OCR_RESULTS_FOLDER = 'static/OCR_Results'
LINESWEEP_RESULTS_FOLDER = 'static/LineSweep_Results'

app.config['SECRET_KEY'] = 'nishantdalal'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OCR_RESULTS_FOLDER, exist_ok=True)
os.makedirs(LINESWEEP_RESULTS_FOLDER, exist_ok=True)

# Configure Tesseract Path (Modify as per system)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Load SVM Model
try:
    model = pickle.load(open("../Code_Directory/Verification_Phase/SVM/model.pkl", "rb"))
    scaler = pickle.load(open("../Code_Directory/Verification_Phase/SVM/scaler.pkl", "rb"))
    print("Model and Scaler Loaded Successfully.")
except FileNotFoundError:
    print("Error: Model or Scaler file not found.")
    model, scaler = None, None

def allowed_file(filename):
    """Check if the file type is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    """Extract features for prediction."""
    try:
        print(f"🔍 Extracting features from: {image_path}")
        preprocessed_image = preproc.preproc(image_path, display=False)
        if preprocessed_image is None:
            return None

        features_result = features.get_contour_features(preprocessed_image.copy(), display=False)
        if features_result is None:
            return None

        aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = features_result
        hash_val = int(str(imagehash.phash(Image.open(image_path))), 16)
        ratio = features.Ratio(preprocessed_image.copy())
        centroid_0, centroid_1 = features.Centroid(preprocessed_image.copy())
        eccentricity, solidity = features.EccentricitySolidity(preprocessed_image.copy())
        (skew_0, skew_1), (kurt_0, kurt_1) = features.SkewKurtosis(preprocessed_image.copy())

        return np.array([[aspect_ratio, convex_hull_area / bounding_rect_area, contours_area / bounding_rect_area, ratio,
                          centroid_0, centroid_1, eccentricity, solidity, skew_0, skew_1, kurt_0, kurt_1, hash_val]])
    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        return None
    

@app.route('/process_ocr', methods=['POST'])
def process_image():
    """Processes the image through OCR, Line Sweep, and SVM Verification."""
    print("Processing OCR and LineSweep...")

    ocr.ocr_algo()  # Run OCR
    lineSweep.lineSweep_algo()  # Run LineSweep
    result = svm.svm_algo()  # Run SVM Model

    # Classify the signature
    final_result = "Genuine Signature ✅" if result == "Genuine" else "Forged Signature ❌"

    # Ensure we correctly get the extracted signature (latest OCR result)
    extracted_signature = get_latest_file(OCR_RESULTS_FOLDER)
    
    # Get the latest images for processing
    uploaded_image = get_latest_file(UPLOAD_FOLDER)
    ocr_processed_image = extracted_signature  # Ensure correct extracted signature
    line_sweep_image = get_latest_file(LINESWEEP_RESULTS_FOLDER)

    response = {
        "uploaded_image": uploaded_image,  
        "ocr_processed_image": ocr_processed_image,  # Corrected extracted signature
        "line_sweep_image": line_sweep_image,  
        "final_result": final_result,
        "final_processed_signature": extracted_signature  # Ensure correct final signature
    }

    print(f"🔹 Backend Response: {response}")  # Debugging
    return jsonify(response)



@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and starts prediction."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No image selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        print(f"📂 File saved at: {file_path}")

        # Extract Features
        features_data = extract_features(file_path)
        if features_data is None:
            return jsonify({"error": "Feature extraction failed"}), 400

        if model is None or scaler is None:
            return jsonify({"error": "Model is not loaded"}), 500

        # Scale Features and Predict
        try:
            print("⚙️ Scaling features and making prediction...")
            features_scaled = scaler.transform(features_data)
            prediction = model.predict(features_scaled)
            result = "Genuine Signature ✅" if prediction[0] == 1 else "Forged Signature ❌"

            print(f"📌 Sending response to frontend: {result}")

            response = {
                "uploaded_image": f"http://localhost:5001/static/uploads/{filename}",
                "prediction": result
            }
            print(f"🔹 Backend Response: {response}")  # Debugging
            return jsonify(response)
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return jsonify({"error": f"Prediction error: {str(e)}"}), 500
    else:
        return jsonify({"error": "Allowed image types are png, jpg, jpeg, gif"}), 400


@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(UPLOAD_FOLDER, filename)

def get_latest_file(directory):
    """Get the most recent file from the specified directory."""
    files = [f for f in os.listdir(directory) if not f.startswith('.')]
    if not files:
        return None
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory, f)))
    return f"http://localhost:5001/{directory}/{latest_file}"

if __name__ == '__main__':
    app.run(debug=True, port=5001)















# # original code
# # app.py
# from flask import Flask,render_template, flash, request, redirect, url_for
# import pickle
# import ocr
# import lineSweep
# import svm
# import pytesseract
# from PIL import Image
# from werkzeug.utils import secure_filename
# import urllib.request
# import os

# app = Flask(__name__)
# UPLOAD_FOLDER = 'static/uploads'
# app.config['SECRET_KEY'] = 'nishantdalal'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
# # app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 102

# # Load SVM Model
# model = pickle.load(open('../Code_Directory/Verification_Phase/SVM/model.pkl','rb'))

# # Configure Tesseract Path
# pytesseract.pytesseract.tesseract_cmd="/opt/homebrew/bin/tesseract"

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# @app.route('/')
# def home():
#     return render_template('home.html')


# @app.route('/reload')
# def reload_page():
#     dir1 = 'static/uploads'
#     dir2 = 'static/OCR_Results'
#     dir3 = 'static/LineSweep_Results'
#     for f in os.listdir(dir1):
#         os.remove(os.path.join(dir1, f))

#     for f in os.listdir(dir2):
#         os.remove(os.path.join(dir2, f))

#     for f in os.listdir(dir3):
#         os.remove(os.path.join(dir3, f))
#     return redirect('/')


# @app.route('/process_ocr', methods=['POST'])
# def process_image():
#     res = ocr.ocr_algo()
#     lineSweep.lineSweep_algo()
#     result = svm.svm_algo()

#     flash("Algorithm successfully completed for IFSC Code : " + res)
#     if result == "Genuine":
#         ret = "Genuine Signature"
#         return render_template("home.html", result=ret)
#     else:
#         ret = "Forged Signature"
#         return render_template("home.html", result=ret)

#     return redirect('/')


# @app.route('/predict', methods=['POST'])
# def upload_image():
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)

#     file = request.files['file']
#     print(file)
#     if file.filename == '':
#         flash('No image selected for uploading')
#         return redirect(request.url)
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
#         print('upload_image filename: ' + filename)
#         flash('Image successfully uploaded')
#         return render_template('home.html', filename=filename)
#     else:
#         flash('Allowed image types are - png, jpg, jpeg, gif')
#         return redirect(request.url)

# # @app.route("/about")
# # def about():
# #     return render_template("about.html")


# # @app.route("/upload", methods=['GET', 'POST'])
# # def upload_image():
# #     return render_template('upload.html')

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)
