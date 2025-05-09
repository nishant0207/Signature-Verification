import { useState, useEffect } from "react";
import axios from "axios";

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [processingImages, setProcessingImages] = useState([]);
  const [finalImage, setFinalImage] = useState(null);
  const [outputText, setOutputText] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [previewImage, setPreviewImage] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [accuracy, setAccuracy] = useState(null);

  const getCacheBuster = () => new Date().getTime();

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (file) {
      setIsProcessing(true);
      setProcessingImages([]);
      setFinalImage(null);
      setOutputText("");
      setPreviewImage(null);
      setErrorMessage("");
      setAccuracy(null);

      const formData = new FormData();
      formData.append("file", file);

      try {
        console.log("Uploading image...");
        const response = await axios.post("http://localhost:5001/predict", formData);
        console.log("Upload Response:", response.data);

        if (response.data.uploaded_image) {
          setSelectedImage(response.data.uploaded_image);
          setPreviewImage(response.data.uploaded_image);
        }

        await processOCR();
      } catch (error) {
        console.error("Error uploading image:", error);
        setErrorMessage("Signature verification failed. Please try again.");
      } finally {
        setIsProcessing(false);
      }
    }
  };

  // backend processing API
  const processOCR = async () => {
    try {
      console.log("Processing OCR...");
      const response = await axios.post("http://localhost:5001/process_ocr");

      console.log("OCR Process Response:", response.data);

      if (response.data) {
        // Set the processing images directly from base64 data
        const newProcessingImages = [
          response.data.ocr_processed_image || null,
          response.data.line_sweep_image || null,
        ].filter(Boolean);
        
        setProcessingImages(newProcessingImages);
        
        if (newProcessingImages.length > 0) {
          setPreviewImage(newProcessingImages[0]);
        }

        setOutputText(response.data.final_result);
        setAccuracy(response.data.accuracy);
        setIsProcessing(false);
      }
    } catch (error) {
      console.error("Error processing image:", error);
      setErrorMessage("Failed to process image. Please try again.");
      setIsProcessing(false);
    }
  };

  // reset state for new upload
  const handleNewUpload = () => {
    setSelectedImage(null);
    setProcessingImages([]);
    setFinalImage(null);
    setOutputText("");
    setIsProcessing(false);
    setPreviewImage(null);
    setErrorMessage("");
    setAccuracy(null);
  };

  return (
    <div className="min-h-screen min-w-screen flex flex-row bg-gray-700 text-white">
      {/* left Panel - controls & processing stages */}
      <div className="w-1/2 p-5 flex flex-col items-center">
        <h1 className="text-3xl font-bold mb-5">Signature Verification System</h1>

        {/* file upload */}
        {!selectedImage && (
          <div className="mb-5">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageUpload}
              className="p-2 border rounded-md bg-white text-black"
            />
          </div>
        )}

        {/* error message */}
        {errorMessage && (
          <p className="mt-3 p-2 bg-red-500 text-white shadow-md rounded-md text-center">
            {errorMessage}
          </p>
        )}

        {/* image preview*/}
        {selectedImage && (
          <div className="mb-5">
            <h2 className="text-lg font-semibold">Uploaded Image:</h2>
            <img
              src={selectedImage}
              alt="Uploaded Preview"
              className="w-64 h-auto border rounded-md shadow-md cursor-pointer"
              onClick={() => setPreviewImage(selectedImage)}
              onError={(e) => {
                console.error("Error loading uploaded image:", selectedImage);
                e.target.src = "fallback_image.png"; // Provide a fallback image
              }}
            />
          </div>
        )}

        {/* processing stages */}
        {processingImages.length > 0 && (
          <div className="mb-5">
            <h2 className="text-lg font-semibold">Processing Stages:</h2>
            <div className="flex space-x-4">
              {processingImages.map((img, index) => (
                img && (
                  <img
                    key={`${index}`}
                    src={img}
                    alt={`Processing Stage ${index + 1}`}
                    className="w-40 h-auto border rounded-md shadow-md cursor-pointer"
                    onClick={() => setPreviewImage(img)}
                    onError={(e) => {
                      console.error("Error loading processing image:", img);
                      e.target.src = "fallback_image.png";
                    }}
                  />
                )
              ))}
            </div>
          </div>
        )}

        {/* processing indicator */}
        {isProcessing && (
          <p className="mt-3 p-2 bg-yellow-400 text-black shadow-md rounded-md text-center">
            ⏳ Processing...
          </p>
        )}

        {/* final prediction */}
        {outputText && (
          <div className="mt-5">
            <h2 className="text-lg font-semibold">Final Prediction:</h2>
            <p className="mt-3 p-2 bg-white shadow-md rounded-md text-center text-black font-bold">
              {outputText}
            </p>

            {/* accuracy display */}
            {accuracy && (
              <p className="mt-2 p-2 bg-green-500 text-white shadow-md rounded-md text-center font-bold">
                🎯 Accuracy: {accuracy}
              </p>
            )}
          </div>
        )}

        {/* final output */}
        {finalImage && (
          <div className="mt-5">
            <h2 className="text-lg font-semibold">Extracted Signature:</h2>
            <img
              src={finalImage}
              alt="Final Processed Signature"
              className="w-64 h-auto border rounded-md shadow-md cursor-pointer"
              onClick={() => setPreviewImage(finalImage)}
              onError={(e) => {
                console.error("Error loading final image:", finalImage);
                e.target.src = "fallback_image.png"; // Provide a fallback image
              }}
            />
            <p className="mt-3 p-2 bg-white shadow-md rounded-md text-center text-black font-bold">
              {outputText}
            </p>
          </div>
        )}

        {/* new upload button */}
        {selectedImage && (
          <button
            onClick={handleNewUpload}
            className="mt-5 px-4 py-2 bg-blue-500 hover:bg-blue-700 text-white font-bold rounded-md"
          >
            Upload New Image
          </button>
        )}
      </div>

      {/* right panel */}
      <div className="w-1/2 p-5 flex items-center justify-center bg-black">
        {previewImage ? (
          <img
            src={previewImage}
            alt="Large Preview"
            className="w-full h-auto border rounded-md shadow-md"
            onError={(e) => {
              console.error("Error loading preview image:", previewImage);
              e.target.src = "fallback_image.png"; // Provide a fallback image
            }}
          />
        ) : (
          <p className="text-gray-400">Click an image to preview it here.</p>
        )}
      </div>
    </div>
  );
}

export default App;