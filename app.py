from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the saved model
model_path = "trained_model5_inceptionv3old.joblib"
loaded_model = joblib.load(model_path)

# Ensure the loaded model is compiled (if necessary)
# loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define cropping dimensions
top_crop = 250  # Pixels from the top
bottom_crop = 100  # Pixels from the bottom

# Define image size
image_size = (299, 299)

# Define the lower and upper bounds for white color (used in mask creation)
lower_white = np.array([50, 50, 50], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)

# Class labels for the predictions
class_labels = [
    'Patient that have History of MI',
    'Myocardial Infarction Patients',
    'Normal Person',
    'Patient that have abnormal heart beats',
    'COVID-19 Patients'
]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        img_file = request.files['image']
        
        # Read the image
        img = cv2.imdecode(np.frombuffer(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Check if the image is loaded successfully
        if img is not None:
            # Crop the image
            cropped_img = img[top_crop:-bottom_crop, :]

            # Apply bilateral filter to remove noise
            img_filtered = cv2.bilateralFilter(cropped_img, d=9, sigmaColor=75, sigmaSpace=75)

            # Create a mask to identify white areas in the image
            mask = cv2.inRange(img_filtered, lower_white, upper_white)

            # Morphological operation to remove small white points
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

            # Invert the mask
            mask = cv2.bitwise_not(mask)

            # Create a white background image
            bk = np.full(cropped_img.shape, 255, dtype=np.uint8)

            # Apply the mask to the foreground image
            fg_masked = cv2.bitwise_and(cropped_img, cropped_img, mask=mask)

            # Invert the mask again
            mask = cv2.bitwise_not(mask)

            # Apply the mask to the background image
            bk_masked = cv2.bitwise_and(bk, bk, mask=mask)

            # Combine the foreground and background images
            final = cv2.bitwise_or(fg_masked, bk_masked)

            # Resize the preprocessed image to match the model input size
            resized_photo = cv2.resize(final, image_size)

            # Normalize pixel values
            normalized_photo = resized_photo / 255.0

            # Expand dimensions to match the model input shape
            expanded_photo = np.expand_dims(normalized_photo, axis=0)

            # Make prediction
            prediction = loaded_model.predict(expanded_photo)

            # Get the predicted category
            predicted_category = class_labels[np.argmax(prediction)]

            return jsonify({"predicted_category": predicted_category})
        else:
            return jsonify({"error": "Failed to load the image."})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/')
def index():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(debug=True)
