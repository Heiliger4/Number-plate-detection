import numpy as np
import cv2
import tensorflow as tf
import pytesseract as pt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects
from keras.metrics import MeanSquaredError

# Custom Layer Definition
class CustomScaleLayer(tf.keras.layers.Layer):
    def __init__(self, scale, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.scale = scale

    def call(self, inputs):
        if isinstance(inputs, list):
            combined_input = tf.add_n(inputs)
            return combined_input * self.scale
        return inputs * self.scale

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            return input_shape[0]
        return input_shape

# Register the custom layer and the custom metric
get_custom_objects().update({
    'CustomScaleLayer': CustomScaleLayer,
    'mse': MeanSquaredError
})

# Load the model once when the script starts
model = tf.keras.models.load_model('./static/models/object_detection.h5', custom_objects={'CustomScaleLayer': CustomScaleLayer, 'mse': MeanSquaredError})

def object_detection(path, filename):
    # Read image
    image = load_img(path)  # PIL object
    image = np.array(image, dtype=np.uint8)  # 8-bit array (0, 255)
    image1 = load_img(path, target_size=(224, 224))
    
    # Data preprocessing
    image_arr_224 = img_to_array(image1) / 255.0  # Normalize
    h, w, d = image.shape
    test_arr = image_arr_224.reshape(1, 224, 224, 3)
    
    # Make predictions
    coords = model.predict(test_arr, verbose=0)  # Suppress prediction logs
    
    # Denormalize the values
    denorm = np.array([w, w, h, h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    
    # Draw bounding box on the image
    xmin, ymin, xmax, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    print(f"Bounding box coordinates: {pt1}, {pt2}")
    
    # Validate bounding box coordinates
    if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0 or xmax > w or ymax > h:
        print("Invalid bounding box coordinates. Skipping drawing.")
    else:
        cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    
    # Convert to BGR and save the image
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./static/predict/{filename}', image_bgr)
    
    return coords

def OCR(path, filename):
    img = np.array(load_img(path))
    cods = object_detection(path, filename)
    xmin, ymin, xmax, ymax = cods[0]
    
    # Validate ROI coordinates
    if xmin >= xmax or ymin >= ymax or xmin < 0 or ymin < 0 or xmax > img.shape[1] or ymax > img.shape[0]:
        print("Invalid ROI coordinates. Skipping OCR.")
        return "Invalid ROI coordinates"
    
    # Extract ROI
    roi = img[ymin:ymax, xmin:xmax]
    
    # Check if ROI is empty
    if roi.size == 0:
        print("ROI is empty. Skipping OCR.")
        return "ROI is empty"
    
    # Convert ROI to BGR and save
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./static/roi/{filename}', roi_bgr)
    
    # Perform OCR
    text = pt.image_to_string(roi)
    print(f"OCR Result: {text}")
    return text

# Example usage
# object_detection('path_to_image.jpg', 'output_filename.jpg')
# OCR('path_to_image.jpg', 'output_filename.jpg')