import os
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import joblib
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
import tensorflow as tf

keras=tf.keras
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

# Set up the upload folder and allowed extensions
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

UPLOAD_FOLDER = 'Facial Expression Detection Project/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained models
sgd_model = joblib.load('E:\Machine Learning\Image Classification\Facial Expression Detection Project\models\hog_model.pkl')
cnn_model = load_model('E:\Machine Learning\Image Classification\Facial Expression Detection Project\models\model.keras')

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocessing function for HOG features
def is_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        b, g, r = cv2.split(image)
        return np.array_equal(b, g) and np.array_equal(b, r)
    return True

def preprocess_image(image_path):
    demo_image=cv2.imread(image_path)
    if not is_grayscale(demo_image):
        demo_image = cv2.cvtColor(demo_image, cv2.COLOR_BGR2GRAY)
    demo_image=cv2.resize(demo_image,(180,180))
    demo_image=demo_image / 255.0
    return demo_image

def extract_hog_features(image): 
    # Convert to grayscale only if it has 3 channels
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = rgb2gray(image)  # Ensure the image is grayscale

    features = hog(image, pixels_per_cell=(2, 2), 
                   cells_per_block=(2, 2), orientations=8)

    # Reshape the features into a 2D array with one sample and multiple features
    features = features.reshape(1, -1)  # Ensure this is a 2D array with shape (1, n_features)
    return features

# Preprocessing function for CNN model
def preprocess_cnn(image_path):
    img = image.load_img(image_path, target_size=(250, 250))
    
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = img_array / 255.0 # Normalize the image to [0, 1]
    
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    image_filename = None
    if request.method == 'POST':
        # Handle file upload
        if 'image' not in request.files:
            return 'No file part'
        
        file = request.files['image']
        
        if file.filename == '':
            return 'No selected file'

        if file and allowed_file(file.filename):
            # Save the uploaded file to the static folder
            image_filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
            file.save(file_path)

            class_names=["Happy","Sad"]
            # Predict using both models
            # For SGD Classifier (HOG)
            image = preprocess_image(file_path)
            hog_features=extract_hog_features(image)
            sgd_prediction = sgd_model.predict(hog_features)
            sgd_prediction = "Happy" if sgd_prediction==0 else "Sad"

            # For CNN Classifier
            preprocessed_image = preprocess_cnn(file_path)

            predictions = cnn_model.predict(preprocessed_image)

            # Get the index of the predicted class (class with the highest probability)
            predicted_class_index = np.argmax(predictions, axis=1)
            # Map the index to the actual class name
            cnn_prediction = class_names[predicted_class_index[0]]

            result = [sgd_prediction, cnn_prediction]

    return render_template('index.html', result=result, image_filename=image_filename)

if __name__ == '__main__':
    app.run(debug=True)
