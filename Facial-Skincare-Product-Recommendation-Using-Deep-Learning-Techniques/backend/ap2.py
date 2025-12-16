import os
from typing import List
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import cv2


from keras.models import load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from models.skin_tone.skin_tone_knn import identify_skin_tone
from flask import Flask, request, render_template
from flask_restful import Api, Resource, reqparse, abort
import werkzeug
from models.recommender.rec import recs_essentials, makeup_recommendation
import base64
from io import BytesIO
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
import time

from tensorflow.python.keras.engine.sequential import Sequential

from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

image_data_generator = tf.keras.preprocessing.image.ImageDataGenerator()

# app = Flask(__name__)


app = Flask(__name__, static_url_path='/static')
api = Api(app)

class_names1 = ['Dry_skin', 'Normal_skin', 'Oil_skin']
class_names2 = ['Low', 'Moderate', 'Severe']
skin_tone_dataset = 'backend/models/skin_tone/skin_tone_dataset.csv'


def get_model():
    global model1, model2
    model1 = load_model('backend/models/skin_model')
    print('Model 1 loaded')
    model2 = load_model('backend/models/acne_model')
    print("Model 2 loaded!")


def load_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    # (height, width, channels)
    img_tensor = image.img_to_array(img)
    # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    # imshow expects values in the range [0, 1]
    img_tensor /= 255.
    return img_tensor


def prediction_skin(img_path):
    new_image = load_image(img_path)
    pred1 = model1.predict(new_image)
    print(pred1)
    if len(pred1[0]) > 1:
        pred_class1 = class_names1[tf.argmax(pred1[0])]
    else:
        pred_class1 = class_names1[int(tf.round(pred1[0]))]
    return pred_class1


def prediction_acne(img_path):
    new_image = load_image(img_path)
    pred2 = model2.predict(new_image)
    print(pred2)
    if len(pred2[0]) > 1:
        pred_class2 = class_names2[tf.argmax(pred2[0])]
    else:
        pred_class2 = class_names2[int(tf.round(pred2[0]))]
    return pred_class2


get_model()


img_put_args = reqparse.RequestParser()
img_put_args.add_argument(
    "file", help="Please provide a valid image file", required=True)


rec_args = reqparse.RequestParser()

rec_args.add_argument(
    "tone", type=int, help="Argument required", required=True)
rec_args.add_argument(
    "type", type=str, help="Argument required", required=True)
rec_args.add_argument("features", type=dict,
                      help="Argument required", required=True)


class Recommendation(Resource):
    def put(self):
        args = rec_args.parse_args()
        print(args)
        features = args['features']
        tone = args['tone']
        skin_type = args['type'].lower()
        skin_tone = 'light to medium'
        if tone <= 2:
            skin_tone = 'fair to light'
        elif tone >= 4:
            skin_tone = 'medium to dark'
        print(f"{skin_tone}, {skin_type}")
        fv = []
        for key, value in features.items():
            # if key == 'skin type':
            #     skin_type = key
            # elif key == 'skin tone':
            #     skin_tone = key
            #     continue
            fv.append(int(value))

        general = recs_essentials(fv, None)

        makeup = makeup_recommendation(skin_tone, skin_type)
        return {'general': general, 'makeup': makeup}


class SkinMetrics(Resource):
    def put(self):
        args = img_put_args.parse_args()
        print()
        print(args)
        print()
        file = args['file']
        starter = file.find(',')
        image_data = file[starter+1:]
        print()
        print(image_data)
        print()
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data+b'==')))

        filename = 'image.png'
        file_path = os.path.join('backend/static', filename)
        im.save(file_path)
        skin_type = prediction_skin(file_path).split('_')[0]
        acne_type = prediction_acne(file_path)
        tone = identify_skin_tone(file_path, dataset=skin_tone_dataset)
        print(skin_type)
        print(acne_type)
        print(tone)

        return {'type': skin_type, 'tone': str(tone), 'acne': acne_type}, 200


api.add_resource(SkinMetrics, "/upload")
api.add_resource(Recommendation, "/recommend")


import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def predict_img(image_path):
    # Load an example image
    original_image = cv2.imread(image_path)
    original_image_bgr = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Convert the image to YCrCb color space
    ycrcb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

    # Apply initial segmentation (example: simple skin color segmentation in HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    initial_segmentation_mask = cv2.inRange(hsv_image, lower_skin, upper_skin)

    # Pseudo ground truth
    lower_acne_color = np.array([0, 20, 70], dtype=np.uint8)
    upper_acne_color = np.array([20, 255, 255], dtype=np.uint8)
    pseudo_ground_truth_mask = cv2.inRange(hsv_image, lower_acne_color, upper_acne_color)

    # Apply additional processing for final segmentation if needed
    # For demonstration, let's use the initial segmentation as the final segmentation
    final_segmentation = initial_segmentation_mask

    # Display the images with different color maps
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    axs[0, 0].imshow(original_image_bgr)
    axs[0, 0].set_title('Original (BGR)')

    axs[0, 1].imshow(gray_image, cmap='gray')
    axs[0, 1].set_title('Grayscale')

    axs[0, 2].imshow(cv2.cvtColor(ycrcb_image, cv2.COLOR_YCrCb2RGB))
    axs[0, 2].set_title('YCrCb')

    axs[0, 3].imshow(cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB))
    axs[0, 3].set_title('HSV')

    axs[1, 0].imshow(initial_segmentation_mask, cmap='viridis')  # Use 'viridis' colormap for segmentation
    axs[1, 0].set_title('Initial Segmentation')

    axs[1, 1].imshow(pseudo_ground_truth_mask, cmap='plasma')  # Use 'plasma' colormap for ground truth
    axs[1, 1].set_title('Pseudo Ground Truth')

    # Display the final segmentation with a custom colormap
    cmap = plt.cm.colors.ListedColormap(['#ffffff', '#ff0000'], name='custom_map', N=2)  # White for non-acne, Red for acne
    axs[1, 2].imshow(cv2.cvtColor(final_segmentation, cv2.COLOR_GRAY2RGB), cmap=cmap)
    axs[1, 2].set_title('Final Segmentation')

    # Display ground truth mask with a custom colormap
    axs[1, 3].imshow(pseudo_ground_truth_mask, cmap=cmap)
    axs[1, 3].set_title('Pseudo Ground Truth')

    # Convert the plot to image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    plt.close()

    return image_base64

######################################################################################################
## for cosine recoomedn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from collections.abc import Mapping
import gensim
from gensim.models import Word2Vec

# Download NLTK resources
nltk.download('punkt')

# Read the CSV file
df = pd.read_csv("C:/Users/SuryaMurugan/Downloads/skincare_products_clean.csv")

# Preprocess text data
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    return tokens

# Tokenize and preprocess product types
df['product_name12'] = df['product_name'].apply(preprocess_text)

# Train Word2Vec model on product types
model = Word2Vec(df['product_name12'], vector_size=100, window=5, min_count=1, workers=4)

# Function to encode text into vectors
def encode_text(text):
    vector = np.zeros(100)  # Assuming embedding size of 100
    count = 0
    for token in text:
        if token in model.wv.key_to_index:
            vector += model.wv[token]
            count += 1
    if count > 0:
        vector /= count
    return vector

# Encode product types into vectors
df['product_type_vector'] = df['product_name12'].apply(encode_text)

# Function to recommend top products based on input skin type and acne severity
def recommend_products(skin_type, acne_severity):
    # Encode input skin type and acne severity
    skin_type_vector = encode_text(skin_type)
    acne_severity_vector = encode_text(acne_severity)

    # Combine skin type and acne severity vectors
    input_vector = (skin_type_vector + acne_severity_vector) / 2

    # Reshape input vectors for compatibility with cosine_similarity function
    input_vector_reshaped = input_vector.reshape(1, -1)
    product_type_vectors_reshaped = np.vstack(df['product_type_vector'])

    # Calculate cosine similarity
    similarities = cosine_similarity(input_vector_reshaped, product_type_vectors_reshaped)

    # Recommend top products based on cosine similarity
    top_indices = np.argsort(similarities[0])[::-1][:10]
    # Fetch top products with additional columns
    top_products_data = df.iloc[top_indices][['product_name', 'product_url', 'price']].copy()
    top_products_data['cosine_similarity'] = similarities[0][top_indices]
    
    # Convert DataFrame to list of dictionaries
    top_products = top_products_data.to_dict(orient='records')
    
    return top_products

# # Example usage
# skin_type_input = 'dry skin'
# acne_severity_input = 'Moderate'
# recommended_products = recommend_products(skin_type_input, acne_severity_input)
# print("Recommended products:")
# for product in recommended_products:
#     print(product)



@app.route("/")
def index():
    return render_template('index.html')

@app.route("/quiz")
def quiz():
    return render_template('quiz.html')


@app.route("/home", methods=['GET', 'POST'])
def home():
    return render_template('home.html')


class DataStore():
    skin_r="hey"
    acne_r="b"

data = DataStore()



@app.route("/predict", methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        filename = file.filename
        file_path = os.path.join('backend/static',filename)                    #slashes should be handeled properly
        file.save(file_path)
        skin_type = prediction_skin(file_path)
        data.skin_r=skin_type
        acne_type = prediction_acne(file_path)
        data.acne_r=acne_type
        print(skin_type)
        print(acne_type)
        # Example usage:
        ## for saving augmented image
        image_path = file_path  # Update with your actual path
        result_image_base64 = predict_img(image_path)
        image_path = 'backend/static/image_aug.png'
        binary_data = base64.b64decode(result_image_base64)
        # Write the binary data to a file
        with open(image_path, 'wb') as f:
            f.write(binary_data)
        
        
        
        #return {'type': skin_type, 'acne': acne_type}, 200 #skin_type, acne_type
        return render_template('prediction.html',skin_type=skin_type, acne_type=acne_type)
    


@app.route("/routine")
def routine():
    # Assuming you have obtained skin_type and acne_type from prediction
    skin_type = data.skin_r #"Dry Skin"  # Example value, replace with actual prediction result
    acne_type = data.acne_r #"Low"       # Example value, replace with actual prediction result

    return render_template('recommendation.html', skin_type=skin_type, acne_type=acne_type)


@app.route('/rec')
def rec():
    skin_type = data.skin_r
    acne_severity = data.acne_r
    recommended_products = recommend_products(skin_type, acne_severity)
    print("Recommended products:")
    for product in recommended_products:
        print(product)
    
    return render_template('cos_rec.html', products=recommended_products)




if __name__ == "__main__":
    app.run(debug=False)


# @app.route("/", methods=['GET', 'POST'])
# def home():
#     if request.method == 'POST' and 'file' in request.files:
#         file = request.files['file']
#         filename = file.filename
#         file_path = f"backend/static/{filename}"  # Adjust as per your file structure
#         file.save(file_path)
#         with open("backend/static/current_file.txt", "w") as f:
#             f.write(filename)
#         # return redirect(url_for('loading'))
#     return render_template('home.html')

# @app.route("/loading")
# def loading():
#     with open("backend/static/current_file.txt", "r") as f:
#         filename = f.read().strip()
#     image_url = url_for('static', filename=filename)

#     return render_template('loading.html', image_url=image_url)

# @app.route("/prediction")
# def prediction():
#     with open("backend/static/current_file.txt", "r") as f:
#         filename = f.read().strip()
#     file_path = f"backend/static/{filename}"
#     skin_type = prediction_skin(file_path)
#     acne_type = prediction_acne(file_path)
#     return render_template('prediction.html', skin_type=skin_type, acne_type=acne_type)





# file_path = 'backend/static/test_image.jpeg'                      #slashes should be handeled properly
# skin_type = prediction_skin(file_path)
# acne_type = prediction_acne(file_path)
# print(skin_type)
# print(acne_type)
# tone = identify_skin_tone(file_path, dataset=skin_tone_dataset)
# print(tone)



