from flask import Flask, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from bson import ObjectId
import io
from mambavision import load_model, inference, transform
import torch
from depthanythingv2 import background_cut
from roiextraction import roicut, tensor_to_image
from util import *
from io import BytesIO
from roiextraction import preprocess
# Load environment variables from .env file
load_dotenv()


# Initialize Flask app
app = Flask(__name__)
# connection_string = os.getenv("MONGO_CONNECTION_STRING")
# MongoDB connection string
# Replace this with your actual MongoDB connection string
# if connection_string:
#     client = MongoClient(connection_string)
#     print(client.list_database_names())
#     db = client["HandIdDB"]
#     print(db.list_collection_names())
# else:
#     raise Exception("MongoDB connection string not found in environment variables")

def encode_base64(images):
    processed_images_base64 = []
    for img in images:
        # Convert the processed image back to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        processed_images_base64.append(img_base64)
    return processed_images_base64


def decode_base64_images_to_grayscale(base64_images):
    """
    Decode a list of base64 images into PIL images and convert to grayscale (single channel).
    @base64_images: List of base64 encoded image strings.
    @returns: List of PIL grayscale images.
    """
    decoded_images = []
    
    for base64_img in base64_images:
        # Decode the base64 image
        img_data = base64.b64decode(base64_img)
        img = Image.open(io.BytesIO(img_data))
        
        # Convert to grayscale (1 channel)
        img = img.convert('RGB')
        
        decoded_images.append(img)
    
    return decoded_images

# Example endpoint to retrieve data from MongoDB
@app.route("/ai/register/backgroundcut", methods=["POST"])
def background_cutt():
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    save_path = os.path.join(current_directory, 'processed_image.png')

    data = request.json
    if 'images' not in data:
        return jsonify({"error": "Images not found in request"}), 400
    base64_images = data['images']

    images = decode_base64_images(base64_images)
    background_cut_images = background_cut(images, 0.7)
    
    # roi_cut_image = roicut(background_cut_image)

    
    # darken_image = preprocess.darken_image(roi_cut_image,0.8)
    # final_image = preprocess.enhance_image(darken_image, 1.5)

    return jsonify({"images": encode_base64(background_cut_images)}), 200

@app.route("/ai/register/roicut", methods=["POST"])
def roi_cut():
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    save_path = os.path.join(current_directory, 'processed_image.png')

    data = request.json
    if 'images' not in data:
        return jsonify({"error": "Images not found in request"}), 400
    base64_images = data['images']

    images = decode_base64_images_to_grayscale(base64_images)
    roi_cut_images = roicut(images)
    
    darken_images = preprocess.darken_pilimages(roi_cut_images,0.8)
    final_image = preprocess.enhance_pilimages(darken_images, 1.5)

    # background_cut_images = background_cut(images, 0.7)
    # for idx, img in enumerate(roi_cut_images):
    #     filename = f"processed_image_{idx + 1}.png"
    #     save_path = os.path.join(current_directory, filename)
    #     img.save(save_path)
 

    return jsonify({"images": encode_base64(final_image)}), 200

def calculate_pairwise_euclidean_distance(features):

    squared_norms = torch.sum(features ** 2, dim=1, keepdim=True)
    distances = squared_norms + squared_norms.t() - 2 * torch.mm(features, features.t())
    distances = torch.sqrt(torch.clamp(distances, min=0.0))
    
    return distances
@app.route("/ai/register/inference", methods=["POST"])
def register():
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    save_path = os.path.join(current_directory, 'processed_image.png')

    data = request.json
    if 'images' not in data:
        return jsonify({"error": "Images not found in request"}), 400
    base64_images = data['images']
    images = decode_base64_images_for_inference(base64_images)
    batch = torch.stack([transform(img) for img in images])
    with torch.no_grad():
        result = inference(batch.to(torch.float32).to("cuda"))
    # euclidean_distances = calculate_pairwise_euclidean_distance(result)
    return jsonify({"feature_vectors": result.cpu().numpy().tolist()}), 200
    # return jsonify({"feature_vectors": euclidean_distances.cpu().numpy().tolist()}), 200

if __name__ == "__main__": 

    app.run(host="0.0.0.0", port=5000, debug=True)
