from flask import Flask, jsonify, request,make_response
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from bson import ObjectId
import io
from elastic_search_palmprint import *
from mambavision import inference, transform
import torch
from depthanythingv2 import background_cut, background_cut_batch
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

def encode_image_to_base64_opencv(img, format="PNG"):
    """
    Encodes a single image to a base64 string using OpenCV.
    
    Parameters:
    - img (PIL.Image.Image): The image to encode.
    - format (str): The format to save the image in (default is PNG).
    
    Returns:
    - str: The base64-encoded string of the image.
    """
    # Convert PIL Image to NumPy array
    img_array = np.array(img)

    # Check the number of dimensions
    if img_array.ndim == 2:
        # Grayscale image: Convert to BGR by stacking
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    elif img_array.ndim == 3:
        if img_array.shape[2] == 3:
            # RGB to BGR
            img_bgr = img_array[:, :, ::-1]
        elif img_array.shape[2] == 4:
            # RGBA to BGR
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            # Unexpected number of channels
            raise ValueError(f"Unsupported number of channels: {img_array.shape[2]}")
    else:
        # Unexpected image format
        raise ValueError(f"Unsupported image shape: {img_array.shape}")

    # Encode image to memory buffer using OpenCV
    encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]  # Adjust compression level as needed
    result, buffer = cv2.imencode(f'.{format.lower()}', img_bgr, encode_param)
    
    if not result:
        raise ValueError("Image encoding failed")

    # Base64 encode the bytes
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return img_base64

def encode_base64(images, format="PNG", max_workers=None):
    """
    Encodes a list of images to base64 strings in parallel using OpenCV.
    
    Parameters:
    - images (list of PIL.Image.Image): The images to encode.
    - format (str): The format to save the images in (default is PNG).
    - max_workers (int, optional): The maximum number of threads to use. Defaults to the number of processors on the machine.
    
    Returns:
    - list of str: The base64-encoded strings of the images.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        processed_images_base64 = list(executor.map(lambda img: encode_image_to_base64_opencv(img, format), images))
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
import time
@app.route("/ai/register/backgroundcut", methods=["POST"])
def background_cutt_batch():
    print("/ai/register/backgroundcut")
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    save_path = os.path.join(current_directory, 'processed_image.png')

    data = request.json
    if 'images' not in data:
        return jsonify({"error": "Images not found in request"}), 400
    base64_images = data['images']
    # print(data)
    images = decode_base64_images(base64_images)
    background_cut_images = background_cut_batch(images, 0.6)
    
    # roi_cut_image = roicut(background_cut_image)

    
    # darken_image = preprocess.darken_image(roi_cut_image,0.8)
    # final_image = preprocess.enhance_image(darken_image, 1.5)
    torch.cuda.empty_cache()

    start_time = time.time()
    result = encode_base64(background_cut_images)
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Blackout took {inference_time:.4f} seconds")
    return jsonify({"images": result}), 200


@app.route("/ai/register/roicut", methods=["POST"])
def roi_cut():
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    save_path = os.path.join(current_directory, 'processed_image.png')
    print("/ai/register/roicut")
    data = request.json
    if 'images' not in data:
        return jsonify({"error": "Images not found in request"}), 400
    base64_images = data['images']

    images = decode_base64_images_to_grayscale(base64_images)
    roi_cut_images = roicut(images)
    
    darken_images = preprocess.darken_pilimages(roi_cut_images,0.8)
    final_image = preprocess.enhance_pilimages(darken_images, 1.5)

    # background_cut_images = background_cut(images, 0.6)
    # for idx, img in enumerate(roi_cut_images):
    #     filename = f"processed_image_{idx + 1}.png"
    #     save_path = os.path.join(current_directory, filename)
    #     img.save(save_path)
 
    torch.cuda.empty_cache()
    return jsonify({"images": encode_base64(final_image)}), 200

def calculate_pairwise_euclidean_distance(features):

    squared_norms = torch.sum(features ** 2, dim=1, keepdim=True)
    distances = squared_norms + squared_norms.t() - 2 * torch.mm(features, features.t())
    distances = torch.sqrt(torch.clamp(distances, min=0.0))
    
    return distances
@app.route("/ai/register/inference", methods=["POST"])
def register():
    print("/ai/register/inference")
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    save_path = os.path.join(current_directory, 'processed_image.png')

    data = request.json
    if 'images' not in data:
        return jsonify({"error": "Images not found in request"}), 400
    base64_images = data['images']
    student_id = data['id']
    images = decode_base64_images_for_inference(base64_images)
    batch = torch.stack([transform(img) for img in images])
    with torch.no_grad():
        result = inference(batch.to(torch.float32).to("cuda"))
    # euclidean_distances = calculate_pairwise_euclidean_distance(result)
    bulk_index_vectors(es,index_name='palm-print-index', student_id=student_id, feature_vectors= result.cpu().numpy().tolist())
    torch.cuda.empty_cache()
    return jsonify({"feature_vectors": result.cpu().numpy().tolist()}), 200
    # return jsonify({"feature_vectors": euclidean_distances.cpu().numpy().tolist()}), 200


@app.route("/ai/vectorize", methods=["POST"])
def vectorize():
    start_time = time.time()
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    save_path = os.path.join(current_directory, 'processed_image.png')
    print("/ai/vectorize")
    data = request.json
    if 'images' not in data:
        return jsonify({"error": "Images not found in request"}), 400
    base64_images = data['images']
    images = decode_base64_images(base64_images)
    background_cut_images = background_cut_batch(images, 0.6)
    background_cut_images = [img.convert('RGB') for img in background_cut_images]
    roi_cut_images = roicut(background_cut_images)
    darken_images = preprocess.darken_pilimages(roi_cut_images,0.8)
    final_images = preprocess.enhance_pilimages(darken_images, 1.5)
    final_images = [img.convert('RGB') for img in final_images]
    batch = torch.stack([transform(img) for img in final_images])
    with torch.no_grad():
        result = inference(batch.to(torch.float32).to("cuda"))
    end_time = time.time()
    print("Time ",end_time-start_time)
    return jsonify({"feature_vector": result.cpu().numpy().tolist() })

@app.route("/ai/recognize/cosine-only", methods=["POST"])
def cosine_only_search():
    
    data = request.json
    if 'feature_vector' not in data:
        return jsonify({"error": "Feature vector not found in request"}), 400
    
    result = data['feature_vector']

    top1 = bulk_cosine_similarity_search(es, "palm-print-index", result)
    
    return jsonify(verify_palm_print(top1))

@app.route("/ai/recognize/cosine", methods=["POST"])
def cosine_search():
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    save_path = os.path.join(current_directory, 'processed_image.png')
    print("/ai/recognize/cosine")
    data = request.json
    if 'images' not in data:
        return jsonify({"error": "Images not found in request"}), 400
    base64_images = data['images']
    images = decode_base64_images(base64_images)
    background_cut_images = background_cut_batch(images, 0.6)
    background_cut_images = [img.convert('RGB') for img in background_cut_images]
    roi_cut_images = roicut(background_cut_images)
    darken_images = preprocess.darken_pilimages(roi_cut_images,0.8)
    final_images = preprocess.enhance_pilimages(darken_images, 1.5)
    final_images = [img.convert('RGB') for img in final_images]
    batch = torch.stack([transform(img) for img in final_images])
    with torch.no_grad():
        result = inference(batch.to(torch.float32).to("cuda"))
    top1 = bulk_cosine_similarity_search(es, "palm-print-index", result.cpu().numpy().tolist())
    
    return jsonify(verify_palm_print(top1))


@app.route("/ai/recognize/euclidean", methods=["POST"])
def euclidean_search():
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    save_path = os.path.join(current_directory, 'processed_image.png')
    print("/ai/recognize/euclidean")
    data = request.json
    if 'images' not in data:
        return jsonify({"error": "Images not found in request"}), 400
    base64_images = data['images']
    images = decode_base64_images(base64_images)
    background_cut_images = background_cut_batch(images, 0.6)
    background_cut_images = [img.convert('RGB') for img in background_cut_images]
    roi_cut_images = roicut(background_cut_images)
    darken_images = preprocess.darken_pilimages(roi_cut_images,0.8)
    final_images = preprocess.enhance_pilimages(darken_images, 1.5)
    final_images = [img.convert('RGB') for img in final_images]
    batch = torch.stack([transform(img) for img in final_images])
    with torch.no_grad():
        result = inference(batch.to(torch.float32).to("cuda"))
    top1 = bulk_euclidean_similarity_search(es, "palm-print-index", result.cpu().numpy().tolist())
    
    return jsonify(verify_palm_print(top1))

@app.route("/ai/database/delete-all", methods=["GET"])
def delete_all():
    delete_all_documents_in_index(es, "palm-print-index")
    return jsonify({'message': 'delete success'})

@app.route("/ai/database/list-all", methods=["GET"])
def list_all():
    result = list_documents_in_index(es, "palm-print-index",debug=False)
    return jsonify(result)


@app.route("/hello-world", methods=["GET"])
def hello_world():
    return jsonify("hello world")

if __name__ == "__main__":
    create_palm_print_index()
    # delete_all_documents_in_index(es, "palm-print-index")
    # list_documents_in_index(es , "palm-print-index")
    app.run(host="0.0.0.0", port=5000, debug=True)
