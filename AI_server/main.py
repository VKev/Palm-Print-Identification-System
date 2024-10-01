from flask import Flask, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np
from bson import ObjectId
import io
from mambavision import load_model, get_embbedding, preprocess_images
from depthanythingv2 import background_cut
from IPython.display import display
from roiextraction import roicut, tensor_to_image

# Load environment variables from .env file
load_dotenv()


model = load_model(r"mambavision/checkpoints/fine_tuned_mamba_vision_L2_f1_2.pth")


# Initialize Flask app
app = Flask(__name__)
connection_string = os.getenv("MONGO_CONNECTION_STRING")
# MongoDB connection string
# Replace this with your actual MongoDB connection string
if connection_string:
    client = MongoClient(connection_string)
    print(client.list_database_names())
    db = client["HandIdDB"]
    print(db.list_collection_names())
else:
    raise Exception("MongoDB connection string not found in environment variables")


# Example endpoint to retrieve data from MongoDB
@app.route("/ai/register", methods=["GET"])
def get_data():
    # Example: Get data from a collection
    document_id = "SS160738"
    collection = db["student_hand_images"]
    document = collection.find_one({"_id": document_id})

    if document:
        image_data = document["hand_images"]
        images = []
        for idx, image in enumerate(image_data):
            image_stream = io.BytesIO(image)
            img = Image.open(image_stream)
            img = background_cut(img)
            img = roicut(img)
            images.append(img)

        preprocessed_image = preprocess_images(images)

        embeddings = get_embbedding(model, preprocessed_image).cpu().numpy()
        print(embeddings)
        # students = db["students"]
        # for idx, embedding in enumerate(embeddings):
        #     document = {
        #         "_id": f"{document_id}_{idx}",  # Create a unique ID for each embedding
        #         "vector": embedding.tolist(),  # Convert numpy array to list for MongoDB
        #     }
        #     print(document)
        #     students.insert_one(document)

    return jsonify([]), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
