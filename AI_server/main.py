from flask import Flask, jsonify, request
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import cv2
import numpy as np

# Load environment variables from .env file
load_dotenv()
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
@app.route("/api/data", methods=["GET"])
def get_data():
    # Example: Get data from a collection
    collection = db["student_hand_images"]
    data = []
    for x in collection.find():
        data.append(x)  # Retrieve all documents from the collection
    image_data = data[0]["hand_images"]
    nparr = np.frombuffer(image_data[5], np.uint8)

    # Decode the image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imshow("Image", img)
    cv2.imwrite("test.jpg", img)
    cv2.waitKey(0)
    return jsonify([]), 200


if __name__ == "__main__":
    app.run(debug=True)
