import json
from pymongo import MongoClient

def upload_json_to_mongodb(json_file_path, database_name, collection_name):
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client[database_name]
    collection = db[collection_name]

    # Read JSON file
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    # Insert JSON data into MongoDB collection
    collection.insert_many(json_data)

    print("JSON data uploaded successfully to MongoDB.")

if __name__ == "__main__":
    # Specify the path to your JSON file
    json_file_path = "ocr_results.json"
    
    # Specify the name of your MongoDB database and collection
    database_name = "ANPRS"
    collection_name = "anprdata"

    # Upload JSON data to MongoDB
    upload_json_to_mongodb(json_file_path, database_name, collection_name)
