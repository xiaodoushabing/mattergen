#%%
from pymongo import MongoClient
import os
from dotenv import load_dotenv

## ensure files are stored in the format: dft_mag_density_<magnetic density>_<guidance factor>

load_dotenv()

mongo_host = os.environ.get("MONGO_HOST")
mongo_port = os.environ.get("MONGO_PORT")
db_name = os.environ.get("DB_NAME")
collection_name = os.environ.get("COLLECTION_NAME")

def connect_to_mongo():
    """
    Connect to MongoDB and return the client and lattice collection.
    """
    print(f"Connecting to: MONGO_HOST: {mongo_host}, MONGO_PORT: {mongo_port}, DB_NAME: {db_name}, COLLECTION_NAME: {collection_name}")
    client = MongoClient(f"mongodb://{mongo_host}:{mongo_port}/")
    db = client[db_name]
    lattice_collection = db[collection_name]
    return client, lattice_collection

def close_mongo(client: MongoClient):
    """
    Close the MongoDB connection.
    """
    if client:
        client.close()
        print("MongoDB connection closed.")

def get_lattice_collection():
    """
    Dependency injection for FastAPI to provide the lattice collection.
    """
    client, lattice_collection = connect_to_mongo()
    try:
        yield lattice_collection
    finally:
        close_mongo(client)