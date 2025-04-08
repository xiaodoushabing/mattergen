#%%
from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

mongo_host = os.environ.get("MONGO_HOST")
mongo_port = os.environ.get("MONGO_PORT")
db_name = os.environ.get("DB_NAME")
collection_name = os.environ.get("COLLECTION_NAME")

from core.logging_config import get_logger
logger = get_logger(service="database")

def connect_to_mongo():
    """
    Connect to MongoDB and return the client and lattice collection.
    
    This function establishes a connection to a MongoDB instance using the 
    connection details (host, port, database name, and collection name).
    It logs the connection details and returns the MongoDB client and the 
    specified lattice collection for further operations.

    Returns:
        tuple: A tuple containing the MongoDB client and the lattice collection.
    
    Raises:
        ConnectionError: If there is an issue connecting to the MongoDB instance.
    """
    try: 
        logger.debug(f"Connecting to MongoDB at {mongo_host}:{mongo_port} in database {db_name}, collection {collection_name}...")
        client = MongoClient(f"mongodb://{mongo_host}:{mongo_port}/")
        db = client[db_name]
        lattice_collection = db[collection_name]
        logger.info("Successfully connected to MongoDB.")
        return client, lattice_collection
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}", exc_info=True)
        raise ConnectionError(f"Failed to connect to MongoDB at {mongo_host}:{mongo_port}: {e}", exc_info=True)

def close_mongo(client: MongoClient):
    """
    Close the MongoDB connection.

    This function safely closes the MongoDB client connection to ensure
    that resources are freed up and the connection is properly terminated.

    Args:
        client (MongoClient): The MongoDB client to close.
    
    Logs a message confirming that the connection has been closed.
    """
    if client:
        try:
            client.close()
            logger.info("MongoDB connection successfully closed.")
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}", exc_info=True)
            raise

# get_lattice_collection not used for background processing
def get_lattice_collection():
    """
    Dependency injection for FastAPI to provide the lattice collection.

    This function establishes a connection to MongoDB and yields the lattice 
    collection for use in FastAPI routes or background tasks. Once the operation 
    is complete, it ensures the MongoDB client is closed.

    Yields:
        Collection: The MongoDB lattice collection to be used by FastAPI.
    
    Ensures proper resource management by closing the client after use.
    """
    client, lattice_collection = connect_to_mongo()
    try:
        yield lattice_collection
    finally:
        close_mongo(client)