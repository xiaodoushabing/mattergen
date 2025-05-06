#%%
from pymongo import MongoClient

from core.settings import get_settings
settings = get_settings()

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
        logger.debug(f"Connecting to MongoDB at {settings.mongo_host}:{settings.mongo_port} in database {settings.db_name}, collection {settings.collection_name}...")
        client = MongoClient(f"mongodb://{settings.mongo_host}:{settings.mongo_port}/", serverSelectionTimeoutMS=5000)
        logger.debug("Pinging MongoDB server to verify connection...")
        client.admin.command('ping')
        logger.info("Successfully verified connection to MongoDB.")
        db = client[settings.db_name]
        lattice_collection = db[settings.collection_name]
        logger.info("Successfully connected to MongoDB.")
        return client, lattice_collection
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}", exc_info=True)
        raise ConnectionError(f"Failed to connect to MongoDB at {settings.mongo_host}:{settings.mongo_port}: {e}", exc_info=True)

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