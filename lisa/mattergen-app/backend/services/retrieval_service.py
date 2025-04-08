from pymongo import collection
from models.retrieve import LatticeRequest
from bson import ObjectId

from core.logging_config import get_logger
logger = get_logger(service="retrieval")

class RetrievalService:
    """
    Service for retrieving lattice data from MongoDB based on query filters.

    This service provides methods to build query filters from user input and
    retrieve lattice structures stored in a MongoDB collection.It supports
    pagination and error handling.
    """
    def __init__(self, lattice_collection: collection.Collection):
        """
        Initializes the RetrievalService with a MongoDB collection.

        Args:
            lattice_collection (collection.Collection): The MongoDB collection storing lattice data.
        """
        self.lattice_collection = lattice_collection
        logger.info("RetrievalService initialized with lattice collection.")

    def build_filters(self, filters: LatticeRequest):
        """
        Constructs MongoDB query filters from a LatticeRequest object.

        This method converts filter parameters into a dictionary format compatible 
        with MongoDB's query syntax. Special handling is applied for fields like 
        'energy', which is nested inside 'ms_predictions'.

        Args:
            filters (LatticeRequest): The request object containing filter criteria.

        Returns:
            dict: A dictionary of MongoDB query filters.
        """
        logger.debug(f"Building filters from LatticeRequest: {filters}")
        filters_dict = {}

        for field in [
            "lattice_index",
            "guidance_factor",
            "magnetic_density",
            "no_of_atoms",
            "energy",
        ]:
            field_value = getattr(filters, field, None)
            if field_value is not None:
                mongo_field = "ms_predictions.energy" if field == "energy" else field
                filters_dict[mongo_field] = {f"${field_value.op}": field_value.value}
        
        logger.debug(f"Generated MongoDB filters: {filters_dict}")
        return filters_dict


    def get_lattices_by_filters(self, filters: LatticeRequest, last_id: str = None):
        """
        Retrieve lattices from MongoDB that match the provided filters using _id-based pagination.
        
        Args:
            filters (LatticeRequest): The filters to apply to the MongoDB query.
            last_id (str, optional): The _id of the last document from the previous page to continue pagination.
        
        Returns:
            tuple: A tuple containing a list of lattices and the last ID of the next page (or None).
        """
        try:
            logger.debug(f"Retrieving lattices with filters: {filters}")
            filters_dict = self.build_filters(filters)
            if last_id:
                filters_dict["_id"] = {"$gt": ObjectId(last_id)}
                logger.debug("Applied pagination with last_id: {}", last_id)
            
            logger.debug(f'Executing MongoDB query: {filters_dict}')
            lattices = list(self.lattice_collection.find(filters_dict)
                            .sort("_id", 1)
                            # fetch an extra lattice to determine if there'll be next page
                            .limit(filters.limit+1))
            
            for lattice in lattices:
                lattice["_id"] = str(lattice["_id"])

            next_page_last_id = None

            if len(lattices) > filters.limit:
                lattices = lattices[:filters.limit]
                next_page_last_id = str(lattices[-1]["_id"]) if lattices else None

            logger.info(f"Retrieved {len(lattices)} lattices, next page last_id: {next_page_last_id}")
            return lattices, next_page_last_id
            
        except Exception as e:
            logger.error(f"Error retrieving lattices by filters: {e}")
            raise RuntimeError(f"Error retrieving lattices by filters: {e}")
    
    def get_lattice_by_id(self, id: str):
        """
        Retrieve a lattice document from MongoDB by its unique _id.
        
        Args:
            id (str): The unique identifier (_id) of the lattice.

        Returns:
            dict: The lattice document, or None if not found.
        """
        try:
            logger.info(f"Retrieving lattice by id: {id}")
            lattice = self.lattice_collection.find_one({"_id": ObjectId(id)})
            if lattice:
                lattice["_id"] = str(lattice["_id"])
                logger.info(f"Lattice retrieved successfully: {lattice}")
                return lattice
            else:
                logger.warning(f"Lattice with id {id} not found.")
                return None
        except Exception as e:
            logger.error((f"Error retrieving lattices by id {id}: {e}"))
            raise RuntimeError(f"Error retrieving lattices by id {id}: {e}")