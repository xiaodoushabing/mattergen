from pymongo import collection
from typing import Dict, Any
from models.retrieve import LatticeRequest

class RetrievalService:
    """
    Service for retrieving lattice data from MongoDB based on query filters.
    
    This service provides methods to build query filters from user input 
    and retrieve lattice structures stored in a MongoDB collection.
    """
    def __init__(self, lattice_collection: collection.Collection):
        """
        Initializes the RetrievalService with a MongoDB collection.

        Args:
            lattice_collection (collection.Collection): The MongoDB collection storing lattice data.
        """
        self.lattice_collection = lattice_collection

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
        # print(f"generated MongoDB filters: {filters_dict}")
        return filters_dict


    def get_lattices_by_filters(self, filters: LatticeRequest):
        """
        Retrieve lattices from MongoDB that match the provided filters.
        Args:
            limit (int): The number of lattices to return.
            filters (dict): The filters to apply to the MongoDB query.
        Returns:
            list: A list of lattice documents.
        """
        try:
            filters_dict = self.build_filters(filters)
            lattices = list(self.lattice_collection.find(filters_dict).limit(filters.limit))
            print(f"Executing MongoDB query: {filters_dict}")
            print(f"Number of lattices found: {len(lattices)}")
            return lattices
        except Exception as e:
            raise RuntimeError(f"Error retrieving lattices by filters: {e}")


