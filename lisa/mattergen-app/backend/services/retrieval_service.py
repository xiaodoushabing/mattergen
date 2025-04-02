from pymongo import collection
from typing import Dict, Any
from models.retrieve import LatticeRequest

class RetrievalService:
    def __init__(self, lattice_collection: collection.Collection):
        self.lattice_collection = lattice_collection

    def build_filters(self, filters: LatticeRequest):
        filters_dict = {}

        for field in ["lattice_index",
                      "guidance_factor",
                      "magnetic_density",
                      "no_of_atoms",
                      "energy",
                      ]:
            field_value = getattr(filters, field, None)
            if field_value:
                mongo_field = "ms_predictions.energy" if field == "energy" else field
                filters_dict[mongo_field] = {f"${field_value.op}": field_value.value}
        print(f"generated MongoDB filters: {filters_dict}")
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
            # Log the query and results for debugging
            print(filters.limit)
            print(f"Executing MongoDB query: {filters_dict}")
            print(f"Number of lattices found: {len(lattices)}")
            return lattices
        except Exception as e:
            raise RuntimeError(f"Error retrieving lattices by filters: {e}")


