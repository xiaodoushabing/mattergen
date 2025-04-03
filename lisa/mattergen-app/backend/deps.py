from fastapi import Depends
from pymongo.collection import Collection
from database import get_lattice_collection # Dependency to get lattice collection
from services.store_service import StoreService
from services.retrieval_service import RetrievalService

# Dependency to inject StoreService
def get_store_service(lattice_collection: Collection = Depends(get_lattice_collection), cal_mattersim: bool = True, model: int =5):
    """
    FastAPI dependency provider for the StoreService.

    Injects the MongoDB lattice collection dependency and uses optional
    parameters (potentially from query parameters or request body in a web context)
    to configure and return an initialized StoreService instance.

    Args:
        lattice_collection (Collection): The MongoDB collection for lattice data,
                                         injected by FastAPI via `Depends(get_lattice_collection)`.
        cal_mattersim (bool): Flag indicating whether MatterSim calculations
                              should be performed during storage operations.
                              Defaults to True.
        model (int): Identifier for the MatterSim model to use if calculations
                     are performed. Defaults to 5.

    Returns:
        StoreService: An initialized instance of the StoreService.
    """
    return StoreService(lattice_collection=lattice_collection, cal_mattersim=cal_mattersim, model=model)

# Dependency to inject RetrievalService
def get_lattices(lattice_collection: Collection = Depends(get_lattice_collection)):
    """
    FastAPI dependency provider for the RetrievalService.

    Injects the MongoDB lattice collection dependency and returns an initialized
    RetrievalService instance for querying lattice data.

    Args:
        lattice_collection (Collection): The MongoDB collection for lattice data,
                                         injected by FastAPI via `Depends(get_lattice_collection)`.

    Returns:
        RetrievalService: An initialized instance of the RetrievalService.
    """
    return RetrievalService(lattice_collection)
