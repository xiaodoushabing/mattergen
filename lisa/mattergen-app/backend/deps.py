from fastapi import Depends
from pymongo.collection import Collection
from database import get_lattice_collection # Dependency to get lattice collection
from services.store_service import StoreService
from services.retrieval_service import RetrievalService

# Dependency to inject StoreService
def get_store_service(lattice_collection: Collection = Depends(get_lattice_collection), cal_mattersim=True, model=5):
    return StoreService(lattice_collection=lattice_collection, cal_mattersim=cal_mattersim, model=model)

def get_lattices(lattice_collection: Collection = Depends(get_lattice_collection)):
    return RetrievalService(lattice_collection)
