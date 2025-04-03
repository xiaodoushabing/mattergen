from fastapi import APIRouter, Depends, HTTPException
from models.retrieve import LatticeResponse, LatticeRequest
from typing import List
from services.retrieval_service import RetrievalService
from deps import retrieve_lattices


router = APIRouter()

@router.post("/lattices", response_model=List[LatticeResponse])
def get_lattices(filters: LatticeRequest, retrieval_service: RetrievalService = Depends(retrieve_lattices)):
    """
    Retrieve lattices from MongoDB based on the provided filters.
    The filters come from the body of the POST request (via LatticeRequest).
    """
    try:
        lattices = retrieval_service.get_lattices_by_filters(filters)
        if lattices:
            return lattices
        else:
            raise HTTPException(status_code=404, detail="Lattices not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
