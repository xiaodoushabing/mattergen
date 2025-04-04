from fastapi import APIRouter, Depends, HTTPException
from models.retrieve import RetrieveResponse, LatticeRequest, LatticeResponse
from services.retrieval_service import RetrievalService
from deps import retrieve_lattices


router = APIRouter()

@router.post("/lattices", response_model=RetrieveResponse)
def get_lattices(filters: LatticeRequest, last_id: str = None, retrieval_service: RetrievalService = Depends(retrieve_lattices)):
    """
    Retrieve lattices from MongoDB based on the provided filters.
    The filters come from the body of the POST request (via LatticeRequest).
    """
    try:
        lattices, next_page_last_id = retrieval_service.get_lattices_by_filters(filters, last_id)
        
        if lattices:
            return RetrieveResponse(lattices=lattices, next_page_last_id=next_page_last_id)
        else:
            raise HTTPException(status_code=404, detail="Lattices not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/lattice/{id}", response_model=LatticeResponse)
def get_lattice(id: str = None, retrieval_service: RetrievalService = Depends(retrieve_lattices)):
    """
    Retrieve a lattice with the given ID.
    """
    try:
        lattice = retrieval_service.get_lattice_by_id(id)
        
        if lattice:
            return LatticeResponse(**lattice)
        else:
            raise HTTPException(status_code=404, detail="Lattices not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))