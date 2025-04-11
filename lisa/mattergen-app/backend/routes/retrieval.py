from fastapi import APIRouter, Depends, HTTPException
from models.retrieve import RetrieveResponse, LatticeRequest, LatticeResponse
from services.retrieval_service import RetrievalService
from deps import retrieve_lattices

from core.logging_config import get_logger
logger = get_logger(route="retrieval")

router = APIRouter()

@router.post("/lattices/filter", response_model=RetrieveResponse)
async def get_lattices(filters: LatticeRequest, last_id: str = None, retrieval_service: RetrievalService = Depends(retrieve_lattices)):
    """
    Retrieves lattices from MongoDB based on the provided filters.

    This endpoint accepts filters in the body of the POST request (via LatticeRequest) and returns 
    a list of lattices that match the given criteria. If pagination is required, it can return 
    the ID of the last lattice on the current page for retrieving the next page.

    Args:
        filters (LatticeRequest): The filters to apply to the lattice retrieval.
        last_id (str, optional): The ID of the last retrieved lattice for pagination (defaults to None).
        retrieval_service (RetrievalService): The service used to fetch lattices from the database.

    Returns:
        RetrieveResponse: A response containing the list of lattices and the ID of the last lattice for pagination.

    Raises:
        HTTPException: If there is an issue with fetching the lattices or the lattices are not found.
    """
    logger.info(f"Received request to retrieve lattices with filters: {filters}")
    try:
        lattices, next_page_last_id = retrieval_service.get_lattices_by_filters(filters, last_id)
        
        if lattices:
            logger.info(f"Retrieved {len(lattices)} lattices. Next page last ID: {next_page_last_id}")
            return RetrieveResponse(lattices=lattices, next_page_last_id=next_page_last_id)
        else:
            logger.warning("No lattices found matching the provided filters.")
            raise HTTPException(status_code=404, detail="Lattices not found")
    except Exception as e:
        logger.error(f"Error occurred while retrieving lattices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/lattice/{id}", response_model=LatticeResponse)
async def get_lattice(id: str = None, retrieval_service: RetrievalService = Depends(retrieve_lattices)):
    """
    Retrieves a lattice with the specified ID.

    This endpoint returns the lattice corresponding to the given ID.

    Args:
        id (str): The ID of the lattice to retrieve.
        retrieval_service (RetrievalService): The service used to fetch the lattice from the database.

    Returns:
        LatticeResponse: The lattice data corresponding to the given ID.

    Raises:
        HTTPException: If the lattice with the provided ID is not found or an error occurs during retrieval.
    """
    logger.info(f"Received request to retrieve lattice with ID: {id}")
    if not id:
        logger.error("Lattice ID is required.")
        raise HTTPException(status_code=400, detail="Lattice ID is required.")
    try:
        lattice = retrieval_service.get_lattice_by_id(id)
        if lattice:
            logger.info(f"Successfully retrieved lattice with ID: {id}")
            return LatticeResponse(**lattice)
        else:
            logger.warning(f"Lattice with ID: {id} not found.")
            raise HTTPException(status_code=404, detail="Lattices not found")
    except Exception as e:
        logger.error(f"Error occurred while retrieving lattice with ID {id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))