from fastapi import APIRouter, HTTPException, Response, Depends
from services.retrieval_service import RetrievalService
from deps import retrieve_lattices
from models.download import DownloadRequest, DownloadResponse

router = APIRouter()

@router.get("/lattices", response_model=DownloadResponse)
async def download_lattices(request: DownloadRequest,
                            retrieval_service: RetrievalService = Depends(retrieve_lattices)):
    pass