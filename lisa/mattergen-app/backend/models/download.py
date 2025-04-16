from pydantic import BaseModel, Field
from typing import Optional, List

class DownloadRequest(BaseModel):
    lattice_ids: List[str] = Field(..., min_length=1)
    filename: str = Field("lattices.extxyz", description="Desired filename for the downloaded file.")

class DownloadResponse(BaseModel):
    message: str
    details: Optional[str]