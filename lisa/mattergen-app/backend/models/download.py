from pydantic import BaseModel, Field
from typing import Optional, List
import os
# from dotenv import load_dotenv
# from pathlib import Path

# load_dotenv()

# default_dl_path = Path(os.getenv("MATTERGEN_DOWNLOAD_PATH"))
# default_dl_path.mkdir(parents=True, exist_ok=True)

class DownloadRequest(BaseModel):
    lattice_ids: List[str] = Field(..., min_length=1)
    # dl_path: Optional[str] = default_dl_path
    filename: str = Field("lattices.extxyz", description="Desired filename for the downloaded file.")

class DownloadResponse(BaseModel):
    message: str
    details: Optional[str]