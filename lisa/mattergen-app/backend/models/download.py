from pydantic import BaseModel
from typing import Optional, List
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

default_dl_path = Path(os.getenv("MATTERGEN_DOWNLOAD_PATH"))
default_dl_path.mkdir(parents=True, exist_ok=True)

class DownloadRequest(BaseModel):
    lattice_id: List[str]
    dl_path: Optional[str] = default_dl_path

class DownloadResponse(BaseModel):
    message: str = "Download request processed successfully."
    file_name: Optional[str] = None