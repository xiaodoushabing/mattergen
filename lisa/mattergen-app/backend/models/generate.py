from pydantic import BaseModel
from typing import List

class GenerateRequest(BaseModel):
    magnetic_density: List[float | int] = [0]
    guidance_factor: List[float | int] = [1]
    batch_size: int = 64

class GenerateResponse(BaseModel):
    message: str
    details: str