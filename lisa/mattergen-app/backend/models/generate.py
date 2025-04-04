"""
Use Pydantic to define request/response models for MatterGen inference.
"""
from pydantic import BaseModel
from typing import List

class GenerateRequest(BaseModel):
    magnetic_density: List[float | int] = [0]
    guidance_factor: List[float | int] = [1]
    batch_size: int = 64
    # mattersim_cal: bool = True
    # mattersim_model: Literal[1, 5] = 5

class GenerateResponse(BaseModel):
    # status: str
    message: str
    details: str
    # total_permutations: int
    # generated_batches: int
    # db_added_batches: int
