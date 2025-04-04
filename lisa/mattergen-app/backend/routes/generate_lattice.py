#%%
from fastapi import HTTPException, APIRouter, BackgroundTasks 

import logging

from models.generate import GenerateRequest, GenerateResponse
from services.generate_service import run_generation_and_processing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate_lattice", status_code=202, response_model=GenerateResponse)
async def generate_lattice(request: GenerateRequest, 
                           background_task: BackgroundTasks):
    """
    Accepts parameters and schedules lattice generation to run in the background.
    Responds immediately with status 202 Accepted.
    """
    logger.info("Received request to schedule background lattice generation.")
    
    mag_density = request.magnetic_density
    guidance_factor = request.guidance_factor
    batch_size = request.batch_size

    num_mag_density = len(mag_density)
    num_guidance_factor = len(guidance_factor)
    total_permutations = num_mag_density * num_guidance_factor

    if not (mag_density and guidance_factor):
        raise HTTPException(status_code=400, detail="Must provide at least one magnetic_density and one guidance_factor.")

    background_task.add_task(run_generation_and_processing,
                            mag_density,
                            guidance_factor,
                            batch_size
                            )
        
    return {
        "message": "Lattice generation task accepted and running in background.",
        "details": f"Scheduled {total_permutations} permutations."
    }
