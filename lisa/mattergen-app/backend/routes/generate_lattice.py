#%%
from fastapi import HTTPException, APIRouter, BackgroundTasks 
from models.generate import GenerateRequest, GenerateResponse
from services.generate_service import run_generation_and_processing

from core.logging_config import get_logger
logger =get_logger(route="generate")

router = APIRouter()

@router.post("/lattices", status_code=202, response_model=GenerateResponse)
async def generate_lattice(request: GenerateRequest, 
                           background_task: BackgroundTasks):
    """
    Schedules the background task for lattice generation based on provided magnetic density
    and guidance factor permutations.

    The endpoint immediately responds with a 202 status code, indicating that the lattice
    generation task is successfully scheduled in the background. The task will generate lattice
    structures based on the specified parameters.

    Args:
        request (GenerateRequest): Contains the magnetic density, guidance factor, and batch size.
        background_task (BackgroundTasks): Provides functionality to run tasks in the background.

    Returns:
        dict: A message confirming that the lattice generation task has been accepted and scheduled,
              along with details about the number of permutations.

    Raises:
        HTTPException: If either magnetic_density or guidance_factor is not provided in the request.
    """
    logger.info("Received request to schedule background lattice generation.")
    
    mag_density = request.magnetic_density
    guidance_factor = request.guidance_factor
    batch_size = request.batch_size

    total_permutations = len(mag_density) * len(guidance_factor)

    if not (mag_density and guidance_factor):
        logger.error("ERROR: Must provide at least one magnetic density and one guidance factor.")
        raise HTTPException(status_code=400, detail="Must provide at least one magnetic_density and one guidance_factor.")

    logger.info(
        f"Scheduling background task for lattice generation with the following parameters: "
        f"magnetic_density={mag_density}, "
        f"guidance_factor={guidance_factor}, "
        f"batch_size={batch_size}, "
    )

    background_task.add_task(run_generation_and_processing,
                            mag_density,
                            guidance_factor,
                            batch_size
                            )

    logger.info(
        f"Background task scheduled for lattice generation with {total_permutations} permutations."
    )
        
    return {
        "message": "Lattice generation task accepted and running in background.",
        "details": f"Scheduled {total_permutations} permutations."
    }
