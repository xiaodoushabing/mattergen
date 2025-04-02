#%%
from fastapi import HTTPException, APIRouter, Depends
import subprocess
import os
import tempfile
import logging

from models.generate import GenerateRequest, GenerateResponse
from services.store_service import StoreService
from deps import get_store_service

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/generate_lattice", response_model=GenerateResponse)
def generate_lattice(request: GenerateRequest, store_service: StoreService = Depends(get_store_service)):
    """
    Run MatterGen to generate new lattices, then MatterSim for energy calculations.
    Finally, store results in MongoDB. The results folder is temporary.
    Batch/singular inferences are accepted.
    """
    
    mag_density = request.magnetic_density
    guidance_factor = request.guidance_factor
    batch_size = request.batch_size

    num_mag_density = len(mag_density)
    num_guidance_factor = len(guidance_factor)
    total_permutations = num_mag_density * num_guidance_factor

    if num_mag_density == 1 and num_guidance_factor == 1:
        logger.info(f"Starting single batch generation ...")
    else:
        logger.info(f"Starting multi batch generation with {total_permutations} total permutations...")

    with tempfile.TemporaryDirectory() as batch_results_dir:
        count = 0
        for md in mag_density:
            for gf in guidance_factor:
                results_path = os.path.join(batch_results_dir,f"dft_mag_density_{md}_{gf}")

                gen_command = [
                    "mattergen-generate", results_path,
                    "--pretrained-name=dft_mag_density",
                    f"--batch_size={batch_size}",
                    f'--properties_to_condition_on={{"dft_mag_density": {md}}}',
                    f"--diffusion_guidance_factor={gf}"
                ]

                try:
                    logger.info(f"🚀\tRunning {count+1}/{total_permutations} MatterGen inference:\n \t{' '.join(gen_command)}")
                    gen_result = subprocess.run(gen_command, capture_output=True, text=True, check=True)
                    logger.info(f"✅ MatterGen success:\n {gen_result.stdout}")
                    count += 1
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌\tMatterGen failed for mag_density={md}, guidance_factor={gf}:\n \t{e.stderr}")
                    logger.error("Continuing with the next permutation.")
                    continue
                
        logger.info(f"Completed {count}/{total_permutations} permutations successfully for MatterGen\n {'%'*120}\n")
        
        try:
            logger.info(f"⚙️\tRunning MatterSim & storing results in DB...")
            success_count = store_service.process_batch(batch_results_dir)
            logger.info(f"✅\tMatterSim successful and results stored in DB")
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"MatterSim/DB Error: {str(e)}")

    return GenerateResponse(
        status="success",
        message="Lattice generation and simulation completed!",
        total_permutations=total_permutations,
        generated_batches=count,
        db_added_batches=success_count
    )
