#%%
from fastapi import APIRouter
import subprocess
import os
import tempfile
from typing import List

from services.store_service import StoreService
from database import connect_to_mongo, close_mongo

from logging_config import get_logger
logger = get_logger(service="generate")

router = APIRouter()

def run_generation_and_processing(mag_density: List,
                                  guidance_factor: List,
                                  batch_size: int):
    """
    Runs lattice generation and processing in the background using MatterGen and StoreService.

    This function generates crystal lattices for each permutation of magnetic densities and guidance factors.
    It calls the 'mattergen-generate' command for each combination and stores the results in temporary directories.
    Failures during individual MatterGen runs are logged, and processing continues with the next permutation.

    After lattice generation, the function uses the StoreService to process the generated directories.
    This includes parsing `.extxyz` files, optionally running MatterSim for energy, force, and stress calculations,
    and storing the results in the configured MongoDB collection.

    Args:
        mag_density (List[float]): List of magnetic densities to use for lattice generation.
        guidance_factor (List[float]): List of guidance factors to use for lattice generation.
        batch_size (int): The batch size parameter for the MatterGen generation command.

    Returns:
        None: This function runs as a background task and does not return values directly.
    
    Raises:
        HTTPException: Raises an HTTPException with a status code 500 if a critical error occurs during
                       the StoreService batch processing or MongoDB storage.
    """
    logger.info("Background task started: Running MatterGen permutations.")
    
    num_mag_density = len(mag_density)
    num_guidance_factor = len(guidance_factor)
    total_permutations = num_mag_density * num_guidance_factor

    if num_mag_density == 1 and num_guidance_factor == 1:
        logger.info(f"Background Task: Starting single batch generation (1 permutation) ...")
    else:
        logger.info(f"Background Task: Starting multi batch generation with {total_permutations} total permutations...")

    generated_batches_count = 0
    db_added_batches_count = 0

    with tempfile.TemporaryDirectory() as batch_results_dir:
        logger.info(f"Background Task: Storing generated lattices in temporary directory: {batch_results_dir}")
        current_permutation_index = 0
        for md in mag_density:
            for gf in guidance_factor:
                current_permutation_index += 1
                results_path = os.path.join(batch_results_dir,f"dft_mag_density_{md}_{gf}")

                gen_command = [
                    "mattergen-generate", results_path,
                    "--pretrained-name=dft_mag_density",
                    f"--batch_size={batch_size}",
                    f'--properties_to_condition_on={{"dft_mag_density": {md}}}',
                    f"--diffusion_guidance_factor={gf}"
                ]

                try:
                    logger.info(f"🚀 Background Task: Running {current_permutation_index}/{total_permutations} MatterGen inference:\n \t{' '.join(gen_command)}")
                    gen_result = subprocess.run(gen_command, capture_output=True, text=True, check=True)
                    logger.info(f"✅ Background Task: MatterGen for magnetic density {md}, guidance factor {gf} - successful\n {gen_result.stdout}")
                    generated_batches_count += 1
                except FileNotFoundError:
                    logger.critical(f"❌ CRITICAL ERROR in Background Task: 'mattergen-generate' command not found. Ensure it is installed and in the system PATH.")
                    return
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Background Task: MatterGen failed for mag_density={md}, guidance_factor={gf}:\n \t{e.stderr}")
                    logger.error("Background Task: Continuing with the next permutation.")
                    continue

        logger.info(f"Background Task: Completed {generated_batches_count}/{total_permutations} permutations successfully for MatterGen\n\n {'%'*120}\n")
        
        if generated_batches_count > 0:
            client = None
            try:
                client, lattice_collection = connect_to_mongo()
                if lattice_collection is None:
                    logger.warning(f"Failed to retrieve the 'lattice' collection from the database. Please verify the DB connection details and the collection configuration.")
                
                logger.info(f"⚙️ Background Task: Handing over {generated_batches_count} generated result directories in {batch_results_dir} to StoreService...")
                store_service = StoreService(lattice_collection)
                db_added_batches_count  = store_service.process_batch(batch_results_dir)
                logger.info(f"Background Task: StoreService completed processing results. {db_added_batches_count} batches added to DB.")
            except Exception as e:
                logger.warning(f"❌ Background Task: Error during StoreService processing (MatterSim/DB): {e}", exc_info=True)
            finally:
                if client:
                    close_mongo(client)
        else:
            logger.warning("Background Task: No MatterGen batches were generated successfully. Skipping database processing.")
        if generated_batches_count == db_added_batches_count:
            logger.info("✅ Background Task: All generated batches successfully added into database.")

    logger.info(f"Background Task finished. Generated {generated_batches_count}/{total_permutations} total permutations. Added {db_added_batches_count} batches to DB.")
