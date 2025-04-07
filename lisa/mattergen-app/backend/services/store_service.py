#%%
import os
# os.path.join(os.path.dirname(__file__), "/utils")
from .utils import parse_extxyz_to_json, mattersim_prediction, extract_metadata
from pymongo import collection

from logging_config import get_logger
logger = get_logger(service="store")

#%% define function to add data into db
class StoreService:
    """
    A service class for processing and storing lattice data into a MongoDB collection.

    This service handles parsing lattice information from `.extxyz` files within specified
    directories, extracting metadata (magnetic density, guidance factor) from the directory names, 
    and optionally calculating and storing MatterSim predictions (energy, forces, stresses).

    Attributes:
        lattice_collection (pymongo.collection.Collection): MongoDB collection to store lattice data.
        cal_mattersim (bool): Flag indicating whether to calculate and store MatterSim predictions (default: True).
        model (int): MatterSim model size identifier (e.g., 1 or 5) used for predictions (default: 5).
    """
    def __init__(self, lattice_collection: collection.Collection, cal_mattersim: bool = True, model: int=5):
        """
        Initializes the StoreService.

        Args:
            lattice_collection (pymongo.collection.Collection): The MongoDB collection for storing lattice data.
            cal_mattersim (bool, optional): Whether to calculate and store MatterSim predictions (default: True).
            model (int, optional): The MatterSim model size identifier for predictions (default: 5).
        """
        self.lattice_collection= lattice_collection
        self.cal_mattersim = cal_mattersim
        self.model = model
        if self.lattice_collection is not None:
            logger.info(f"StoreService initialized with lattice_collection {self.lattice_collection.name}, model size: {self.model}, cal_mattersim flag: {self.cal_mattersim}")
        else:
            logger.critical("StoreService initialized without a valid lattice_collection. Please check the MongoDB connection.")

    def add_data(self, model_results_dir: str):
        """
        Processes results from a directory, parses lattice data, and stores it in MongoDB.

        Extracts magnetic density and guidance factor from the directory name and parses
        the 'generated_crystals.extxyz' file to retrieve lattice structures. If `cal_mattersim`
        is True, it calculates MatterSim predictions (energy, forces, stresses) and stores them in MongoDB.

        Args:
            model_results_dir (str): Path to the directory containing model results, expected format: 
                                      "..._<mag_density>_<guidance_factor>", must contain "generated_crystals.extxyz".

        Raises:
            ValueError: If the directory name format is incorrect or if the 'generated_crystals.extxyz' 
                        file is empty or cannot be parsed.
            FileNotFoundError: If the 'generated_crystals.extxyz' file does not exist.
            RuntimeError: If there is an error during MongoDB insertion (`insert_one`).
            Exception: Re-raises unexpected exceptions encountered during the processing (e.g., parsing errors).
        """
        # extract metadata - magnetic density and guidance factor
        try:
            magnetic_density, guidance_factor = extract_metadata(model_results_dir)
            logger.debug(f"Extracted metadata: magnetic_density={magnetic_density}, guidance_factor={guidance_factor}")

            # parse .extxyz into structured JSON
            structures_path = os.path.join(model_results_dir, "generated_crystals.extxyz")
            logger.debug(f"Structures_path: {structures_path}")

            if not os.path.isfile(structures_path):
                logger.error(f"ERROR: {structures_path} does not exist.")
                raise FileNotFoundError(f"{structures_path} does not exist.")
            
            # only parse lattice data, don't do any energy predictions
            logger.debug(f"Parsing .extxyz file at {structures_path}...")
            lattices_data = parse_extxyz_to_json(structures_path)
            logger.info(f"Parsed {len(lattices_data)} lattices from {structures_path}")

            if not lattices_data:
                logger.error(f"ERROR: Failed to parse .extxyz file at {structures_path}.")
                raise ValueError(f"ERROR: Failed to parse .extxyz file at {structures_path}")

            if self.cal_mattersim:
                flag = False
                ms_predictions = None
                logger.info(f"Calculating MatterSim predictions for {len(lattices_data)} lattices...")
                try: 
                    ms_predictions = mattersim_prediction(structures_path, model=self.model)
                    flag = True
                    logger.info(f"MatterSim predictions completed.")
                except Exception as e:
                    logger.warning(f"WARNING: Failed to calculate MatterSim predictions for {model_results_dir}: {e}", exc_info=True)
                    logger.warning(f"Skipping MatterSim predictions due to error.")

            # insert lattices into MongoDB
            for (index, lattice) in enumerate(lattices_data, start=1):
                lattice_doc = {
                    "lattice_index": index,
                    "guidance_factor": guidance_factor,
                    "magnetic_density": magnetic_density,
                    # "file_path": model_results_dir,
                    "no_of_atoms": lattice["no_of_atoms"],
                    "cell_parameters": lattice["cell_params"],
                    "pbc": lattice["pbc"],
                    "atoms_list": lattice["atoms_list"],  # JSON: {"Fe": 2, "O": 3}
                    "atoms": {str(k): v for k, v in lattice["atoms"].items()}
                }

                if self.cal_mattersim and ms_predictions:
                    lattice_doc["ms_predictions"] = {
                        "energy": ms_predictions[0][index-1],
                        "forces": ms_predictions[1][index-1].tolist(),
                        "stresses": ms_predictions[2][index-1].tolist()
                    }
                try:
                    logger.debug(f"Inserting lattice {index} into MongoDB...")
                    self.lattice_collection.insert_one(lattice_doc)
                    logger.info(f"Successfully inserted lattice {index} from {model_results_dir} {'with' if flag else 'without'} mattersim predictions.")
                except Exception as e:
                    logger.error("ERROR: Failed to insert lattice into MongoDB", exc_info=True)
                    raise RuntimeError(f"ERROR: Failed to insert lattice {index} from {model_results_dir} into MongoDB: {e}", exc_info=True)
                    
        except Exception as e:
            logger.critical(f"ERROR: An unexpected error occurred while adding data for directory {model_results_dir}: {e}", exc_info=True)
            raise e

    #%%
    def process_batch(self, batch_results_dir: str):
        """
        Processes a batch of model results directories by calling `add_data` for each subdirectory.

        Iterates through directories in the `batch_results_dir`, processing and storing results from each subdirectory.
        Tracks success/failure for each directory and logs the summary.

        Args:
            batch_results_dir (str): Path to the parent directory containing subdirectories of model results.

        Returns:
            int: The number of directories successfully processed.
        """
        if not os.path.isdir(batch_results_dir):
            logger.error(f"The provided batch directory is not valid: {batch_results_dir}")
            return 0

        try:
            folders = os.listdir(batch_results_dir)
            num_folders = len(folders)
            
            if num_folders == 0:
                logger.warning(f"No subdirectories found in batch directory {batch_results_dir}.")
                return 0
            
            failed_dirs = []
            success_count = 0
            
            for (idx, folder) in enumerate(folders, start=1):
                single_result_dir = os.path.join(batch_results_dir, folder)
                if os.path.isdir(single_result_dir):
                    try:
                        logger.info(f"--> Processing directory {idx}/{num_folders}: {single_result_dir}")
                        self.add_data(single_result_dir)
                        success_count+=1
                        logger.info(f"✅ Successfully processed {single_result_dir}")
                    except Exception as e:
                        failed_dirs.append(single_result_dir)
                        logger.error(f"❌ ERROR: Failed to process {single_result_dir}:\n \t: {e}", exc_info=True)
                    logger.info("-"*120)
                else:
                    failed_dirs.append(single_result_dir)
                    logger.warning(f"⚠️ Skipping non-directory item: {single_result_dir}")
                    
        except Exception as e:
            logger.error(f"ERROR: An error occurred during batch processing: {e}", exc_info=True)
            return success_count
        
        logger.info(f"--- Batch Processing Summary ---")
        logger.info(f"Total items found: {num_folders}")
        logger.info(f"Directories successfully processed: {success_count}")
        
        if failed_dirs:
            logger.warning(f"⚠️ Directories that failed to process :")
            for d in failed_dirs:
                logger.warning(f"  - {d}")
        else:
            logger.info(f"✅ All directories processed successfully.")

        return success_count
