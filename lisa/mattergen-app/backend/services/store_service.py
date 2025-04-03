#%%
import os
# os.path.join(os.path.dirname(__file__), "/utils")
from .utils import parse_extxyz_to_json, mattersim_prediction, extract_metadata
from pymongo import collection

#%% define function to add data into db
class StoreService:
    """
    A service class for processing and storing lattice data from model results into a MongoDB collection.

    This service handles parsing lattice information from `.extxyz` files within specified
    directories, extracting metadata (magnetic density, guidance factor) from the
    directory names, and optionally calculating and storing MatterSim predictions
    (energy, forces, stresses).

    Attributes:
        lattice_collection (pymongo.collection.Collection): The MongoDB collection instance
                                                            to store lattice data.
        cal_mattersim (bool): Flag indicating whether to calculate and store
                              MatterSim predictions.
        model (int): The MatterSim model size identifier (e.g., 1 or 5) to use
                     for predictions. Note: The default is 5 (as defined in deps.py),
                     adjust in deps.py if a different default is standard.
    """
    def __init__(self, lattice_collection: collection.Collection, cal_mattersim: bool, model: int):
        """
        Initializes the StoreService.

        Args:
            lattice_collection (pymongo.collection.Collection): The MongoDB collection
                instance for storing lattice data.
            cal_mattersim (bool, optional): Whether to calculate and store MatterSim
                predictions.
            model (int, optional): The MatterSim model size identifier to use for
                predictions.
        """
        self.lattice_collection= lattice_collection
        self.cal_mattersim = cal_mattersim
        self.model = model

    def add_data(self, model_results_dir: str):
        """
        Processes results from a directory, parses lattice data, and stores it in MongoDB.

        Extracts magnetic density and guidance factor from the directory name (expected
        format: "..._<mag_density>_<guidance_factor>"). Parses the
        'generated_crystals.extxyz' file within the directory to retrieve lattice structures.
        Inserts each structure as a document into the configured MongoDB collection.

        If `cal_mattersim` is True, it attempts to run MatterSim predictions (energy,
        forces, stresses) using the configured model and adds them to the document.
        MatterSim prediction failures are logged but do not stop the insertion process
        (data will be inserted without predictions in case of failure).

        Args:
            model_results_dir (str): Path to the directory containing the model results.
                The directory name must end with '_<float>_<float>' representing
                magnetic density and guidance factor respectively
                (e.g., "dft_mag_density_<magnetic density>_<guidance factor>").
                Must contain a file named "generated_crystals.extxyz".

        Raises:
            ValueError: If the directory name format is incorrect or if the
                        'generated_crystals.extxyz' file is empty or cannot be parsed.
            FileNotFoundError: If the 'generated_crystals.extxyz' file does not exist
                               in the specified directory.
            RuntimeError: If there is an error during MongoDB insertion (`insert_one`).
                          Note: This specifically wraps MongoDB exceptions.
            Exception: Re-raises other unexpected exceptions encountered during processing
                       (e.g., from `extract_metadata`, `parse_extxyz_to_json` setup).
        """
        # extract magnetic density and guidance factor
        try:
            magnetic_density, guidance_factor = extract_metadata(model_results_dir)

            # parse .extxyz into structured JSON
            structures_path = os.path.join(model_results_dir, "generated_crystals.extxyz")

            if not os.path.isfile(structures_path):
                raise FileNotFoundError(f"{structures_path} does not exist.")
            
            #only parse lattice data, don't do any energy predictions
            lattices_data = parse_extxyz_to_json(structures_path)

            if not lattices_data:
                raise ValueError(f"ERROR: Failed to parse .extxyz file at {structures_path}")

            if self.cal_mattersim:
                flag = False
                print(f"Starting Mattersim predictions...")
                try: 
                    ms_predictions = mattersim_prediction(structures_path, model=self.model)
                    flag = True
                    print(f"Mattersim predictions completed.")
                except Exception as e:
                    print(f"ERROR: Failed to run MatterSim predictions. Error: {e}")
                    print("Proceeding without Mattersim predictions")
                    ms_predictions = None

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
                    # print(f"Lattice document to insert: {lattice_doc}")
                    self.lattice_collection.insert_one(lattice_doc)
                    # print(f"Successfully inserted lattice {index} for {model_results_dir}. With mattersim predictions: {flag}")
                except Exception as e:
                    raise RuntimeError(f"ERROR: Failed to insert lattice {index} into MongoDB for {model_results_dir}: {e}")
                    
        except Exception as e:
            raise e

    #%%
    def process_batch(self, batch_results_dir: str):
        """
        Processes a batch of model results directories by calling `add_data` for each.

        Iterates through items within the `batch_results_dir`. For each item that is
        a directory, it calls the `add_data` method to process and store the results
        contained within that subdirectory. Tracks the number of successfully processed
        directories and lists any directories that failed processing.

        Args:
            batch_results_dir (str): Path to the parent directory containing
                                     subdirectories of individual model results.

        Returns:
            int: The number of subdirectories successfully processed (i.e., `add_data`
                 completed without raising an exception).
        """
        if not os.path.isdir(batch_results_dir):
            print(f"The provided batch directory is not valid: {batch_results_dir}")
            return 0

        try:
            folders = os.listdir(batch_results_dir)
            num_folders = len(folders)
            
            if num_folders == 0:
                print(f"No subdirectories found to process in {batch_results_dir}.")
                return 0
            
            failed_dirs = []
            success_count = 0
            processed_dirs_count = 0
            
            for (idx, folder) in enumerate(folders, start=1):
                print(f"Processing folder {idx}/{num_folders}")
                single_result_dir = os.path.join(batch_results_dir, folder)
                if os.path.isdir(single_result_dir):
                    processed_dirs_count += 1
                    print(f"--> Processing directory: {single_result_dir}")
                    try:
                        self.add_data(single_result_dir)
                        success_count+=1
                        print(f"✅ Successfully processed {single_result_dir}")
                    except Exception as e:
                        failed_dirs.append(single_result_dir)
                        print(f"❌ ERROR: Failed to process {single_result_dir}:\n \t{type(e).__name__}: {e}")
                    print("-"*120)
                else:
                    failed_dirs.append(single_result_dir)
                    print(f"--> Skipping non-directory item: {single_result_dir}")
                    
        except Exception as e:
            print(f"❌ ERROR: An error occurred during data processing: {e}")
            return success_count
        
        print("\n--- Batch Processing Summary ---")
        print(f"Total items found: {num_folders}")
        print(f"Items processed as directories: {processed_dirs_count}")
        print(f"Successfully processed directories: {success_count}")
        
        if failed_dirs:
            print(f"⚠️ Directories that failed processing ({len(failed_dirs)}):")
            for d in failed_dirs:
                print(f"  - {d}")
        elif processed_dirs_count == success_count:
            print("✅ All directories processed successfully.")
        return success_count

    # #%%
    # from database import connect_to_mongo, close_mongo
    # if __name__ == "__main__":
    #     client, collection = connect_to_mongo()
    #     process_batch(sys.argv[2], collection)
    #     close_mongo(client)
