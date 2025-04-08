#%% import libraries
import torch
from ase.io import read
from mattersim.forcefield.potential import Potential
from mattersim.datasets.utils.build import build_dataloader
import re

from core.logging_config import get_logger
logger = get_logger(service="utils")

def extract_metadata(input_dir):
    """Extracts magnetic density and guidance factor from a directory name.

    Assumes the directory name ends with a pattern like
    '_<magnetic_density>_<guidance_factor>', where magnetic_density and
    guidance_factor are float or integer values. Handles potential trailing
    slashes in the input directory path.

    Args:
        input_dir (str): The path to the directory whose name contains the metadata.
                         Expected format: "dft_mag_density_<magnetic density>_<guidance factor>"
                         or "dft_mag_density_<magnetic density>_<guidance factor>/".

    Returns:
        tuple[float, float]: A tuple containing the extracted magnetic density
                             and guidance factor as floats.

    Raises:
        ValueError: If the directory name does not match the expected pattern
                    or if the extracted values cannot be converted to floats.
    """
    # The directory name should include magnetic density and
    # guidance factor in the format: "mag_density_guidance_factor"
    # (e.g., "dft_mag_density_<magnetic density>_<guidance factor>")
    logger.info(f"Extracting metadata from directory: {input_dir}")
    match = re.search(r'_(\d+\.?\d*)_(\d+\.?\d*)$', input_dir.rstrip("/"))
    if not match:
        logger.error(f"ERROR: Could not extract magnetic density and guidance factor from: {input_dir}")
        raise ValueError(f"Could not parse magnetic density and guidance factor from: {input_dir}")
    
    try:
        magnetic_density = float(match.group(1))
        guidance_factor = float(match.group(2))
        logger.info(f"Extracted magnetic density: {magnetic_density}, guidance factor: {guidance_factor}")
    except (IndexError, ValueError) as e:
        logger.error(f"ERROR: Could not convert extracted values to float. {e}", exc_info=True)
        raise
    return magnetic_density, guidance_factor

#%%
def mattersim_prediction(structures_path, model):
    """Runs MatterSim inference on structures from an extxyz file.

    Loads a specified MatterSim model checkpoint, reads atomic structures
    using ASE, prepares a dataloader, and performs predictions for energy,
    forces, and stresses.

    Args:
        structures_path (str): Path to the .extxyz file containing the atomic
                               structures.
        model (str): The size identifier for the MatterSim model (e.g., 1 or 5).
                     This is used to construct the checkpoint filename like
                     "MatterSim-v1.0.0-{model}M.pth".

    Returns:
        dict: A dictionary containing the predicted properties (energy, forces,
              stresses) for the structures in the input file. The exact structure
              depends on the `potential.predict_properties` method of MatterSim.

    Raises:
        FileNotFoundError: If the specified model checkpoint file does not exist.
        Various ASE/MatterSim/PyTorch errors: If issues occur during file reading,
                                              model loading, or prediction.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running MatterSim on {device}")

    # load the model
    try:
        potential = Potential.from_checkpoint(load_path=f"MatterSim-v1.0.0-{model}M.pth", device=device)
        logger.debug(f"Loading of MatterSim {model}M potential model: {'successful' if potential else 'failed'}")
        # load all structures from the extxyz file
        # ASE's read returns a list of Atoms objects, which can be directly passed to build_dataloader().
        structures = read(structures_path, index=":")
        logger.debug(f"Reading structures from {structures_path}: {'successful' if structures else 'failed'}")

        # build the dataloader that is compatible with MatterSim
        dataloader = build_dataloader(structures, only_inference=True)
        logger.debug(f"Building dataloader from structures: {'successful' if dataloader else 'failed'}")
        # make predictions
        predictions = potential.predict_properties(dataloader, include_forces=True, include_stresses=True)
        logger.info(f"✅ Predictions made for {len(structures)} lattices using {model}M model.")
    except FileNotFoundError as e:
        logger.error(f"❌ Model checkpoint file not found: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"❌ Error occurred during MatterSim inference: {e}", exc_info=True)
        raise
    return predictions

#%%
## ensure files are stored in the format: dft_mag_density_<magnetic density>_<guidance factor>
def parse_extxyz_to_json(structures_path):
    """Parses an .extxyz file into a list of dictionaries representing structures.

    Reads an extended XYZ file, processing each frame (structure) into a
    standardized dictionary format containing metadata and atomic information.
    Handles common parsing errors for individual frames by printing an error
    and skipping the problematic frame.

    Args:
        structures_path (str): The path to the .extxyz file.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a
                    single lattice structure (frame) read from the file.
                    Each dictionary contains the following keys:
                    - 'no_of_atoms' (int): Number of atoms in the frame.
                    - 'cell_params' (list[str]): Lattice vectors (9 values) as strings.
                    - 'pbc' (str): Periodic boundary conditions (e.g., "TTT").
                    - 'atoms_list' (dict): Dictionary mapping element symbols (str)
                                           to their counts (int) in the frame.
                    - 'atoms' (dict): Dictionary mapping 1-based atom index (int) to
                                      another dictionary containing the element symbol
                                      (str) mapped to a list of its coordinates
                                      ([x, y, z] as floats).

    Raises:
        OSError: If the file specified by `structures_path` cannot be opened or read.
        Exception: For unexpected errors during file processing before frame iteration.
                   Note: IndexError, ValueError, and other Exceptions during the
                   parsing of individual frames are caught, printed, and the
                   frame is skipped, allowing the function to potentially return
                   data from valid frames in a partially malformed file.
    """
    try:
        with open(structures_path, 'r') as f:
            lines = f.read().strip().split("\n")
    except OSError as e:
        logger.error(f"ERROR: Could not read the file '{structures_path}'. || {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"ERROR: An unexpected error occurred while processing the file {structures_path}. || {e}")
        raise
    
    lattices = []
    i=0
    idx=1

    while i < len(lines):
        try:
            atom_count = int(lines[i].strip())
            lattice = lines[i + 1].strip()
            datas = lines[i + 2:i + 2 + atom_count]
            
            atom_list = {}
            atoms_data = {}

            lattice_split = lattice.replace('"', '').split()
            cell_params = lattice_split[:9]
            cell_params[0] = cell_params[0].split("=")[-1]

            pbc = lattice_split[-3:]
            pbc[0] = pbc[0].split("=")[-1]
            pbc = "".join(pbc)

            for index, data in enumerate(datas, start=1):
                split_data = data.split()
                atom = split_data[0]
                atom_list[atom] = atom_list.get(atom, 0) + 1
                atoms_data[index] = {atom: list(map(float, split_data[1:]))}
            
            lattices.append({
                "no_of_atoms": atom_count,
                "cell_params": cell_params,
                "pbc": pbc,
                "atoms_list": atom_list,
                "atoms": atoms_data,
            })
            i += 2 + atom_count
            idx += 1
        
        except IndexError as e:
            logger.warning(f"ERROR: IndexError encountered while parsing lattice at line {i + 1}. || {e}")
            i += 1
            idx += 1
            continue

        except ValueError as e:
            logger.warning(f"ERROR: ValueError encountered while parsing lattice at line {i + 1}. || {e}")
            i += 1
            idx += 1
            continue

        except Exception as e:
            logger.warning(f"ERROR: Unexpected error encountered while parsing lattice at line {i + 1}. || {e}")
            i += 1
            idx += 1
            continue

    return lattices
