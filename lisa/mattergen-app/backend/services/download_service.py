from typing import Dict

from core.logging_config import get_logger
logger = get_logger(service="download")

def format_lattice_extxyz(id: str, lattice: Dict):
    """
    Formats a lattice dictionary into an EXTXYZ string representation.

    This function takes a dictionary describing a crystal lattice structure,
    validates its contents, and converts it into the EXTXYZ file format commonly
    used for atomic structure data. The function performs thorough validation 
    checks for number of atoms, cell parameters, periodic boundary conditions (PBC),
    and atomic positions.

    Parameters:
    ----------
    id : str
        A unique identifier for the lattice, used in log messages.
    lattice : Dict
        A dictionary containing lattice data. Expected keys include:
            - "no_of_atoms": int
            - "cell_parameters": list of 9 floats
            - "pbc": string of length 3 (e.g., "TFT")
            - "atoms": dict mapping atom indices (as strings) to 
              {element: [x, y, z]} coordinates.

    Returns:
    -------
    str or None
        A formatted EXTXYZ string if the input is valid, otherwise None.
    """
    try: 
        logger.info(f'Processing lattice id {id}...')

        if lattice is None:
            logger.error(f"Lattice data is missing for ID {id}.")
            return None

        num_atoms = int(lattice.get("no_of_atoms", 0))
        if num_atoms < 1:
            logger.error(f"Invalid or missing 'no_of_atoms' ({num_atoms}) for lattice {id}")
            return None 

        cell_params = lattice.get("cell_parameters", [])
        if not isinstance(cell_params, list) or len(cell_params) != 9:
            logger.error(f"Invalid or missing 'cell_parameters' (length {len(cell_params) if cell_params is not None else 'None'}) for lattice {id}")
            return None

        pbc=lattice.get("pbc", "")
        if len(pbc) != 3:
            logger.error(f"Invalid pbc: {pbc}")
            return None

        header_line = f"""Lattice="{" ".join(map(str, cell_params))}" Properties=species:S:1:pos:R:3 pbc="{" ".join(pbc)}\""""
        
        atom_lines = []
        atoms_dict = lattice.get("atoms", {})
        if not atoms_dict or len(atoms_dict) != num_atoms:
            logger.error(f"Atom count mismatch for lattice. Header: {num_atoms}, Atoms found: {len(atoms_dict)}")
            return None
        
        try:
            sorted_atom_keys = sorted(atoms_dict.keys(), key=int)
        except ValueError:
            logger.error(f"Atom dictionary keys are not valid indices for lattice {id}. Keys: {list(atoms_dict.keys())}")
            return None

        for key in sorted_atom_keys:
            atom_data = atoms_dict[key]
            if not isinstance(atom_data, dict) or len(atom_data) != 1:
                logger.error(f"Invalid atom data format for key '{key}' in lattice {id}. Data: {atom_data}")
                return None

            element, coords = list(atom_data.items())[0]
            
            if not isinstance(coords, list) or len(coords) != 3:
                 logger.error(f"Invalid coordinates format for element '{element}' (key '{key}') in lattice {id}. Coords: {coords}")
                 return None

            try:
                atom_lines.append(f"{element} {' '.join(map(str, coords))}")
            except TypeError:
                 logger.error(f"Failed to format coordinates to string for element '{element}' (key '{key}') in lattice {id}. Coords: {coords}")
                 return None
            
        if len(atom_lines) != num_atoms:
            logger.error(f"Final atom lines count ({len(atom_lines)}) does not match expected atom count ({num_atoms}) for lattice {id}.")
            return None

        frame = f"{num_atoms}\n{header_line}\n" + "\n".join(atom_lines) + "\n"
        logger.debug(f"Successfully formatted lattice {id}.")
        return frame

    except Exception as e:
        logger.error(f"Unexpected error formatting lattice {id} to extxyz: {e}", exc_info=True)
        return None



     