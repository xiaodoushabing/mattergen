#%% import libraries
import torch
from ase.io import read
from mattersim.applications.batch_relax import BatchRelaxer
from mattersim.forcefield.potential import Potential
from mattersim.datasets.utils.build import build_dataloader

    #%%
def mattersim_prediction(structures_path, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running MatterSim on {device}")

    # load the model
    potential = Potential.from_checkpoint(load_path=f"MatterSim-v1.0.0-{model}M.pth", device=device)

    # load all structures from the extxyz file
    # ASE's read returns a list of Atoms objects, which can be directly passed to build_dataloader().
    structures = read(structures_path, index=":")

    # build the dataloader that is compatible with MatterSim
    dataloader = build_dataloader(structures, only_inference=True)
    # make predictions
    predictions = potential.predict_properties(dataloader, include_forces=True, include_stresses=True)

    return predictions
