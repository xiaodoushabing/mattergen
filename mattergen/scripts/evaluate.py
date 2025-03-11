# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from pathlib import Path
from typing import Literal

import fire
import numpy as np

from mattergen.common.utils.eval_utils import load_structures
from mattergen.common.utils.globals import get_device
from mattergen.evaluation.evaluate import evaluate
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher,
    DefaultOrderedStructureMatcher,
)


def main(
    structures_path: str,
    relax: bool = True,
    energies_path: str | None = None,
    structure_matcher: Literal["ordered", "disordered"] = "disordered",
    save_as: str | None = None,
    potential_load_path: (
        Literal["MatterSim-v1.0.0-1M.pth", "MatterSim-v1.0.0-5M.pth"] | None
    ) = None,
    reference_dataset_path: str | None = None,
    device: str = str(get_device()),
):
    structures = load_structures(Path(structures_path))
    energies = np.load(energies_path) if energies_path else None
    structure_matcher = (
        DefaultDisorderedStructureMatcher()
        if structure_matcher == "disordered"
        else DefaultOrderedStructureMatcher()
    )
    reference = None
    if reference_dataset_path:
        reference = LMDBGZSerializer().deserialize(reference_dataset_path)

    metrics = evaluate(
        structures=structures,
        relax=relax,
        energies=energies,
        structure_matcher=structure_matcher,
        save_as=save_as,
        potential_load_path=potential_load_path,
        reference=reference,
        device=device,
    )
    print(json.dumps(metrics, indent=2))


def _main():
    fire.Fire(main)


if __name__ == "__main__":
    _main()
