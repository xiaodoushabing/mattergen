"""
Use Pydantic to define request/response models.
"""
from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Union

class FilterCondition(BaseModel):
    """
    Represents a single filter condition with an operator and a value.

    Attributes:
        op (Literal["lt", "gt", "lte", "gte", "eq"]): The MongoDB comparison operator to use
            for the filter condition. Allowed values are "lt" (less than), "gt" (greater than),
            "lte" (less than or equal to), "gte" (greater than or equal to), and "eq" (equal to).
        value (Union[int, float]): The value to compare against.
    """
    op: Literal["lt", "gt", "lte", "gte", "eq"]  # Allowed MongoDB operators
    value: Union[int | float]  # The value to compare against

class LatticeRequest(BaseModel):
    limit: int = 10
    lattice_index: Optional[FilterCondition] = None
    guidance_factor: Optional[FilterCondition] = None
    magnetic_density: Optional[FilterCondition] = None
    no_of_atoms: Optional[FilterCondition] = None
    # atoms_list: Optional[Dict[str, int]] = None
    # cell_parameters: Optional[List[float]] = None
    energy: Optional[FilterCondition] = None
    
    class Config:
        schema_extra = {
            "example": {
                "lattice_index": 8,
                "guidance_factor": 4,
                "magnetic_density": 3,
                "no_of_atoms": 12,
                # "cell_parameters": [
                #     3.570263147354126, 0.0, -0.6019660830497742, -0.11143380736674079,
                #     3.560295917167594, -0.6013088822364807, 0.0, 0.0, 11.015626907348633
                # ],
                # "atoms_list": { "Mn": 2, "Fe": 10 },
                "ms_predictions": {
                    "energy": -101.5636215209961
                }
            }
        }

class MatterSimPredictions(BaseModel):
    energy: Optional[float] = None
    forces: Optional[List[List[float]]] = None
    stresses: Optional[List[List[float]]] = None

class LatticeResponse(BaseModel):
    id: str = Field(alias="_id")
    lattice_index: int  
    guidance_factor: float
    magnetic_density: float
    no_of_atoms: int
    cell_parameters: List[float]
    pbc: str
    atoms_list: Dict[str, int]
    atoms: Dict[int, Dict[str, List[float]]]
    ms_predictions: Optional[MatterSimPredictions]

    class Config:
        json_schema_extra = {
            "example": {
                "lattice_index": 5,
                "guidance_factor": 4,
                "magnetic_density": 3,
                "no_of_atoms": 4,
                "cell_parameters": [
                    "2.424989700317383",
                    "0.0",
                    "-0.4290729761123657",
                    "-1.2108657529824995",
                    "4.009423200395767",
                    "-2.138798713684082",
                    "0.0",
                    "0.0",
                    "4.7041401863098145"
                ],
                "pbc": "TTT",
                "atoms_list": { "Fe": 4 },
                "atoms": {
                    "1": { "Fe": [ 0.36769923, 3.719278, -1.57624356 ] },
                    "2": { "Fe": [ -0.84765803, 3.71705548, 1.00461013 ] },
                    "3": { "Fe": [ 0.9697504, 1.70913964, 1.85766182 ] },
                    "4": { "Fe": [ -0.24349569, 1.71584686, -0.28625911 ] }
                },
                "ms_predictions": {
                    "energy": -33.91051483154297,
                    "forces": [
                        [
                            -0.0007370933890342712,
                            -0.004133265465497971,
                            0.09751494228839874
                        ],
                        [
                            -0.015158799476921558,
                            -0.013675919733941555,
                            -0.07872786372900009
                        ],
                        [
                            -0.0020486577413976192,
                            0.006459377706050873,
                            -0.05314664542675018
                        ],
                        [
                            0.017944592982530594,
                            0.011349787935614586,
                            0.03435956686735153
                        ]
                    ],
                    "stresses": [
                        [
                            0.6633874177932739,
                            0.03405218943953514,
                            -0.05720433592796326
                        ],
                        [
                            0.034051451832056046,
                            0.08872920274734497,
                            -0.03484552726149559
                        ],
                        [
                            -0.05720467120409012,
                            -0.03484563156962395,
                            0.18524159491062164
                        ]
                    ]
                }
            }
        }

class RetrieveResponse(BaseModel):
    lattices: List[LatticeResponse]
    next_page_last_id: Optional[str]