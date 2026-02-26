from typing import Annotated
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

@dataclass
class ErrorModel:
    """
    Parameters defining an error model
    
    Attributes:
        type (str): type of error distribution
        std (float): error standard deviation, in variable units
        hcorr (float): horizontal correlation scale, in grid.x units
        vcorr (float): vertical correlation scale, in z units
        tcorr (float): temporal correlation scale, in hours
        cross_corr (dict[str, float]): cross-variable correlation dictionary
    """
    type: str
    std: float
    hcorr: float
    vcorr: float
    tcorr: float
    cross_corr: dict[str, float]
        
    def asdict(self) -> dict:
        return asdict(self)

@dataclass
class ObsRecord:
    """
    Represents a single observation record in the full list of observations

    Attributes:
        name (str): name of the observation record
        dataset_src (str): name of dataset source module for this observation
        model_src (str): name of the model source module for this observation
        nobs (int): number of individual observations in this record
        obs_window_min (int): offset from analysis time for the start of the observation window (hours)
        obs_window_max (int): offset from analysis time for the end of the observation window (hours)
        err (ErrorModel): error model used for this observation
        dtype (str): data type
        is_vector (bool): if this variable is a vector
        units (str): physical units of this observation
        z_units (str): vertical coordinate units
        time (datetime): time coordinate for this observation
        dt (float): representative time interval (hours) for this observation
    """
    name: str
    dataset_src: str
    model_src: str
    nobs: int
    obs_window_min: int
    obs_window_max: int
    dtype: str
    is_vector: bool
    units: str
    z_units: str
    time: datetime
    dt: float
    err: ErrorModel
    hroi: float
    vroi: float
    troi: float
    impact_on_state: dict
    def asdict(self) -> dict:
        return asdict(self)

@dataclass
class FieldRecord:
    """
    Represents a single 2D slice in the state vector.

    Attributes:
        name (str): name of the state variable
        model_src (str): name of the model source module for this variable
        dtype (str): data type
        is_vector (bool): if this variable is a vector
        units (str): physical units of this variable
        err_type (str): type of error model to use for this variable
        time (datetime): time coordinate for this field
        dt (float): representative time interval (hours) for this field
        k (float): vertical z coordinate index for this field
        pos (int): seek position (number of bytes) for the start of this field in the binary file
    """
    name: str
    model_src: str
    dtype: str
    is_vector: bool
    units: str
    err_type: str
    time: datetime
    dt: float
    k: float
    pos: int  # Byte offset in the binary file

    def asdict(self) -> dict:
        return asdict(self)

ProcID = Annotated[int, 'process id in comm']
ProcIDMem = Annotated[int, 'process id in comm_mem']
ProcIDRec = Annotated[int, 'process id in comm_rec']

MemID = Annotated[int, 'member id']
FieldRecordID = Annotated[int, 'field record id']
ObsRecordID = Annotated[int, 'obs record id']
PartitionID = Annotated[int, 'partition id']

FieldEns = Annotated[dict[tuple[MemID, FieldRecordID], np.ndarray], 'field-complete ensemble data']
StateEns = Annotated[dict[tuple[MemID, FieldRecordID], dict[PartitionID, np.ndarray]], 'state-complete ensemble data']
ObsSeq = Annotated[dict[ObsRecordID, dict[str, np.ndarray]], 'obs sequence']
ObsEns = Annotated[dict[tuple[MemID, ObsRecordID], np.ndarray], 'obs ensemble data']
LocalObsSeq = Annotated[dict[ObsRecordID, dict[PartitionID, dict[str, np.ndarray]]], 'local obs sequence']
LocalObsEns = Annotated[dict[tuple[MemID, ObsRecordID], dict[PartitionID, np.ndarray]], 'local obs ensemble data']
