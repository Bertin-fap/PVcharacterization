__all__ = ['DEFAULT_DIR',
           'DATA_BASE_NAME',
           'DATA_BASE_TABLE',
           'DATA_BASE_TABLE_EXP',
           'USED_COLS',
           'PARAM_UNIT_DIC',
           'IRRADIANCE_DEFAULT_LIST',
           'TREATMENT_DEFAULT_LIST',]

# Standard library imports
from pathlib import Path

PARAM_UNIT_DIC = {
    "IrrCorr": "W/m2",
    "Pmax": "W",
    "Voc": "V",
    "Isc": "A",
    "Fill Factor": "1",
    "Rseries":chr(937),
}
DEFAULT_DIR = Path.home()
DATA_BASE_NAME = "pv.db"
DATA_BASE_TABLE = "PV_descp"
DATA_BASE_TABLE_EXP = "exp_values"
USED_COLS = [
    "Title",
    "Pmax",
    "Fill Factor",
    "Voc",
    "Isc",
    "Rseries",
    "Rshunt",
    "Vpm",
    "Ipm",
]
IRRADIANCE_DEFAULT_LIST = [200,400,600,800,1000,2000,4000]
TREATMENT_DEFAULT_LIST =  ["T0", "T1", "T2", "T3", "T4"]