__all__ = ['DEFAULT_DIR',
           'DATA_BASE_NAME',
           'DATA_BASE_TABLE',
           'DATA_BASE_TABLE_EXP',
           'USED_COLS',
           'PARAM_UNIT_DIC',
           ]

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
    "Voc",
    "Isc",
    "Rseries",
    "Rshunt",
    "Pmax",
    "Vpm",
    "Ipm",
    "Fill Factor",
]