__all__ = ['DEFAULT_DIR',
           'DATA_BASE_NAME',
		   'DATA_BASE_TABLE',
		   'USED_COLS',
		   'PARAM_UNIT_DIC']

# Standard library imports
from pathlib import Path

PARAM_UNIT_DIC = {
    "IrrCorr": "W/m2",
    "Pmax": "W",
    "Voc": "V",
    "Isc": "A",
    "Fill Factor": "",
}
DEFAULT_DIR = Path.home()
DATA_BASE_NAME = "pv.db"
DATA_BASE_TABLE = "PV_descp"
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