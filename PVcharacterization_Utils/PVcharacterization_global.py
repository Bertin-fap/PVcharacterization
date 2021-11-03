__all__ = ['COL_NAMES',
           'DEFAULT_DIR',
           'DATA_BASE_NAME',
           'DATA_BASE_TABLE_FILE',
           'DATA_BASE_TABLE_EXP',
           'PLOT_PARAMS_DICT',
           'IRRADIANCE_DEFAULT_LIST',
           'PARAM_UNIT_DIC',
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
DATA_BASE_TABLE_FILE = "PV_descp"
DATA_BASE_TABLE_EXP = "exp_values"
COL_NAMES = [
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

# TO DO deal the case with more than 10 modules type
PLOT_PARAMS_DICT = {
                     'markers': ["o", "+", "s", "<", ">", "p", "1", "2", "3", "4"], 
                     'marker_colors': ['#0000A0','#1569C7','#78f89d','#FFEB3B','#E64A19'],
                     'marker_size': 40,
                     'legend_fontsize': 16,
                     'ticks_fontsize': 12,
                     'labels_fontsize': 14,
                     'title_fontsize':16,
                     'fig_width': 10, 
                     'fig_height_unit': 4,
                     'fig_title_height': 3,
                     'bbox_x0': 0.6, 
                     'bbox_y0': 0, 
                     'bbox_width': 0.5, 
                     'bbox_height': 1,
                     'irr_add_nbr':2,}