__all__ = ['COL_NAMES',
           'DEFAULT_DIR',
           'DATA_BASE_NAME',
           'DATA_BASE_TABLE_FILE',
           'DATA_BASE_TABLE_EXP',
           'ENCODING',
           'IRRADIANCE_DEFAULT_LIST',
           'NBR_MAX_PARAMS_PLOT',
           'PLOT_PARAMS_DICT',
           'PARAM_UNIT_DIC',
           'TREATMENT_DEFAULT_LIST',
           'change_config_pvcharacterization',
           'config_pvcharacterization',
           'WORKING_DIR',]
           
def change_config_pvcharacterization(data_folder):

    from functools import reduce
    from pathlib import Path
    

    import yaml
    

    path_config_file = Path(__file__).parent / Path('Pvcharacterization.yaml')
    with open(path_config_file) as file:
        parse_yaml = yaml.safe_load(file)
        DEFAULT_DIR = Path.home()
        data_folder_list = list(Path(data_folder).parts)
        del data_folder_list[0:len(DEFAULT_DIR.parts)]
        WORKING_DIR = str(reduce(lambda a, b:Path(a) / Path(b), data_folder_list))
        parse_yaml['WORKING_DIR'] = WORKING_DIR
    
    with open(path_config_file, 'w') as file:
        outputs = yaml.dump(parse_yaml, file)

def config_pvcharacterization():

    from pathlib import Path

    import yaml
    

    path_config_file = Path(__file__).parent / Path('Pvcharacterization.yaml')
    with open(path_config_file) as file:
        parse_yaml = yaml.safe_load(file)
    PARAM_UNIT_DIC = parse_yaml['PARAM_UNIT_DIC']
    PARAM_UNIT_DIC['Rseries'] = chr(937)
    DATA_BASE_NAME = parse_yaml['DATA_BASE_NAME']
    ENCODING = parse_yaml['ENCODING']
    DATA_BASE_TABLE_FILE = parse_yaml['DATA_BASE_TABLE_FILE']
    DATA_BASE_TABLE_EXP = parse_yaml['DATA_BASE_TABLE_EXP']
    COL_NAMES = parse_yaml['COL_NAMES']
    IRRADIANCE_DEFAULT_LIST = parse_yaml['IRRADIANCE_DEFAULT_LIST']
    TREATMENT_DEFAULT_LIST = parse_yaml['TREATMENT_DEFAULT_LIST']
    PLOT_PARAMS_DICT = parse_yaml['PLOT_PARAMS_DICT']
    NBR_MAX_PARAMS_PLOT = parse_yaml['NBR_MAX_PARAMS_PLOT']
    DEFAULT_DIR = Path.home()
    WORKING_DIR = Path.home() / Path(parse_yaml['WORKING_DIR'])
    
    
    return (COL_NAMES,
            DEFAULT_DIR, 
            DATA_BASE_NAME,
            DATA_BASE_TABLE_FILE,
            DATA_BASE_TABLE_EXP,
            ENCODING,
            IRRADIANCE_DEFAULT_LIST,
            NBR_MAX_PARAMS_PLOT,
            PARAM_UNIT_DIC,
            PLOT_PARAMS_DICT,
            TREATMENT_DEFAULT_LIST,
            WORKING_DIR)

(COL_NAMES,
 DEFAULT_DIR, 
 DATA_BASE_NAME,
 DATA_BASE_TABLE_FILE,
 DATA_BASE_TABLE_EXP,
 ENCODING,
 IRRADIANCE_DEFAULT_LIST,
 NBR_MAX_PARAMS_PLOT,
 PARAM_UNIT_DIC,
 PLOT_PARAMS_DICT,
 TREATMENT_DEFAULT_LIST,
 WORKING_DIR,) = config_pvcharacterization()
 
