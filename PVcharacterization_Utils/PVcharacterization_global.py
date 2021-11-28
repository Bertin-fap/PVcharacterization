__all__ = ['change_config_pvcharacterization', 'GLOBAL',]
           
def change_config_pvcharacterization(data_folder):

    from functools import reduce
    from pathlib import Path
    

    import yaml
    

    path_config_file = Path(__file__).parent / Path('PVcharacterization.yaml')
    with open(path_config_file) as file:
        GLOBAL = yaml.safe_load(file)
        DEFAULT_DIR = Path.home()
        data_folder_list = list(Path(data_folder).parts)
        del data_folder_list[0:len(DEFAULT_DIR.parts)]
        WORKING_DIR = str(reduce(lambda a, b:Path(a) / Path(b), data_folder_list))
        GLOBAL['WORKING_DIR'] = WORKING_DIR
    
    with open(path_config_file, 'w') as file:
        outputs = yaml.dump(GLOBAL, file)

def _config_pvcharacterization():

    from pathlib import Path

    import yaml
    

    path_config_file = Path(__file__).parent / Path('PVcharacterization.yaml')
    with open(path_config_file) as file:
        GLOBAL = yaml.safe_load(file)
    GLOBAL['PARAM_UNIT_DIC']['Rseries'] = chr(937)
    GLOBAL['DEFAULT_DIR'] = Path.home()
    GLOBAL['WORKING_DIR'] = Path.home() / Path(GLOBAL['WORKING_DIR'])
    
    
    return GLOBAL
            


GLOBAL = _config_pvcharacterization()
 
