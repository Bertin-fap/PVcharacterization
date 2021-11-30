__all__ = ['change_config_pvcharacterization', 'GLOBAL',]
           
def change_config_pvcharacterization(flashtest_dir,working_dir):

    from functools import reduce
    from pathlib import Path

    import yaml
    
    path_config_file = Path(__file__).parent / Path('PVcharacterization.yaml')
    with open(path_config_file) as file:
        GLOBAL = yaml.safe_load(file)
        GLOBAL['FLASHTEST_DIR'] = flashtest_dir
        GLOBAL['WORKING_DIR'] = working_dir
    
    with open(path_config_file, 'w') as file:
        outputs = yaml.dump(GLOBAL, file)

def _config_pvcharacterization():

    from pathlib import Path

    import yaml
    
    path_config_file = Path(__file__).parent / Path('PVcharacterization.yaml')
    with open(path_config_file) as file:
        global_ = yaml.safe_load(file)
    global_['PARAM_UNIT_DIC']['Rseries'] = chr(937)
       
    return global_


GLOBAL = _config_pvcharacterization()
 
