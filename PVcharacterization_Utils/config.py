"""
config.py docs
A custom config yaml may be provided in the two   following locations:
      ~/AppData/Roaming/Pvcharacterization.yaml  # user config
      .PVcharacterization_Utils/Pvcharacterization.yaml   # global config
If the user's Pvcharacterization.yaml configuration file exits it  will be used. Otherwise a user's configuration file Pvcharacterization.yaml 
will be created in the user's ~/AppData/Roaming.
The modification of the config variables will be stored in the Pvcharacterization.yaml stor in the user's WORRKING_DIR folder.

"""



__all__ = ['change_config_pvcharacterization',
           'get_config_dir',
           'GLOBAL',]

def get_config_dir():

    """
    Returns a parent directory path
    where persistent application data can be stored.

    # linux: ~/.local/share
    # macOS: ~/Library/Application Support
    # windows: C:/Users/<USER>/AppData/Roaming
    adapted from : https://stackoverflow.com/questions/19078969/python-getting-appdata-folder-in-a-cross-platform-way
    """
    # Standard library imports
    import sys
    from pathlib import Path
    
    home = Path.home()

    if sys.platform == 'win32':
        return home / Path('AppData/Roaming')
    elif sys.platform == 'linux':
        return home / Path('.local/share')
    elif sys.platform == 'darwin':
        return home / Path('Library/Application Support')

           
def change_config_pvcharacterization(flashtest_dir,working_dir):

    '''Saves the new flashtest_dir and working_dir in the local 
    Pvcharacterization.yaml config file
    '''

    # Standard library imports
    import os.path
    from pathlib import Path
    
    # 3rd party imports
    import yaml
    
    local_config_path = get_config_dir() / Path('Pvcharacterization.yaml')
    with open(local_config_path) as file:
        global_ = yaml.safe_load(file)
        global_['FLASHTEST_DIR'] = flashtest_dir
        global_['WORKING_DIR'] = working_dir
       
    with open(local_config_path, 'w') as file:
        outputs = yaml.dump(global_, file)

def _config_pvcharacterization():

    # Standard library imports
    import os.path
    from pathlib import Path

    # 3rd party imports
    import yaml
    
    # Reads the default PVcharacterization.yaml config file
    path_config_file = Path(__file__).parent / Path('PVcharacterization.yaml')
    date1 = os.path.getmtime(path_config_file)
    with open(path_config_file) as file:
        global_ = yaml.safe_load(file)
        
    # Overwrite if a local PVcharacterization.yaml config file otherwise create it.
         
    local_config_path = get_config_dir() / Path('Pvcharacterization.yaml')
    
    if os.path.exists(local_config_path):
        date2 = os.path.getmtime(local_config_path)
        if date2>date1:
            with open(local_config_path) as file:
                global_ = yaml.safe_load(file)
        else:
            with open(local_config_path, 'w') as file:
                outputs = yaml.dump(global_, file)
    else:
        with open(local_config_path, 'w') as file:
            outputs = yaml.dump(global_, file)
    
    
    global_['PARAM_UNIT_DIC']['Rseries'] = chr(937)
       
    return global_

GLOBAL = _config_pvcharacterization()

 
