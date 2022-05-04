__all__ = [
    "parse_filename_test_control",
    "build_df_meta_test_control",
]

def parse_filename_test_control(file,warning=False):

    '''
    Let the string "file" structured as follow:
      '~/XXXXXXX_<YYMMDD>_measure<dd>.csv'
    where <> is a placeholder, d a digit, X a capital letter and ~ the relative or absolute path of the file
    
    parse_filename parses "file" in three chunks: , <YYMMDD>, measure<dd> and stores them in
    the nametuple FileInfo. In addition the extention is checked and must be .csv
    
    Args:
       file (str): filename to parse
       warning (bool): print the warning if true (default=False)
    
    Returns:
        data (namedtuple): results of the file parsing (see summary)
        
    Examples:
    let file = 'C:/Users/franc/PVcharacterization_files/QCELLS-2724_220307_mesure05.csv'
    we obtain:
        FileInfo.module_name = QCELLS-2724
        FileInfo.date = 220307
        FileInfo.mesure = "mesure05"
        FileInfo.status= True
     
    
    '''
    
    # Standard library imports
    import os
    from collections import namedtuple
    from datetime import datetime, timedelta 
    
    
    convert_to_date = lambda s: datetime.strptime(s, "%Y%m%d")
    
    FileNameInfo = namedtuple("FileNameInfo", "exp_id date exp_num module_type file_full_path status")
    
    file_full_path = file
    file = os.path.split(file)[-1]
    status=True
    
    try: # Find irradiance field
        module,date,mesure = file[:-4].split('_')
        date = convert_to_date('20'+date)
    except IndexError:
        module,date,mesure=None,None,None
        status=False
        
               
    if not status and warning:  
        print(f'Warning: the file {file}  is not a flash test format')
        
    FileInfo = FileNameInfo(
        exp_id=module,
        date=date,
        exp_num=mesure,
        module_type=module,
        file_full_path=file_full_path,
        status=status,
    )
        
    return FileInfo

def build_df_meta_test_control(list_files): 
    ''' 
    build_df_meta is the master function used to build the dataframe df_meta.
    df_meta has index= module name and columns = `exp_idx` , GLOBAL['COL_NAMES'], `Isc_corr`, `Fill_Factor_corr`, `date`,
    `exp_num`, `module_type`
    where:
        GLOBAL['COL_NAMES'] is defined in PVcharacterization_Utils.config and must contain 'Title', 'Pmax', 'Fill Factor',
    'Voc','Isc', 'Rseries', 'Rshunt', 'Vpm', 'Ipm' these values are extracted from the header of the flash test .cSV files
    
        `Isc_corr`, `Fill_Factor_corr` are the corrected values of Isc and of the fill factor
        
        `date`,`exp_num`, `module_type` are obtained by parsing the filename
    
    Args:
        list_files (list): list of files used to build the df_meta dataframe
    
    Returns:
        A dataframe containing the metadata (columns) of the list of experiences (rows)
    '''
 
    # Standard library imports 
    import os
    
    #3rd party imports
    import numpy as np
    import pandas as pd
    
    #Internal import
    from PVcharacterization_Utils.config import GLOBAL
    from PVcharacterization_Utils.PVcharacterization_flashtest import read_flashtest_file
    from PVcharacterization_Utils.PVcharacterization_flashtest import correct_iv_curve
    from PVcharacterization_Utils.PVcharacterization_flashtest import parse_filename
    
    
    COL_NAMES = GLOBAL['COL_NAMES']
    
    # Building of the dataframe df_meta out of the flashtest files 
    isc_corr = []
    fill_factor_corr = []
    list_files_name = []  # List of files basenames without extension
    list_dict_metadata = []
    list_exp_id = []
    list_date =[]
    list_measure = []
    list_module_type = []
    
    for file in list_files:
        iv_info = read_flashtest_file(file, parse_all=True)
        list_dict_metadata.append(iv_info.meta_data)
      
        
        # Compure the corrected Isc current and Fill Factor out of the I/V curves
        voltage = iv_info.IV0["Voltage"]
        current = iv_info.IV0["Current"]
        corrected_current = correct_iv_curve(voltage,current)
        isc_corr.append(np.round(corrected_current[0],3))
        fill_factor_corr.append(np.round(max(voltage*current)/(corrected_current[0]*max(voltage)),3))
        list_files_name.append(os.path.splitext(os.path.basename(file))[0])
        
        # Add exp_id, irradiance, treatment, module_type from the filename prsing 
        file_info = parse_filename_test_control(file)
        list_exp_id.append(file_info.exp_id)
        list_date.append(file_info.date)
        list_measure.append(file_info.exp_num)
        list_module_type.append(file_info.module_type)
        
        
    df_meta = pd.DataFrame.from_dict(list_dict_metadata)
    df_meta.index = list_files_name    #df_meta['ID']
    df_meta = df_meta.loc[:,COL_NAMES] # keep only the columns which names COL_NAMES 
                                       #  defined in PVcharacterization_GUI.py
    df_meta['Isc_corr'] = isc_corr
    df_meta['Fill Factor_corr'] = fill_factor_corr
    df_meta['date'] = list_date
    df_meta['exp_num'] = list_measure
    df_meta['module_type'] = list_module_type
    df_meta.insert(0, "exp_id", list_exp_id)
    
    return df_meta