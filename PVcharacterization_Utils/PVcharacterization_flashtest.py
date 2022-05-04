''' Creation: 2021.09.07
    Last update: 2021.12.01
    
    Useful functions for correctly parsing the aging data files
'''
__all__ = [
    "add_exp_to_database",
    "assess_path_folders",
    "batch_filename_correction",
    "build_files_database",
    "build_df_meta",
    "build_metadata_dataframe",
    "build_metadata_df_from_db",
    "build_modules_filenames",
    "build_modules_list",
    "correct_filename",
    "correct_iv_curve",
    "data_dashboard",
    "fit_curve",
    "parse_filename",
    "pv_flashtest_pca",
    "read_and_clean",
    "read_flashtest_file",
    "select_irradiance",
    "select_module",
]

#Internal imports 
from .config import GLOBAL
from .PVcharacterization_GUI import (select_data_dir,
                                     select_items,)
from .PVcharacterization_database import (add_files_to_database,
                                          df2sqlite,
                                          sieve_files,
                                          sqlite_to_dataframe,
                                          suppress_duplicate_database,
                                           )
                                       
def read_flashtest_file(filepath, parse_all=True):

    '''
    The function `read_flashtest_file` reads a csv file organized as follow:
    
                ==========  =================================
                Title:       HET JNHM72 6x12 M2 0200W
                Comment:     
                Op:          Util
                ID:          JINERGY3272023326035_0200W_T0
                Mod Type:    ModuleType1
                Date:        2/15/2021
                ...          ...
                Voltage:     Current:
                -0.740710    1.8377770
                -0.740387    1.8374640
                -0.734611    1.8376460
                ...          ....
                Ref Cell:   Lamp I:
                199.9875    200.0105
                199.9824    200.1674
                ...         ...
                Voltage1:   Current1:
                -0.740710   1.8377770
                -0.740387   1.8374640
                -0.734611   1.8376460
                ...         ....
                Ref Cell1:  Lamp I1:
                ...         ....
                Voltage2:   Current2:
                -0.740710   1.8377770
                -0.740387   1.8374640
                -0.734611   1.8376460
                ...         ....
                Ref Cell2:  Lamp I2:
                0.008593    1.823402
                0.043122    1.823085
                ...         ....
                DarkRsh:    0
                DarkV:       ark I:
                ==========  =================================
    
    The `.csv` file is parsed in a namedtuple `data` where:
       
       - data.IV0, data.IV1, data.IV2 are dataframes containing the `IV` curves as :
       
                ======== ==========
                Voltage	 Current
                ======== ==========
                0.008593  1.823402
                0.043122  1.823085
                0.070891  1.823253
                xxxx      xxxx
                50.0      1.823253
                ======== ==========
       - data.Ref_Cell0, data.Ref_Cell1, data.Ref_Cell2 are dataframes containing
       the irradiance curves as:
       
                ======== ==========
                Ref_Cell  Lamp_I
                ======== ==========
                199.9875  200.0105
                199.9824  200.1674
                xxxxx     xxxxx
                199.9824  200.0074
                ======== ==========
       - data.meta_data is a dict containing the header :
    .. code-block:: python 
    
      data.meta_data = {
      "Title":"HET JNHM72 6x12 M2 0200W",
      "Comment":"",
      "Op":"Util",
      .... :.....,
      }
      
    Args:
        filename (Path): name of the .csv file
    
    Returns:
        data (namedtuple): results of the file parsing (for details see above)
        
    Note:
    Amended 28/04/2022 to comply the new file format of sofware version 5.5.5. Add 
    the dataframe  data.IV.raw to the name tuple. 
    
    '''

    # Standard library imports
    from collections import namedtuple

    # 3rd party imports
    import numpy as np
    import pandas as pd
    
    ENCODING = GLOBAL['ENCODING']

    data_struct = namedtuple(
        "PV_module_test",
        ["meta_data", "IV0", "IV1", "IV2", "Ref_Cell0", "Ref_Cell1", "Ref_Cell2","IV_raw"],
    )
    # For significance of -1.#IND see:
    #https://stackoverflow.com/questions/347920/what-do-1-inf00-1-ind00-and-1-ind-mean#:~:text=This%20specifically%20means%20a%20non-zero%20number%20divided%20by,1%29%20sqrt%20or%20log%20of%20a%20negative%20number
  
    df_data = pd.read_csv(filepath,
                          sep=",",
                          skiprows=0,
                          header=None,
                          na_values=[' -1.#IND ',' -1.#IND'], # Takes care of " -1.#IND " values
                          keep_default_na=False,
                          encoding=ENCODING) # encoding = latin-1 by default to avoid 
                                             # trouble with u'\xe9' with utf-8
    df_data = df_data.dropna()
    # Builds the list (ndarray) of the index of the beginnig of the data blocks (I/V and Ref cell) 
    
    index_data_header = np.where(
                                 df_data.iloc[:, 0].str.contains(
                                                    '^ Volt| Raw Voltage|Ref Cell',  # Find the indice of the
                                                    case=True,                       # headers of the IV curve
                                                    regex=True,
                                                               )
                                )[0]  # Ref Cell data

    index_data_header = np.insert(
        index_data_header,            
        [0],       # add index 0 for the beginning of the header
        [0]        
    )
    
    # Builds the meta data dict meta_data {label:value}
    meta_data = {}
    meta_data_df = df_data.iloc[np.r_[index_data_header[0] : index_data_header[1]]] 
    for key,val in dict(zip(meta_data_df[0], meta_data_df[1])).items():
        try:
            meta_data[key.split(":")[0]] = float(val)
        except:
            meta_data[key.split(":")[0]] = val
            
    skip_lines = 0 if 'Soft Ver' in meta_data.keys() else 3
    index_data_header = np.insert(
        index_data_header,            
        [len(index_data_header)],  
        [len(df_data) - skip_lines],        # add index of the last numerical value
    )
    
    # Extract I/V curves and Ref_cell curves
    if not parse_all:
        data = data_struct(
            meta_data=meta_data,
            IV0=None,
            IV1=None,
            IV2=None,
            Ref_Cell0=None,
            Ref_Cell1=None,
            Ref_Cell2=None,
            IV_raw=None,
        )
        return data

    list_df = []
    for i in range(1, len(index_data_header) - 1):
        dg = df_data.iloc[
            np.r_[index_data_header[i] + 1 : index_data_header[i + 1]]
        ].astype(float)

        dg = dg.loc[dg[0] > 0] # Keep only positive voltage values (physically meaningfull values)
        dg.index = list(range(len(dg)))

        if "Voltage" in df_data.iloc[index_data_header[i]][0]:
            dg.columns = ["Voltage", "Current"]
        else:
            dg.columns = ["Ref_Cell", "Lamp_I"]

        list_df.append(dg)
    if len(list_df)==6:
        data = data_struct(
            meta_data=meta_data,
            IV0=list_df[0],
            IV1=list_df[2],
            IV2=list_df[4],
            Ref_Cell0=list_df[1],
            Ref_Cell1=list_df[3],
            Ref_Cell2=list_df[5],
            IV_raw=None,
    )
    else: # for soft ver 5.5.5 add a new dataframe IV_raw
        data = data_struct(
            meta_data=meta_data,
            IV0=list_df[0],
            IV1=list_df[2],
            IV2=list_df[4],
            Ref_Cell0=list_df[1],
            Ref_Cell1=list_df[3],
            Ref_Cell2=list_df[5],
            IV_raw=list_df[6],
    )       
        
    return data

def parse_filename(file,warning=False):

    '''
    Let the string "file" structured as follow:
      '~/XXXXXXX<ddddddddddddd>_<dddd>W_T<d>.csv'
    where <> is a placeholder, d a digit, X a capital letter and ~ the relative or absolute path of the file
    
    parse_filename parses "file" in three chunks: JINERGY<ddddddddddddd>, <dddd>, T<d> and stores them in
    the nametuple FileInfo. In addition the extention is checked and must be .csv
    
    Args:
       file (str): filename to parse
       warning (bool): print the warning if true (default=False)
    
    Returns:
        data (namedtuple): results of the file parsing (see summary)
        
    Examples:
    let file = 'C:/Users/franc/PVcharacterization_files/JINERGY3272023326035_0200W_T2.csv'
    we obtain:
        FileInfo.power = 200
        FileInfo.treatment = "T2"
        FileInfo.time = "JINERGY3272023326035"
        status= True
     
    
    '''
    
    # Standard library imports
    from collections import namedtuple
    import os
    import re
    
    FileNameInfo = namedtuple("FileNameInfo", "exp_id irradiance treatment module_type file_full_path status")
    re_irradiance = re.compile(r"(?<=\_)\d{3,4}(?=W\_)")
    re_treatment = re.compile(r"(?<=\_)T\d{1}(?=\.csv)")
    #re_module_type = re.compile(r"[a-zA-Z\-#0-9]*\d{1,50}(?=\_)")
    re_module_type = re.compile(r"[a-zA-Z\-#0-9]*(?=\_)")
    
    file_full_path = file
    file = os.path.split(file)[-1]
    status=True
    
    try: # Find irradiance field
        irradiance=int(re.findall(re_irradiance, file)[0])
    except IndexError:
        irradiance=None
        status=False
        
    try: # Find treatment field
        treatment=re.findall(re_treatment, file)[0]
    except IndexError:
        treatment=None
        status=False
        
    try: # Find module type field
        module_type=re.findall(re_module_type, file)[0]
    except IndexError:
        module_type=None
        status=False
               
    if not status and warning:  
        print(f'Warning: the file {file}  is not a flash test format')
        
    FileInfo = FileNameInfo(
        exp_id=f'{module_type}_{irradiance}W_{treatment}',
        irradiance=irradiance ,
        treatment=treatment,
        module_type=module_type,
        file_full_path=file_full_path,
        status=status,
    )
        
    return FileInfo

def assess_path_folders(path_root=None):
    
    '''
    Interactivelly sets the path to the working folder
    
    Args:
      path_root (str): if none the root for the interactive selection is the user path home otherwise path_root
      
    Returns:
      The data folder name.
    '''
    
    # Standard library imports
    from pathlib import Path
    
    # Local imports
    from .PVcharacterization_sys import DISPLAYS
    
    if path_root is None:
        root = Path.home()
    else:
        root = path_root
        
    # Set the GUI display
    gui_disp = [i for i in range(len(DISPLAYS)) if DISPLAYS[i]['is_primary']][0]
    # Get the prime display choice
    # TO DO: replace input by a GUI to select the activ_display
    disp_select = input('Select Id of gui prime-display '+
                   '(value: 0 to '+ str(len(DISPLAYS)-1)+
                  '; default:'+ str(gui_disp)+')')
    if disp_select: gui_disp = int(disp_select)
        
    # Setting the GUI titles
    gui_titles = {'main':   'Folder selection window',
                  'result': 'Selected folder'}
    gui_buttons = ['SELECTION','HELP']

    working_dir = select_data_dir(root,gui_titles,gui_buttons,gui_disp)  # Selection of the root folder
    
    return working_dir

def build_files_database(db_folder,ft_folder,verbose=True):
    ''' 
    Build the table DATA_BASE_TABLE_FILE in the data base DATA_BASE_NAME with the following fields
    irradiance, treatment, module_type, file_full_path.
    
    Args:
       db_folder (path):  path  of the folder containing the database.
       ft_folder (path):  path  of the folder containing the experimental flashtest files.
        
    Returns:
        (data frame) dataframe df_files_descp with the following columns: irradiance, treatment, module_type, 
        file_full_path.
    '''

    # Standard library imports
    from collections import Counter
    import os
    from pathlib import Path

    # 3rd party import
    import pandas as pd
    
    DATA_BASE_NAME = GLOBAL['DATA_BASE_NAME']
    DATA_BASE_TABLE_FILE = GLOBAL['DATA_BASE_TABLE_FILE']
    
    datafiles_list = list(Path(ft_folder).rglob("*.csv")) # Recursive collection all the .csv lies
    
    if not datafiles_list:
        raise Exception(f"No .csv files detected in {ft_folder} and sub folders")
        
    list_files_descp=[]
    for file in datafiles_list:
        fileinfo = parse_filename(str(file),warning=True)  
        if fileinfo.status:
            list_files_descp.append(fileinfo)
        else:
            pass
        
    file_check = True  # Check for the multi occurrences of a file
    list_multi_file = []
    for file,frequency in Counter([os.path.basename(x) for x in datafiles_list]).items(): # Check the the uniqueness of a file name
        if frequency>1:
            list_multi_file.append(file)
            file_check = False
    if not file_check:
        print(f"WARNING: the file(s) {' ,'.join(list_multi_file)} has(have) a number of occurrence(s) greater than 1.\nOnly the first occurrence(s) will be retained.\n")


    df_files_descp  = pd.DataFrame(list_files_descp) # Build the database
    #df_files_descp = df_files_descp.drop_duplicates('exp_id') # we can drop duplicates in the database

    database_path = Path(db_folder) / Path(DATA_BASE_NAME)

    df2sqlite(df_files_descp.drop('status',axis=1), path_db=database_path, tbl_name=DATA_BASE_TABLE_FILE)
    suppress_duplicate_database(db_folder)
    
    if verbose:
        print(f'{len(datafiles_list)} flash test files detected.\n{len(list_multi_file)} duplicates suppressed\nThe database table {DATA_BASE_TABLE_FILE} in {database_path} is built\n\n')
    
    return #df_files_descp


def build_metadata_dataframe(working_dir,interactive=False):

    '''
    Args:
       working_dir (path): path of the folder holding the database
       interactive (boolean): if True select interactivelly the modules otherwise takes all the modules
    '''

    DATA_BASE_TABLE_FILE = GLOBAL['DATA_BASE_TABLE_FILE']
    
    df_files_descp = sqlite_to_dataframe(working_dir, DATA_BASE_TABLE_FILE)
    if interactive:
        # Interactive selection of the modules
        list_mod_selected = build_modules_list(df_files_descp)                                

    else:
        list_mod_selected = df_files_descp['module_type'].unique()
        
    # Extraction from the file database all the filenames related to the selected modules
    list_files_path = build_modules_filenames(list_mod_selected,working_dir)
    df_meta = _build_metadata_dataframe(list_files_path,working_dir)
    
    return df_meta
        

def _build_metadata_dataframe(list_files_path, working_dir):

    '''Building of the dataframe df_meta out of the interactivelly selected module type.
    The df_meta index are the file names without extention (ex: QCELLS901219162417702718_0200W_T0).
    The df_meta columns are: Title, Pmax, Fill Factor, Voc, Isc, Rseries, Rshunt,
    Vpm, Ipm, Isc_corr, Fill Factor_corr, irradiance, treatment, module_type.
    Isc_corr, Fill Factor_corr are respesctivelly the correcteted short circuit current
    and fill factor.

    Args:
        df_files_descp (dataframe): dataframe built by the function build_files_database 
        working_dir (path):  path  of the folder containing the database.

    Returns:
        (dataframe)  : dataframe of the experimental data  
    '''
    
    # Standard library imports
    import os
    from pathlib import Path

    #3rd party imports
    import pandas as pd
    
    DATA_BASE_NAME = GLOBAL['DATA_BASE_NAME']
    DATA_BASE_TABLE_EXP = GLOBAL['DATA_BASE_TABLE_EXP']

    df_meta = build_df_meta(list_files_path)

    # Builds a database
    database_path = Path(working_dir) / Path(DATA_BASE_NAME)
    df2sqlite(df_meta, path_db=database_path, tbl_name=DATA_BASE_TABLE_EXP)
    
    return df_meta


def build_metadata_df_from_db(working_dir,list_mod_selected,list_irradiance):

    '''
    Args:
        working_dir (str): full path of the folder containing the database.
        mode (list): if None select interactivelly the list of module types, otherwise takes all the module type.
   
    '''
    
    #3rd party imports
    import pandas as pd
    
    DATA_BASE_TABLE_FILE = GLOBAL['DATA_BASE_TABLE_FILE']
    DATA_BASE_TABLE_EXP = GLOBAL['DATA_BASE_TABLE_EXP']
    

    # Extraction from the file database all the filenames related to the selected modules
    df_meta = sqlite_to_dataframe(working_dir,DATA_BASE_TABLE_EXP)
    df_meta = df_meta.query('module_type in @list_mod_selected')
    
    df_meta = df_meta.query('irradiance in @list_irradiance')
    
    return df_meta

    
def select_module(working_dir,mode=None):
    
    '''Module selection if mode=None we interactively choose the modules otherwise, we select all the modules
    '''
    df_files_descp = sqlite_to_dataframe(working_dir,GLOBAL['DATA_BASE_TABLE_FILE'])

    if mode is None:    
        list_mod_selected = build_modules_list(df_files_descp)
    else:
        list_mod_selected = df_files_descp['module_type'].unique()
        
    return list_mod_selected

def select_irradiance(working_dir,list_mod_selected, mode=None):
    
    '''Module selection if mode=None we interactively choose the irradiances otherwise, we select all the irradiances
    '''
    df_files_descp = sqlite_to_dataframe(working_dir,GLOBAL['DATA_BASE_TABLE_FILE'])
    list_all_irradiance = df_files_descp.query('module_type in @list_mod_selected').irradiance.unique()

    if mode is None:    
        list_irradiance = list_all_irradiance
    else:
        list_irradiance =  select_items(list_all_irradiance,
                                           'Select the irradiance',
                                           mode = 'multiple')
        list_irradiance = [int(irradiance) for irradiance in list_irradiance]
        
    return list_irradiance
    
def build_modules_list(df_files_descp):

    '''Interactive selection of modules type out of the dataframe df_meta.
   
    Args:
        df_files_descp (dataframe): dataframe built by the function build_files_database 
       
    Returns:
        list_mod_selected (list os str): list of selected modules type.
    '''
    

    # Interactive selection of the modules
    list_modules_type = df_files_descp['module_type'].unique()
    list_mod_selected = select_items(list_modules_type,
                                'Select the modules type',
                                mode = 'multiple') 
    
    
    return list_mod_selected
    
def build_modules_filenames(list_mod_selected,working_dir):

    '''Builds out of the modules type list_mod_selected the list of all filename
    related to these modules.

    Args:
        list_mod_selected (list of str):  list of module names.
        working_dir (str): full path of the folder containing the database.
        
    Returns
        list_files_path (list of str): list of the path of the files related to the modules type
                                    defined in the list list_mod_selected.
    '''
    
    # Standard library imports
    import os
    from pathlib import Path   

    DATA_BASE_NAME = GLOBAL['DATA_BASE_NAME'] 
    IRRADIANCE_DEFAULT_LIST = GLOBAL['IRRADIANCE_DEFAULT_LIST']  
    TREATMENT_DEFAULT_LIST = GLOBAL['TREATMENT_DEFAULT_LIST']    
    
    # Extract from the file database all the filenames related to the selected modules
    database_path = Path(working_dir) / Path(DATA_BASE_NAME)
    list_files_path = sieve_files(IRRADIANCE_DEFAULT_LIST,
                             TREATMENT_DEFAULT_LIST,
                             list_mod_selected,
                             database_path)
    return list_files_path
    


def pv_flashtest_pca(df_meta,scree_plot = False,interactive_plot=False):
    
    '''PCA analysis of the flashtest data. The features are the parameters defined in the global COL_NAMES and the data vectors the
    rows of the dataframe df_data.
    '''

    # 3rd party imports
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import seaborn as sns
    
    COL_NAMES = GLOBAL['COL_NAMES']
    
    def title(M_items_per_row = 3 ):
        '''Builds th PCA plot title out of the dict dict_module_type with M_items_per_row module type name
        per row'''    
        nl = '\n'      
        list_mod = [f'{value} : {key}' for key,value in dict_module_type['module_type'].items()]
        list_mod = [', '.join( list_mod[idx:idx+M_items_per_row]) for idx in range(0,len(list_mod),M_items_per_row)]
        text = f"Module names are:{nl}{nl.join(list_mod)}"
        return text
    
    list_params = COL_NAMES.copy()
    list_params = [x for x in list_params if x not in ['Title','Fill Factor','Isc','exp_id']] + ['Fill Factor_corr','Isc_corr']
    X = df_meta[list_params].to_numpy()
    X = X-X.mean(axis=0)
    X = X/np.std(X, axis=0)


    Cor = np.dot(X.T,X) # Build a square symetric correlation matrix

    lbd,Eigen_vec = np.linalg.eig(Cor) # Compute the eigenvalues and eigenvectors

    # Sort by decreasing value of eigenvalues.
    w = sorted(list(zip(lbd,Eigen_vec.T)), key=lambda tup: tup[0],reverse=True)
    vp = np.array([x[0] for x in w ])
    L = np.array([x[1] for x in w]).reshape(np.shape(Eigen_vec)).T

    F = np.real(np.matmul(X,L))
 
    x = -F[:,0]
    y = F[:,1]


    
    # Conditional plot the scree plot.
    if scree_plot:
        labels=['PC'+str(x) for x in range(1,len(vp)+1)]

        plt.figure()
        plt.bar(x=range(1,len(lbd)+1), height=np.cumsum(100*vp/sum(vp)), tick_label=labels)
    
    
    # Plot the PCA
    # Builds the df_meta_pca dataframe
    df_meta_pca = df_meta[['irradiance','treatment','module_type']].copy()
    df_meta_pca['x'] = x
    df_meta_pca['y'] = y
    
    if not interactive_plot:
        dict_module_type = {"module_type": {x:i for i,x in enumerate(df_meta_pca['module_type'].unique())}}
        df_meta_pca.replace(dict_module_type,inplace=True)
        label = df_meta_pca["module_type"]
        fig, ax = plt.subplots(figsize=(10,10))
        p = sns.scatterplot(data=df_meta_pca,
                            x='x',
                            y='y',
                            hue='irradiance',
                            style='treatment', palette='tab10', ax=ax, s=50)

        for lbl,xp,yp in zip(label,x,y):
                plt.annotate(str(lbl),(xp+0.05,yp+0.05))

        # Builds and plot the title and the x,y labels

        plt.title(title())
        plt.xlabel('PC1 _ {0}%'.format(np.rint(100*vp[0]/sum(vp) )))
        plt.ylabel('PC2 _ {0}%'.format(np.rint(100*vp[1]/sum(vp)) ))
        _ = p.legend(bbox_to_anchor=(1, 1.02), loc='upper left')
    else:
        df_meta_pca['exp. conditions'] = df_meta_pca.apply(lambda x : f'irradiance: {x[0]}, treatment: {x[1]}',axis=1)
        fig = px.scatter(df_meta_pca,
                         x="x",
                         y="y",
                         color="module_type",
                         symbol="treatment",
                         hover_data=['exp. conditions'],
                         labels={'x':'PC1 _ {0}%'.format(np.rint(100*vp[0]/sum(vp) )),
                                 'y':'PC2 _ {0}%'.format(np.rint(100*vp[1]/sum(vp)) )})

        fig.update_traces(marker=dict(size=12,
                          line=dict(width=2,
                          color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.show()
        
        
    return df_meta_pca
    
def data_dashboard(working_dir,list_params):

    '''
    Args:
        working_dir (str): full path of the folder containing the database.
        list_params (list): list of parameters to be processed.
        
    Returns:
        A dataframe containing the dashboard
    '''
    
    # Standard library imports
    from pathlib import Path
    
    #3rd party imports
    import pandas as pd

    list_mod_selected = select_module(working_dir)
    list_irradiance = select_irradiance(working_dir,list_mod_selected,mode='select')
    df_meta = build_metadata_df_from_db(working_dir,list_mod_selected,list_irradiance)
    df_meta_dashboard = df_meta.pivot(values= list_params,index=['module_type','treatment',],
                                      columns=['irradiance',]) 
    df_meta_dashboard.to_excel(working_dir/Path('exp_summary.xlsx'))
    
    print(f'The file {str(working_dir/Path("exp_summary.xlsx"))} has been created')
    
    return df_meta_dashboard
    
def correct_filename(filename,new_moduletype_name):
    
    '''Correction of the filename with a new module type name.
    ex. : filename = 'C:/Users/franc/PVcharacterization_files/JINERGY26039_200W_T2.csv'
          new_moduletype_name = 'JINERGY3272023326039'
          corrected_filename = 'C:/Users/franc/PVcharacterization_files\JINERGY3272023326039_0200W_T2.csv'
    The field irradiance, if necessary, is also upgraded  by addind leading zeros to obtain a four digits number.
    
    Args:
      filename (str): the full path of the file name to be corrected
      new_moduletype_name (str): the new module name
      
     Returns:
       The corrected full path name.
          
    '''

    # Standard library imports
    import os
    from pathlib import Path
    
    FileInfo = parse_filename(filename)
    new_file_name = f'{new_moduletype_name}_{FileInfo.irradiance:04d}W_{FileInfo.treatment}.csv'
    corrected_filename = os.path.join(os.path.dirname(filename), new_file_name)
    
    return corrected_filename


def batch_filename_correction(working_dir, verbose=False):

    ''' The `batch_filename_correction` function corrects the wrong file names 
    by replacing the wrong module type names with the longest one supposed to be the correct one
    and adding the missing leading zeros  in the irradiance field.
    
    Args:
        working_dir (path): folder full path of the experimental files.
        verbose (bool): allows printings if True (default: False).
        
    Returns:
        `(str)`: status of the function run.   

    ''' 

    # Standard library imports
    import os
    
    DATA_BASE_TABLE_FILE = GLOBAL['DATA_BASE_TABLE_FILE']

    # Get dataframe describing the experimental files
    #df_files_descp = build_files_database(working_dir, verbose=False)
    df_files_descp = sqlite_to_dataframe(working_dir,DATA_BASE_TABLE_FILE)
   
    # Select the module types which names has to be corrected
    list_mod_selected = build_modules_list(df_files_descp)

    # Choose the longest module type name
    list_mod_selected =sorted(list_mod_selected, key=len, reverse=True) # Descending sort by length of items
    new_moduletype_name = list_mod_selected[0] # Select the first items of the sorted list which corresponds 
                                               # to the longest module name

    # Picks in the database all the filenames related to the selected module except the correct first one
    list_wrong_files_path = build_modules_filenames(list_mod_selected[1:],working_dir)

    for filename in list_wrong_files_path: # Modifies the incorrect filenames
        corrected_filename = correct_filename(filename,new_moduletype_name)
        if verbose :
            print(f'old name: {filename}')
            print(f'new name: {corrected_filename}')
            print()
        os.rename(filename,corrected_filename)
    
    status = 'Correction done on :'+ ', '.join(list_mod_selected[1:]) + '\nnew name: ' + new_moduletype_name
    return status 

def correct_iv_curve(voltage,current):
    
    '''Correct improper values of the iv curve for low voltage.
    Method: we fit iv curve between min_voltage_fit (5 V) and max_voltage_fit (20 V)
    by a polynomial of order 1 and extrapolate its values for voltage > min_voltage_fit.
    
    Args:
       voltage (list): list of voltage of the IV curve.
       current (list): list of current of the IV curve.
       
    Returns:
       (list) corrected current.
    '''
    
    # Standard library imports
    import bisect
    
    # 3rd party imports
    import numpy as np

    min_voltage_fit = 5   # in Volt
    max_voltage_fit = 20  # in A
    error_max = 0.3       # in percent
    
    voltage_idx_min = bisect.bisect_left(voltage, 5, lo=0, hi=len(voltage))
    voltage_idx_max = bisect.bisect_left(voltage, 25, lo=0, hi=len(voltage))
    current_fit = current[voltage_idx_min:voltage_idx_max]
    voltage_fit = voltage[voltage_idx_min:voltage_idx_max]
    
    polynomial_coeff = np.polyfit(voltage_fit ,current_fit,1)

    ynew = np.poly1d(polynomial_coeff)
    
    current_corrected = [y if 100*(y-yfit)/y< error_max else yfit for y,yfit 
               in zip(current[0:voltage_idx_max],ynew(voltage[0:voltage_idx_max]))] + list(current[voltage_idx_max:])
    

    return current_corrected

def read_and_clean(file):
    
    '''
    The read_and_clean function reads an .xlsx file organized as flow:

    ======= ===== ====== ====== =====  ===== ====== ======
                   500h  1000h  1500h  2000h  2500h 3000h
    Hetna   XDH    -6%    -12%   -22%  -37%     
            DH            -6%          -15%         -30%
    Q cells XDH    -5%    -5%     50%  -6%          -8%
            DH            -5%    -6,    50%         -8%
    Jinergy XDH    -1%     50%   -2%   -3%          -5%
            DH     -2%    -2%     50%  -2%           50%
    ======= ===== ====== ====== =====  ===== ====== ======
    
    The data are cleaned:
    
    - the missing values of the column are filled as follow Hetna,Hetna,Qcell,Qcell,...
    - for each label Hetna-XDH, Hetna-DH,... the x_clean, y_clean lists are built by retaining
    only the (x_clean, y_clean) tuples where y_clean is not an nan.
    
    Args:
       file (Path): absolute na of the .xlsx file
       
    Returns:
       dic_values (dict): dictionary keyed by label of the named tuples:
                          data_struct.x containing the x clean values
                          data_struct.y containing the y clean values
    
    Examples:
       {'Hetna-XDH':([500,1000,1500,2000],[-6,-12,-22,-37]),
       'Hetna-DH':([1000,2000,3000],[-6,-15,-30]),
       ...}
                          
    '''
    
    # Standard library imports
    from collections import namedtuple
    import re
  
    # 3rd party imports
    import numpy as np
    import pandas as pd
    
    re_col_name = re.compile('[0-9]{1,4}')

    dic_values = {}
    data_struct = namedtuple("x_y",["x", "y",])
    
    # Read the excel file and rename the columns
    df = pd.read_excel(file)
    col_names = {df.columns[0]:"module",
                 df.columns[1]:"experiment"}
    df.rename(columns=col_names,inplace=True)
    col_names.update({x: re_col_name.findall(str(x))[0]+'h' for x in df.columns if  re_col_name.findall(str(x))})
    df.rename(columns=col_names,inplace=True)

    # Takes care of missing values in the "module" column
    titre1_corrige = []
    list_titre = df['module'].tolist()
    for i, titre in enumerate(list_titre): 
        if isinstance(titre,float): # Convoluted check for missing value
            titre1_corrige.append(list_titre[i-1])
        else:
            titre1_corrige.append(list_titre[i])      
    df['module'] = titre1_corrige
    
    
    # Takes care of nan y_i data value by skipping (x_i,y_i) tuples when y_i in nan
    x = [float(val_col[0:-1]) for val_col in df.columns if 'h' in val_col] # Built the time list
    
    for index_row in df.index:
        label = df.iloc[index_row,0]+ '-' + df.iloc[index_row,1]
        y = df.iloc[index_row,np.r_[2:len(df.columns)]].tolist()
        x_clean = []
        y_clean = []
        for x1, y1, test in zip(x,y,np.isnan(y)):
            if not test:
                x_clean.append(x1)
                y_clean.append(y1*100)

        dic_values[label] = data_struct(x_clean,y_clean)
    
    return dic_values

def fit_curve(x,y,order=2,n_fit=200):
    
    '''
    The function fit_curve fit the set of tuples (x_i,y_i) by a polynom of order ORDER.
    
    Args:
       x (ndarray): list of absissa
       y (ndarray): list of ordinate
       dic_coef (dict): dict kayed by label of the fitting coefficients (use mutability)
       order (int): order of the fitting polynomial
       n_fit (int): number of points to plot using the fitting polynomial
       
    Returns:
       (x_fit,y_fit) (tuple of ndarrays): x_fit list of the n_fit fitting absissa
                                          x_fit list of the n_fit fitting ordinate
       
    '''
    
    # 3rd party imports
    import numpy as np
    
    assert len(x)>= order+1, f'Cannot fit {len(x)} with a polynomial of order {ORDER}'
    
    x_fit = np.linspace(min(x),max(x),n_fit)
    poly_coef = np.polyfit(x, y, order)
    p = np.poly1d(poly_coef)
    y_fit = p(x_fit)
    return (x_fit,y_fit,poly_coef)

def add_exp_to_database(working_dir, new_data_folder):

    '''
     Args:
        working_dir (str): full path of the folder containing the database.
        new_data_folder (str): full path of the folder containing the experiences to be added to the database.
    '''
    
    # Standard library imports 
    import os 
    from pathlib import Path

    #3rd party imports
    import pandas as pd
    
    DATA_BASE_NAME = GLOBAL['DATA_BASE_NAME']
    DATA_BASE_TABLE_FILE = GLOBAL['DATA_BASE_TABLE_FILE']
    DATA_BASE_TABLE_EXP = GLOBAL['DATA_BASE_TABLE_EXP']

    df_files_descp = sqlite_to_dataframe(working_dir, DATA_BASE_TABLE_FILE)
    files_before_add = df_files_descp['file_full_path']

    files = [os.path.join(new_data_folder,file) for file in os.listdir(new_data_folder) if 
             Path(file).suffix=='.csv']

    add_files_to_database(files,working_dir)
    suppress_duplicate_database(working_dir)
    df_files_descp = sqlite_to_dataframe(working_dir, DATA_BASE_TABLE_FILE)
    files_after_add = df_files_descp['file_full_path']
    
    added_files = list(set(files_after_add) - set(files_before_add))
    if added_files:
        x = "\n"
        print(f'the following {len(added_files)} files has been added :\n {x.join(added_files)}')
        df_meta = build_df_meta(added_files)
        df_meta_concat = pd.concat([sqlite_to_dataframe(working_dir,DATA_BASE_TABLE_EXP),df_meta],ignore_index=True)
        # Builds a database
        database_path = Path(working_dir) / Path(DATA_BASE_NAME)
        df2sqlite(df_meta_concat, path_db=database_path, tbl_name=DATA_BASE_TABLE_EXP)
    else:
        print('The database is already up to date. No file has been added.')

def build_df_meta(list_files): 
    ''' 
    build_df_meta is the master function used to build the dataframe df_meta.
    df_meta has index= module name and columns = `exp_idx` , GLOBAL['COL_NAMES'], `Isc_corr`, `Fill_Factor_corr`, `ìrradiance`,
    `treatment`, `module_type`
    where:
        GLOBAL['COL_NAMES'] is defined in PVcharacterization_Utils.config and must contain 'Title', 'Pmax', 'Fill Factor',
    'Voc','Isc', 'Rseries', 'Rshunt', 'Vpm', 'Ipm' these values are extracted from the header of the flash test .cSV files
    
        `Isc_corr`, `Fill_Factor_corr` are the corrected values of Isc and of the fill factor
        
        `ìrradiance`,`treatment`, `module_type` are obtained by parsing the filename
    
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
    
    COL_NAMES = GLOBAL['COL_NAMES']
    
    # Building of the dataframe df_meta out of the flashtest files 
    isc_corr = []
    fill_factor_corr = []
    list_files_name = []  # List of files basenames without extension
    list_dict_metadata = []
    list_exp_id = []
    list_irradiance =[]
    list_treatment = []
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
        file_info = parse_filename(file)
        list_exp_id.append(file_info.exp_id)
        list_irradiance.append(file_info.irradiance)
        list_treatment.append(file_info.treatment)
        list_module_type.append(file_info.module_type)
        
        
    df_meta = pd.DataFrame.from_dict(list_dict_metadata)
    df_meta.index = list_files_name    #df_meta['ID']
    df_meta = df_meta.loc[:,COL_NAMES] # keep only the columns which names COL_NAMES 
                                       #  defined in PVcharacterization_GUI.py
    df_meta['Isc_corr'] = isc_corr
    df_meta['Fill Factor_corr'] = fill_factor_corr
    df_meta['irradiance'] = list_irradiance
    df_meta['treatment'] = list_treatment
    df_meta['module_type'] = list_module_type
    df_meta.insert(0, "exp_id", list_exp_id)
    
    return df_meta