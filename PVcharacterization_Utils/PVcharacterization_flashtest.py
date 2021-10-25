""" Creation: 2021.09.07
    Last update: 2021.10.21
    
    Useful functions for correctly parsing the aging data files
"""
__all__ = [
    "assess_path_folders",
    "build_files_database",
    "build_metadata_dataframe",
    "build_modules_filenames",
    "build_modules_list",
    "correct_filename",
    "data_dashboard",
    "df2sqlite",
    "parse_filename",
    "pv_flashtest_pca",
    "read_flashtest_file",
    "sieve_files",
]

#Internal imports 
from .PVcharacterization_global import (COL_NAMES,
                                        DATA_BASE_NAME,
                                        DATA_BASE_TABLE_FILE,
                                        DATA_BASE_TABLE_EXP,
                                        DEFAULT_DIR,
                                        IRRADIANCE_DEFAULT_LIST,
                                        PARAM_UNIT_DIC,
                                        TREATMENT_DEFAULT_LIST,)
from .PVcharacterization_GUI import (select_data_dir,
                                     select_items,)
                                       
def read_flashtest_file(filepath, parse_all=True):

    """
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
                ...          ....
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
        data (namedtuple): results of the file parsing (see summary)
    
    """

    # Standard library imports
    from collections import namedtuple

    # 3rd party imports
    import numpy as np
    import pandas as pd

    data_struct = namedtuple(
        "PV_module_test",
        ["meta_data", "IV0", "IV1", "IV2", "Ref_Cell0", "Ref_Cell1", "Ref_Cell2"],
    )

    df_data = pd.read_csv(filepath, sep=",", skiprows=0, header=None)
    
    # Builds the list (ndarray) of the index of the beginnig of the data blocks (I/V and Ref cell) 
    index_data_header = np.where(
        df_data.iloc[:, 0].str.contains(
            "^ Volt|Ref Cell",  # Find the indice of the
            case=True,          # headers of the IV curve
            regex=True,
        )
    )[
        0
    ]  # Ref Cell data

    index_data_header = np.insert(
        index_data_header,            
        [0, len(index_data_header)],  # add index 0 for the beginning of the header
        [0, len(df_data) - 3],        # add index of the last numerical value
    )
    
    # Builds the meta data dict meta_data {label:value}
    meta_data = {}
    meta_data_df = df_data.iloc[np.r_[index_data_header[0] : index_data_header[1]]] 
    for key,val in dict(zip(meta_data_df[0], meta_data_df[1])).items():
        try:
            meta_data[key.split(":")[0]] = float(val)
        except:
            meta_data[key.split(":")[0]] = val
    
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
        )
        return data

    list_df = []
    for i in range(1, len(index_data_header) - 1):
        dg = df_data.iloc[
            np.r_[index_data_header[i] + 1 : index_data_header[i + 1]]
        ].astype(float)

        dg = dg.loc[dg[0] > 0]
        dg.index = list(range(len(dg)))

        if "Voltage" in df_data.iloc[index_data_header[i]][0]:
            dg.columns = ["Voltage", "Current"]
        else:
            dg.columns = ["Ref_Cell", "Lamp_I"]

        list_df.append(dg)

    data = data_struct(
        meta_data=meta_data,
        IV0=list_df[0],
        IV1=list_df[2],
        IV2=list_df[4],
        Ref_Cell0=list_df[1],
        Ref_Cell1=list_df[3],
        Ref_Cell2=list_df[0],
    )
    return data

def parse_filename(file):

    """
    Let the string "file" structured as follow:
      '~/XXXXXXX<ddddddddddddd>_<dddd>W_T<d>.csv'
    where <> is a placeholder, d a digit, X a capital letter and ~ the relative or absolute path of the file
    
    parse_filename parses "file" in three chunks: JINERGY<ddddddddddddd>, <dddd>, T<d> and stores them in
    the nametuple FileInfo.
    
    Args:
       file (str): filename to parse
    
    Returns:
        data (namedtuple): results of the file parsing (see summary)
        
    Examples:
    let file = 'C:/Users/franc/PVcharacterization_files/JINERGY3272023326035_0200W_T2.csv'
    we obtain:
        FileInfo.power = 200
        FileInfo.treatment = "T2"
        FileInfo.time = "JINERGY3272023326035"
    
    """
    # Standard library imports
    from collections import namedtuple
    import re

    FileNameInfo = namedtuple("FileNameInfo", "irradiance treatment module_type file_full_path")
    re_irradiance = re.compile(r"(?<=\_)\d{3,4}(?=W\_)")
    re_treatment = re.compile(r"(?<=\_)T\d{1}(?=\.)")
    re_module_type = re.compile(r"[A-Z-\_]*\d{1,50}(?=\_)")

    FileInfo = FileNameInfo(
        irradiance=int(re.findall(re_irradiance, file)[0]),
        treatment=re.findall(re_treatment, file)[0],
        module_type=re.findall(re_module_type, file)[0],
        file_full_path=file,
    )
    return FileInfo


def df2sqlite(dataframe, file=None, tbl_name="import"):

    """The function df2sqlite converts a dataframe into a squlite database.
    
    Args:
       dataframe (panda.DataFrame): the dataframe to convert in a data base
       file (Path): full pathname of the database
       tbl_name (str): name of the table
    """

    import sqlite3

    if file is None:
        conn = sqlite3.connect(":memory:")
    else:
        conn = sqlite3.connect(file)

    cur = conn.cursor()
    wildcards = ",".join(["?"] * len(dataframe.columns))
    data = [tuple(x) for x in dataframe.values]
    cur.execute(f"DROP TABLE IF EXISTS {tbl_name}")
    col_str = '"' + '","'.join(dataframe.columns) + '"'
    cur.execute(f"CREATE TABLE {tbl_name} ({col_str})")
    cur.executemany(f"insert into {tbl_name} values ({wildcards})", data)
    conn.commit()
    cur.close()
    conn.close()


def sieve_files(irradiance_select, treatment_select, module_type_select, database_path):

    """The sieve_files select the file witch names satisfy the foolowing querry:
         - the irradiance (200,400,...) must be part of the irradiance_select list
         - the treatment (T0, T1, T2,...) must be part of the treatment_select list
         - the module type (JINERGY, QCELLS, BOREALIS,...) must be part of the module_type_select list
         
        Args:
           irradiance_select (list of int): list of irradiances to be selected
           treatment_select (list of str): list of treatments to be selected
           module_type_select (list of str): list of modules to be selected
           database_path (path): full path of the data base
           
        Return:
          List of the full path of the selected files.
        
    """
    # Standard library imports
    import sqlite3
    from string import Template
    

    conv2str = lambda list_: str(tuple(list_)).replace(",)", ")")

    conn = sqlite3.connect(database_path)
    cur = conn.cursor()

    querry_d = Template(
        """SELECT file_full_path
                        FROM $table_name 
                        WHERE module_type  IN $module_type_select
                        AND irradiance IN $irradiance_select
                        AND treatment IN $treatment_select
                        ORDER BY module_type ASC
                        """
    )

    cur.execute(
        querry_d.substitute(
            {
                "table_name": DATA_BASE_TABLE_FILE,
                "module_type_select": conv2str(module_type_select),
                "irradiance_select": conv2str(irradiance_select),
                "treatment_select": conv2str(treatment_select),
            }
        )
    )

    querry = [x[0] for x in cur.fetchall()]
    cur.close()
    conn.close()
    return querry

def assess_path_folders(path_root=None):
    
    # Standard library imports
    from pathlib import Path
    
    if path_root is None:
        root = Path.home()
    else:
        root = path_root

    data_folder = select_data_dir(root,'Select the root folder')  # Selection of the root folder
    
    return data_folder

def build_files_database(data_folder,verbose= True):
    ''' 
    Build the table DATA_BASE_TABLE_FILE in the data base DATA_BASE_NAME with the following fields
    irradiance, treatment, module_type, file_full_path.
    
    Args:
        data_folder (path):  path  of the folder containing the experimental flashtest files.
    '''

    # Standard library imports
    from collections import Counter
    import os
    from pathlib import Path

    # 3rd party import
    import pandas as pd


    
    datafiles_list = list(Path(data_folder).rglob("*.csv")) # Recursive collection all the .csv lies
    
    if not datafiles_list:
        raise Exception(f"No .csv files detected in {data_folder} and sub folders")

    list_files_descp = [parse_filename(str(file)) for file in datafiles_list]

    file_check = True  # Check for the multi occurrences of a file
    list_multi_file = []
    for file,frequency in Counter([os.path.basename(x) for x in datafiles_list]).items(): # Check the the uniqueness of a file name
        if frequency>1:
            list_multi_file.append(file)
            file_check = False
    if not file_check:
        raise Exception(f"The file(s) {' ,'.join(list_multi_file)} has(have) a number of occurrence greater than 1.\nPlease correct before proceeding")


    df_files_descp  = pd.DataFrame(list_files_descp) # Build the database

    database_path = Path(data_folder) / Path(DATA_BASE_NAME)

    df2sqlite(df_files_descp, file=database_path, tbl_name=DATA_BASE_TABLE_FILE)
    
    if verbose:
        print(f'{len(datafiles_list)} files was detected.\ndf_files_descp and the data base table {DATA_BASE_TABLE_FILE} in {database_path} are built')
    
    return df_files_descp

def build_metadata_dataframe(df_files_descp,data_folder):

    '''Building of the dataframe df_meta out of the interactivelly selected module type.
    The df_meta index are the file names without extention (ex: QCELLS901219162417702718_0200W_T0).
    The df_meta columns are: Title, Pmax, Fill Factor, Voc, Isc, Rseries, Rshunt,
    Vpm, Ipm, irradiance, treatment, module_type.

    Args:
        df_files_descp (dataframe): dataframe built by the function build_files_database 
        data_folder (path):  path  of the folder containing the experimental flashtest files. 
    '''
    
    # Standard library imports
    import os
    from pathlib import Path

    #3rd party imports
    import pandas as pd

    # Interactive selection of the modules
    list_mod_selected = build_modules_list(df_files_descp,data_folder)                                
    
    # Extraction from the file database all the filenames related to the selected modules
    list_files_path = build_modules_filenames(list_mod_selected,data_folder)
    
    # Builds the list of files basenames without extension
    list_files_name = [os.path.splitext(os.path.basename(x))[0] for x in list_files_path]
    list_files_name.sort()

    # Extraction from the dataframe df_files_descp a sub dataframe related to the selected modules 
    df_files_descp_copy = df_files_descp
    df_files_descp_copy.index = [os.path.basename(x).split('.')[0] 
                                for x in df_files_descp_copy['file_full_path'].tolist()]
    df_files_descp_copy = df_files_descp_copy.loc[list_files_name,['irradiance','treatment','module_type'] ]
    
    # Building of the dataframe df_meta out of the flashtest files metadata 
    list_dict_metadata = [read_flashtest_file(file,parse_all=False).meta_data for file in list_files_path]
    df_meta = pd.DataFrame.from_dict(list_dict_metadata)
    df_meta.index = list_files_name #df_meta['ID']
    df_meta = df_meta.loc[:,COL_NAMES] # keep only the columns which names COL_NAMES 
                                       #  defined in PVcharacterization_GUI.py
    
    # Merges df_meta and df_files_descp_copy to add the tree columns: irradiance, treatment, module_type
    df_meta = pd.merge(df_meta,df_files_descp_copy,left_index=True, right_index=True)

    # Builds a database for future uses
    database_path = Path(data_folder) / Path(DATA_BASE_NAME)
    df2sqlite(df_meta, file=database_path, tbl_name=DATA_BASE_TABLE_EXP)
    
    return df_meta
    
def build_modules_list(df_files_descp,data_folder):

    '''Interactive selection of modules type out of the dataframe df_meta.
   
    Args:
        df_files_descp (dataframe): dataframe built by the function build_files_database 
        data_folder (path):  path  of the folder containing the experimental flashtest files. 
       
    Returns:
        list_mod_selected (list os str): list of selected modules type.
    '''
    

    # Interactive selection of the modules
    list_modules_type = df_files_descp['module_type'].unique()
    list_mod_selected = select_items(list_modules_type,
                                'Select the modules type',
                                mode = 'multiple') 
    
    
    return list_mod_selected
    
def build_modules_filenames(list_mod_selected,data_folder):

    '''Builds out of the modules type list_mod_selected the list of all filename
    related to these modulees.

    Args:
        df_files_descp (dataframe): dataframe built by the function build_files_database 
        list_mod_selected (list of str):  list of module names. 
        
    Returns
        list_files_path (list of str): list of the path of the files related to the modules type
                                    defined in the list list_mod_selected.
    '''
    
    # Standard library imports
    import os
    from pathlib import Path                           
    
    # Extract from the file database all the filenames related to the selected modules
    database_path = Path(data_folder) / Path(DATA_BASE_NAME)
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
    
    def title(M_items_per_row = 3 ):
        '''Builds th PCA plot title out of the dict dict_module_type with M_items_per_row module type name
        per row'''    
        nl = '\n'      
        list_mod = [f'{value} : {key}' for key,value in dict_module_type['module_type'].items()]
        list_mod = [', '.join( list_mod[idx:idx+M_items_per_row]) for idx in range(0,len(list_mod),M_items_per_row)]
        text = f"Module names are:{nl}{nl.join(list_mod)}"
        return text
    
    list_params = COL_NAMES.copy()
    list_params.remove('Title')
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
    
def data_dashboard(df_files_descp,data_folder,list_params):
    
    # Standard library imports
    from pathlib import Path
    
    #3rd party imports
    import pandas as pd

    df_meta = build_metadata_dataframe(df_files_descp,data_folder)
    df_meta_dashboard = df_meta.pivot(values= list_params,index=['module_type','treatment',],
                                      columns=['irradiance',]) 
    df_meta_dashboard.to_excel(data_folder/Path('exp_summary.xlsx'))
    
    return df_meta_dashboard
    
def correct_filename(filename,new_moduletype_name):
    
    '''Correction of the filename with a new module type name.
    ex. : filename = 'C:/Users/franc/PVcharacterization_files/JINERGY3272023326035_200W_T2.csv'
          new_moduletype_name = 'JINERGY3272023326039'
          corrected_filename = 'C:/Users/franc/PVcharacterization_files\JINERGY3272023326039_0200W_T2.csv'
    The field irradiance, if necessary, is also upgraded  by addind leading zeros to obtain a four digits number.
          
    '''

    # Standard library imports
    import os
    from pathlib import Path

    import PVcharacterization_Utils as pv
    
    FileInfo = pv.parse_filename(filename)
    new_file_mame = f'{new_moduletype_name}_{FileInfo.irradiance:04d}W_{FileInfo.treatment}.csv'
    corrected_filename = os.path.join(os.path.dirname(filename), new_file_mame)
    
    return corrected_filename