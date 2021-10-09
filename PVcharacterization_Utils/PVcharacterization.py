""" Creation: 2021.09.07
    Last update: 2021.09.20
    
    Useful functions for correctly parsing the aging data files
"""
__all__ = [
    "data_parsing",
    "df2sqlite",
    "parse_filename",
    "plot_params",
    "sieve_files",
]

#Internal imports 
from .PVcharacterization_global import (DEFAULT_DIR,
                                        DATA_BASE_NAME,
                                        DATA_BASE_TABLE,
                                        USED_COLS,
                                        PARAM_UNIT_DIC)
                                       
def data_parsing(filepath, parse_all=True):

    """
    The function `data_parsing` reads a csv file organized as follow:
    
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

    index_data_header = np.where(
        df_data.iloc[:, 0].str.contains(
            "^ Volt|Ref Cell",  # Find the indice of the
            case=True,  # headers of th IV and
            regex=True,
        )
    )[
        0
    ]  # Ref Cell data

    index_data_header = np.insert(
        index_data_header,  # Insersion of index 0 and the index of th
        [0, len(index_data_header)],  # last numerical value
        [0, len(df_data) - 3],
    )

    meta_data = df_data.iloc[np.r_[index_data_header[0] : index_data_header[1]]]
    meta_data = dict(zip(meta_data[0], meta_data[1]))
    meta_data = {key.split(":")[0]: val for key, val in meta_data.items()}

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

    FileNameInfo = namedtuple("FileNameInfo", "power treatment name file")
    re_power = re.compile(r"(?<=\_)\d{4}(?=W\_)")
    re_treatment = re.compile(r"(?<=\_)T\d{1}(?=\.)")
    re_name = re.compile(r"[A-Z-\_]*\d{1,50}(?=\_)")

    FileInfo = FileNameInfo(
        power=int(re.findall(re_power, file)[0]),
        treatment=re.findall(re_treatment, file)[0],
        name=re.findall(re_name, file)[0],
        file=file,
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


def sieve_files(pow_select, treatment_select, name_select, database_path):

    """The sieve_files select 
    """
    # Standard library imports
    import sqlite3
    from string import Template
    

    conv2str = lambda list_: str(tuple(list_)).replace(",)", ")")

    conn = sqlite3.connect(database_path)
    cur = conn.cursor()

    querry_d = Template(
        """SELECT file
                        FROM $table_name 
                        WHERE name  IN $name_select
                        AND power IN $pow_select
                        AND treatment IN $treatment_select
                        ORDER BY name ASC
                        LIMIT 50"""
    )

    cur.execute(
        querry_d.substitute(
            {
                "table_name": DATA_BASE_TABLE,
                "name_select": conv2str(name_select),
                "pow_select": conv2str(pow_select),
                "treatment_select": conv2str(treatment_select),
            }
        )
    )

    querry = [x[0] for x in cur.fetchall()]
    cur.close()
    conn.close()
    return querry


def set_min_max_param(df_meta,diff=False):
    
    if diff:
        min_max_param = {"Rseries":[-30, 30],
                         "Rshunt":[-30, 30],
                         "Voc":[-3.2, 1.6],
                         "Isc":[-3.2, 1.6],
                         "Pmax":[-3.2, 1.6],
                         "Fill Factor":[-3.2, 1.6]}
    else:
        USED_COLS_copy = list(pv.USED_COLS)
        USED_COLS_copy.remove("Title")
        df_meta_ = df_meta.astype(dtype={col_name:float for col_name in USED_COLS_copy}, copy=True)
        df_stat = df_meta_.describe()
        min_max_param = {param:np.array(list(value.values()))
                        for param,value in df_stat.loc[['min','max'],:].to_dict().items()}
        min_max_param = {param:[x[0] - (ecart := np.diff(x)/2),x[1] + ecart]
                        for param,x in min_max_param.items()}
        
    return min_max_param

def plot_params(params, df_meta,diff=False):
    
    '''Plots for different experiments and for different parameters:
       - the relative  evolution (in %) of the parameters vs power if diff=True
       - the parameters vs power if diff=False
    The parameters values vs experiments, times, powers and are store in a dataframe like:
    
      ID                          Voc         Isc   Rseries   power time                                                         
   JINERGY3272023326035_0200W_T0  50.5082    1.827  1.95841     200   T0  
   JINERGY3272023326035_0200W_T1  50.6780  1.82484  1.87985     200   T1   
   JINERGY3272023326035_0200W_T2  50.3452  1.79790  2.09313     200   T2  
   JINERGY3272023326035_0400W_T0  51.8321  3.61464  1.05142     400   T0 
   
   Args:
       params (list of str): list of parameters to be plotted
       df_meta (dataframe): dataframe organized as above
       diff (bool): if true the parameters relative evolution in %, vs power, between every difference time are plotted
                    if false the parameters evolution  vs power are plotted

    '''

    # Standard library imports
    from itertools import combinations

    # 3rd party import
    import matplotlib.pyplot as plt
    import numpy as np
    
    color = ['#0000A0','#1569C7','#78f89d','#FFEB3B','#E64A19'] # markers color
                                                                # a different color per power
    marker = ["o", "v", ">", "<", "s", "p"]                     # maker symbol
                                                                # a different symbol per experiment
        
    list_exp = pd.unique(df_meta['name'])
    nbr_time = len(pd.unique(df_meta['time'])) # Number of different times
    assert nbr_time > 1, "not enough time measurements. Should be greeter than 1"
    
    # Set y dynamic of the plots
    if diff:
        combination_length = 2
        dic_ylim = set_min_max_param(df_meta,diff=True)        
    else:
        combination_length = 1
        dic_ylim = set_min_max_param(df_meta,diff=False)
    
    #  Set x dynamic of the plots
    pow_list = sorted(pd.unique(df_meta['power']))
    pow_add_nbr = 2
    pow_add = pow_add_nbr * (max(pow_list) - min(pow_list))
    pow_min, pow_max = (
        min(pow_list) -  pow_add,
        max(pow_list) +  pow_add,
    )

    
    fig = plt.figure(figsize=(15,15) if len(params)>1 else (10,5))
    gs = fig.add_gridspec(
        len(params),
        int(nbr_time * (nbr_time - 1) / 2) if diff else nbr_time,
        hspace=0,
        wspace=0
    )

    ax = gs.subplots(sharex="col", sharey="row")
    if len(params) ==1: # we trasform a 1D array to a 2D array
        ax = ax.reshape((1,np.shape(ax)[0]))

    for idx_exp, exp in enumerate(list_exp): # Loop on the experiments
        
        df_exp = df_meta.query("name == @exp")

        # split df_exp into a dic keyed by time (T0,T1,...). The values are dataframe df_exp
        # with column time=T0,T1,...
        list_t = sorted(pd.unique(df_exp['time']))
        dic_time = {time : df_exp.loc[df_exp["time"] == time, :] for time in list_t}
        

        for idx_param, param in enumerate(params): # Loop on the parameter
            dic_time_cp = {}

            for time in combinations(list_t, combination_length): # Loop on time difference
                if combination_length>1 :
                    val = np.array(dic_time[time[1]][param].astype(float).tolist())
                    ref = np.array(dic_time[time[0]][param].astype(float).tolist())
                    delta = 100 * (val - ref) / ref
                    dic_time_cp[time[1] + "-" + time[0]] = dic_time[time[1]].copy()
                    dic_time_cp[time[1] + "-" + time[0]]["Delta_" + param] = delta
                else:
                    dic_time_cp[time[0]] = dic_time[time[0]].copy()
                    dic_time_cp[time[0]]["Delta_" + param] = np.array(dic_time[time[0]][param].astype(float).tolist())

            list_times_diff = sorted(dic_time_cp.keys())
    
            for idx_time, key in enumerate(list_times_diff):
                for idx_power,x_y in enumerate(zip(dic_time_cp[key]["power"],
                                                   dic_time_cp[key]["Delta_" + param])):
                    ax[idx_param, idx_time].scatter(
                            x_y[0],
                            x_y[1],
                            c=color[idx_power] ,
                            marker=marker[idx_exp],
                            label=exp+' '+str(x_y[0])
                        )


                ax[idx_param, idx_time].axhline(y=0, color="red", linestyle="--")
                if idx_param == 0:
                    ax[idx_param, idx_time].set_title(key)
                ax[idx_param, idx_time].set_xlabel("Power ($W/{m^2}$)")
                if idx_time == 0:
                    if combination_length > 1:
                        ax[idx_param, idx_time].set_ylabel("$\Delta$ " + param + " (%)")
                    else:
                        ax[idx_param, idx_time].set_ylabel(f'{param} ({pv.PARAM_UNIT_DIC[param]})')
                ax[idx_param, idx_time].tick_params(axis="x", rotation=90)
                ax[idx_param, idx_time].set_xticks(pow_list, minor=False)
                ax[idx_param, idx_time].set_xticklabels(pow_list, fontsize=12)
                ax[idx_param, idx_time].set_xlim([pow_min, pow_max])
                ax[idx_param, idx_time].set_ylim(dic_ylim.get(param,[-3.2, 1.6]))
                for axis in ["top", "bottom", "left", "right"]:
                    ax[idx_param, idx_time].spines[axis].set_linewidth(2)
                    
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right',bbox_to_anchor=(0.6,0, 0.5, 1))
    #title = chr(9679) + " " + list_exp[0]
    #if len(list_exp) ==2 : title = title + ', ' + chr(9660) + " " + list_exp[1]
    #fig.suptitle(title, fontsize=13)
    #fig.subplots_adjust(top=0.95 if len(params)>1 else 0.85)