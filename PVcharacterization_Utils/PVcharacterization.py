""" Creation: 2021.09.07
    Last update: 2021.09.20
    
    Useful functions for correctly parsing the aging data files
"""
__all__ = [
    "data_parsing",
    "df2sqlite",
    "parse_filename",
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

    FileNameInfo = namedtuple("FileNameInfo", "irradiance treatment name file_full_path")
    re_irradiance = re.compile(r"(?<=\_)\d{4}(?=W\_)")
    re_treatment = re.compile(r"(?<=\_)T\d{1}(?=\.)")
    re_name = re.compile(r"[A-Z-\_]*\d{1,50}(?=\_)")

    FileInfo = FileNameInfo(
        irradiance=int(re.findall(re_irradiance, file)[0]),
        treatment=re.findall(re_treatment, file)[0],
        name=re.findall(re_name, file)[0],
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


def sieve_files(irradiance_select, treatment_select, name_select, database_path):

    """The sieve_files select 
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
                        WHERE name  IN $name_select
                        AND irradiance IN $irradiance_select
                        AND treatment IN $treatment_select
                        ORDER BY name ASC
                        LIMIT 50"""
    )

    cur.execute(
        querry_d.substitute(
            {
                "table_name": DATA_BASE_TABLE,
                "name_select": conv2str(name_select),
                "irradiance_select": conv2str(irradiance_select),
                "treatment_select": conv2str(treatment_select),
            }
        )
    )

    querry = [x[0] for x in cur.fetchall()]
    cur.close()
    conn.close()
    return querry
