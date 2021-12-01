__all__ = [
    "add_files_to_database",
    "df2sqlite",
    "sieve_files",
    "suppress_duplicate_database",
    "sqlite_to_dataframe",]
   
from .PVcharacterization_global import GLOBAL                                    


def add_files_to_database(files, working_dir):
    
    '''
    Args:
       files (list): list of the full path of the experiece file to be added to the databe
       working_dir (path): path of the folder holding the database
    '''

    import sqlite3
    from pathlib import Path
    from string import Template
    
    from .PVcharacterization_flashtest import parse_filename 
    
    DATA_BASE_NAME = GLOBAL['DATA_BASE_NAME']
    DATA_BASE_TABLE_FILE = GLOBAL['DATA_BASE_TABLE_FILE']

    database_path = Path(working_dir) / Path(DATA_BASE_NAME)
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor = conn.execute(f'select * from {DATA_BASE_TABLE_FILE} limit 1')
    col_name = ','.join([i[0] for i in cursor.description])
    template = Template('INSERT INTO $table($col_names) VALUES($values)')
    
    for file in files:
        parse = parse_filename(file)
        value = f"'{parse.exp_id}',{parse.irradiance},'{parse.treatment}','{parse.module_type}','{parse.file_full_path}'"
        cursor.execute(template.substitute({'table': DATA_BASE_TABLE_FILE,
                                            'col_names':col_name,
                                            'values':value}))

    conn.commit()
    conn.close()
    
def suppress_duplicate_database(working_dir):
    
    '''Suppresses duplicates from the database.
    Args:
        working_dir (path): path of the folder holding the database
    '''
    
    import sqlite3
    from pathlib import Path
    from string import Template
    
    DATA_BASE_NAME = GLOBAL['DATA_BASE_NAME']
    DATA_BASE_TABLE_FILE = GLOBAL['DATA_BASE_TABLE_FILE']    
    
    database_path = Path(working_dir) / Path(DATA_BASE_NAME)
    conn = sqlite3.connect(database_path)

    cursor = conn.cursor()

    template = Template('''DELETE FROM $table1
                           WHERE  rowid NOT IN
                           (
                           SELECT MIN(rowid)
                           FROM $table2
                           GROUP BY
                                   irradiance,
                                   treatment,
                                   module_type
                           )''')
    cursor.execute(template.substitute({'table1': DATA_BASE_TABLE_FILE,
                                        'table2': DATA_BASE_TABLE_FILE,}))
    conn.commit()

    conn.close()
    
def sqlite_to_dataframe(working_dir,tbl_name):
    
    '''Read a database as a dataframe.
    
    Args:
        working_dir (path): path of the folder holding the database
        tbl_name (str): name of the table
        
    Returns:
         (dataframe): a dataframe containing the database.
    '''
    
    from pathlib import Path
    import sqlite3
    import pandas as pd
    
    DATA_BASE_NAME = GLOBAL['DATA_BASE_NAME']
    
    database_path = Path(working_dir) / Path(DATA_BASE_NAME)

    cnx = sqlite3.connect(database_path)

    df = pd.read_sql_query("SELECT * FROM "+tbl_name, cnx)
    
    return df

def df2sqlite(dataframe, file=None, tbl_name="import"):

    '''The function df2sqlite converts a dataframe into a squlite database.
    
    Args:
       dataframe (panda.DataFrame): the dataframe to convert in a data base
       file (Path): full pathname of the database
       tbl_name (str): name of the table
    '''

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

    '''The sieve_files select the file witch names satisfy the foolowing querry:
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
    '''
    # Standard library imports
    import sqlite3
    from string import Template
    
    DATA_BASE_TABLE_FILE = GLOBAL['DATA_BASE_TABLE_FILE']

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
