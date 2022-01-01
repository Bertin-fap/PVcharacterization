__all__ = ['build_timeline_db',
           'plot_time_schedule',]

def plot_time_schedule(path_suivi_module, path_time_schedule, dep_time):

    '''Plots time scheule of modules under experiment.
    '''

    # Standard library import
    import datetime
    import re

    # 3rd party imports
    import pandas as pd
    import plotly.express as px

    delta_days = 2 # Use to increase range_x
    color_choice = 'N°MODULE', # 'PROJET'
    
    # Selects modules with proper DATE SORTIE PREVUE
    df = read_excel_timeline(path_suivi_module)
    df = df.loc[(df['ETAT'] == 'EN COURS') & (df['DATE SORTIE PREVUE']>dep_time)]
    dg = df.groupby('PROJET').count()
    
    # Builds a dataframe for plotly
    list_dataframe = []
    list_start = []
    list_end = []
   
    for i,(index, start, end) in enumerate(zip(df.index,df['DATE ENTREE'],df['DATE SORTIE PREVUE'] ),0):
        list_start.append(start)
        list_end.append(end)
        days = [start + datetime.timedelta(days=x+1) for x in range(0, (end-start).days)]
        if i%2 : days = days[::-1]
        label_projet = df.iloc[i]['PROJET']
        label_projet += f' ({dg.loc[df.iloc[i]["PROJET"],"N°MODULE"]}, '
        label_projet += f' {end.strftime("%Y-%m-%d")})'
        dict_ = {'Index_exp':[str(index+2)]*len(days),
                 'Date':days,
                 'PROJET':label_projet,
                 'N°MODULE':df.iloc[i]['N°MODULE'],
                 'PROGRAMME DE TEST PREVU':df.iloc[i]['PROGRAMME DE TEST PREVU'],
                 "TYPE D'ESSAI":df.iloc[i]["TYPE D'ESSAI"],
                 "ENCEINTE":df.iloc[i]["ENCEINTE"],
                 "TAILLE":df.iloc[i]["TAILLE"],}
        list_dataframe.append(pd.DataFrame.from_dict(dict_))
        
    df_all = pd.concat(list_dataframe)
    range_x = (min(list_start) - datetime.timedelta(days = delta_days),
               max(list_end) + datetime.timedelta(days = delta_days))
               
    # Plots the timeline
    fig = px.line(df_all,
                      x="Date",
                      y="Index_exp",
                      color='N°MODULE',
                      labels={'Date':'',
                              'Index_exp':'ID'},
                      #facet_row='PROJET',
                      height = 1000,
                      range_x = range_x,
                      custom_data=['PROJET','N°MODULE','PROGRAMME DE TEST PREVU',"TYPE D'ESSAI","ENCEINTE","TAILLE"])

    fig.update_traces(
        line=dict(width=12),
        hovertemplate="<br>".join([
            "Date: %{x}",
            "ID: %{y}",
            "Projet: %{customdata[0]}",
            "N°module: %{customdata[1]}",
            "PROGRAMME DE TEST PREVU: %{customdata[2]}",
            "TYPE D'ESSAI:  %{customdata[3]}",
            "ENCEINTE:  %{customdata[4]}",
            "TAILLE:  %{customdata[5]}",
        ])
    )
    fig.show()
    fig.write_html(path_time_schedule)
    print(f'The .html file {path_time_schedule} has been stored')
    
def read_excel_timeline(path_suivi_module):

    '''Reads and cleans the .xlsx time_line file.
    '''
    # 3rd party imports
    import pandas as pd
    
    df = pd.read_excel(path_suivi_module).astype(str)
    df.dropna()
    df['DATE SORTIE PREVUE'] = df['DATE SORTIE PREVUE'].apply(lambda x: x.strip().split(' ')[0] if x != '00:00:00' else '')
    df['DATE ENTREE'] = pd.to_datetime(df['DATE ENTREE'], format="%Y-%m-%d")
    df['DATE SORTIE PREVUE'] = pd.to_datetime(df['DATE SORTIE PREVUE'], format="%Y-%m-%d")
    df.drop(df.query('`DATE ENTREE` == "NaT"').index,inplace=True)
    df.drop(df.query('`DATE SORTIE PREVUE` == "NaT"').index,inplace=True)
    
    return df

def build_timeline_db(path_suivi_module):

    '''Creation: 2021.12.04
    Last update: 2021.12.04
    Demonstration of database constructruction and querying.
    '''

    # Standard librariy imports
    from pathlib import Path
    
    #Internal imports 
    from .config import GLOBAL
    from .PVcharacterization_database import (df2sqlite,)
    
    db_name = 'module_timeline.db' # Database file name
    tbl_name = 'data'              # Database table name
       
    db_path =  GLOBAL["WORKING_DIR"] / Path(db_name)
    
    # Reads and cleans the .xlsx file
    df = read_excel_timeline(path_suivi_module)
    df.columns = [x.strip().replace(' ','_') for x in df.columns]
    
    df2sqlite(df, path_db=db_path, tbl_name="data")
    