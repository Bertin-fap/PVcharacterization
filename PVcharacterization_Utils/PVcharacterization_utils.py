__all__ = ['build_timeline_db',
           'plot_time_schedule',]

def plot_time_schedule(path_suivi_module, path_time_schedule):

    '''Plots time schedule of modules under experiment.
    '''
    
    # Standard library import
    import datetime
    import re
    
    # 3rd party imports
    import pandas as pd
    import plotly.express as px
    
    
    # Input the init date
    re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
    
    delta_days = 2 # Use to increase range_x
    color_choice = 'PROJET', # 'PROJET'
    
    today = input(f'Enter the dep data using the format yyyy-mm-dd') 
    
    while not re.findall(re_date,today):
        print ('incorrect date format')
        today = input(f'Enter the dep data using the format yyyy-mm-dd: ') 
    
    # Reads and cleans the .xlsx file
    df = pd.read_excel(path_suivi_module).astype(str)
    
    # Selects modules with status 'EN COURS' and with  'DATE SORTIE PREVUE'>=today
    df = df.loc[df['ETAT'] == 'EN COURS']
    df['DATE ENTREE'] = pd.to_datetime(df['DATE ENTREE'], format="%Y-%m-%d")
    df['DATE SORTIE PREVUE'] = pd.to_datetime(df['DATE SORTIE PREVUE'], format="%Y-%m-%d")
    df = df.loc[df['DATE SORTIE PREVUE']>=pd.to_datetime(today, format="%Y-%m-%d")]
    
    # Buids a dataframe multiindexed by ('PROJET','ENCEINTE','DATE ENTREE')and containing
    # the number of experiments corresponding to that multiindex
    dg = df.groupby(['PROJET','ENCEINTE','DATE ENTREE']).count()['N°MODULE'] # The choice of 'N°MODULE' is arbitrary
    
    # Builds a dataframe for plotly
    list_dataframe = []
    list_date_start_exp = []
    list_date_end_exp = []
    
    for i, row_tup in enumerate(df.iterrows()):
        index = row_tup[0]
        row = row_tup[1]
        
        projet = row['PROJET']
        enceinte = row['ENCEINTE']
        
        date_start_exp = row['DATE ENTREE']
        list_date_start_exp.append(date_start_exp)
        
        date_end_exp = row['DATE SORTIE PREVUE']
        list_date_end_exp.append(date_end_exp)
        
        # Builds a list of dates, with a step equal to one day, between date_end_exp and date_start_exp
        days = [date_start_exp + datetime.timedelta(days=x+1) 
                for x in range(0, (date_end_exp-date_start_exp).days)]
        days_indice = [str(index+2)]*len(days)
       
        
        if i%2 : days = days[::-1] # Avoids rastering during plotting
            
        label_projet = row['PROJET']
        label_projet += f', {enceinte}'
        label_projet += f' ( {dg.loc[(projet,enceinte,date_start_exp)]}, '
        label_projet += f' {date_start_exp.strftime("%Y-%m-%d")})'
        
        dict_ = {'Index_exp':days_indice,
                 'Date':days,
                 'PROJET':label_projet,
                 'N°MODULE':row['N°MODULE'],
                 'PROGRAMME DE TEST PREVU':row['PROGRAMME DE TEST PREVU'],
                 "TYPE D'ESSAI":row["TYPE D'ESSAI"],
                 "ENCEINTE":enceinte,
                 "TAILLE":row["TAILLE"],}
        
        list_dataframe.append(pd.DataFrame.from_dict(dict_))
    
    df_all = pd.concat(list_dataframe)
    range_x = (min(list_date_start_exp) - datetime.timedelta(days = delta_days),
               max(list_date_end_exp) + datetime.timedelta(days = delta_days))
    
    # Plots the timeline
    fig = px.line(df_all,
                      x='Date',
                      y='Index_exp',
                      color='PROJET',
                      labels={'Date':'',
                              'Index_exp':'ID'},
                      #facet_row='PROJET',
                      height = 1000,
                      range_x = range_x,
                      custom_data=['PROJET','N°MODULE','PROGRAMME DE TEST PREVU',"TYPE D'ESSAI",'ENCEINTE','TAILLE'])
    
    fig.update_traces(
        line=dict(width=12),
        hovertemplate='<br>'.join([
            'Date: %{x}',
            'ID: %{y}',
            'Projet: %{customdata[0]}',
            'N°module: %{customdata[1]}',
            'PROGRAMME DE TEST PREVU: %{customdata[2]}',
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
    