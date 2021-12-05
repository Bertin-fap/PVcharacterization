__all__ = ['plot_time_schedule',]

def plot_time_schedule(path_suivi_module, path_time_schedule):

    # Standard library import
    import datetime
    import re

    # 3rd party imports
    import pandas as pd
    import plotly.express as px


    # Input the init date
    re_date = re.compile(r'\d{4}-\d{1,2}-\d{1,2}')
    
    delta_days = 2 # Use to increase range_x
    color_choice = 'N°MODULE', # 'PROJET'

    dep_date = input(f'Enter the dep data using the format yyyy-mm-dd') 

    while not re.findall(re_date,dep_date):
        print ('incorrect date format')
        dep_date = input(f'Enter the dep data using the format yyyy-mm-dd: ') 

   
    df = pd.read_excel(path_suivi_module).astype(str)
    df.dropna()
    df['DATE SORTIE PREVUE'] = df['DATE SORTIE PREVUE'].apply(lambda x: x.strip().split(' ')[0] if x != '00:00:00' else '')
    df['DATE ENTREE'] = pd.to_datetime(df['DATE ENTREE'], format="%Y-%m-%d")
    df['DATE SORTIE PREVUE'] = pd.to_datetime(df['DATE SORTIE PREVUE'], format="%Y-%m-%d")
    df.drop(df.query('`DATE ENTREE` == "NaT"').index,inplace=True)
    df.drop(df.query('`DATE SORTIE PREVUE` == "NaT"').index,inplace=True)

    df = df.loc[(df['ETAT'] == 'EN COURS') & (df['DATE SORTIE PREVUE']>pd.to_datetime('2021-12-02', format="%Y-%m-%d"))]

    list_dataframe = []
    list_start = []
    list_end = []
    i=0
    for index, start, end in zip(df.index,df['DATE ENTREE'],df['DATE SORTIE PREVUE'] ):
        list_start.append(start)
        list_end.append(end)
        days = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
        if i%2 : days = days[::-1]
        dict_ = {'Index_exp':[str(index+2)]*len(days),
                 'Date':days,
                 'PROJET':df.iloc[i]['PROJET'],
                 'N°MODULE':df.iloc[i]['N°MODULE'],
                 'PROGRAMME DE TEST PREVU':df.iloc[i]['PROGRAMME DE TEST PREVU'],
                 "TYPE D'ESSAI":df.iloc[i]["TYPE D'ESSAI"],
                 "ENCEINTE":df.iloc[i]["ENCEINTE"],
                 "TAILLE":df.iloc[i]["TAILLE"],}
        list_dataframe.append(pd.DataFrame.from_dict(dict_))
        i += 1
        
    df_all = pd.concat(list_dataframe)
    range_x = (min(list_start) - datetime.timedelta(days = delta_days),
               max(list_end) + datetime.timedelta(days = delta_days))

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