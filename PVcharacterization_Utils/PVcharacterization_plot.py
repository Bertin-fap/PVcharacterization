__all_ =['plot_params','init_plot_diff','plot_params_diff','construct_x_y']

from .PVcharacterization_global import (PARAM_UNIT_DIC,
                                        TREATMENT_DEFAULT_LIST,
                                        USED_COLS,
                                        DATA_BASE_NAME,
                                        PARAM_UNIT_DIC,)
from .PVcharacterization_GUI import select_items
from .PVcharacterization_flashtest import (sieve_files,
                                 read_flashtest_file)


def plot_params(params,list_modules_type, df_meta,list_diff = [],dic_trt_meaning=None):
    
    '''Plots for different modules and for different parameters:
       - the relative  evolution (in %) of the parameters vs irradiance for treatment differences if diff=True
       - the parameters vs irradiance for any treatment if diff=False
    The parameters values vs modules (ID), treatment, irradiance and are store in a dataframe like:
    
      ID                          Voc         Isc   Rseries   irradiance treatment                                                         
   JINERGY3272023326035_0200W_T0  50.5082    1.827  1.95841     200       T0  
   JINERGY3272023326035_0200W_T1  50.6780  1.82484  1.87985     200       T1   
   JINERGY3272023326035_0200W_T2  50.3452  1.79790  2.09313     200       T2  
   JINERGY3272023326035_0400W_T0  51.8321  3.61464  1.05142     400       T0 
   
   Args:
       params (list of str): list of parameters to be plotted
       list_modules_type (list of str): list of modules type to be plotted
       df_meta (dataframe): dataframe organized as above
       list_diff (list of tuple): [(T1,T0),(T2,T0),...]
                                  if list_diff=[] the parameters evolution  vs irradiance are plotted.
                                  else the parameters relative evolution in %, vs irradiance, between
                                  treatment tuple[0] and tuple[1] are plotted.

    '''

    # Standard library imports
    from itertools import combinations

    # 3rd party import
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    assert len(params)>1, "The number of parameters must be greater than one"
    
    diff = bool(list_diff) # if true the parameters relative evolution in %, vs irradiance, 
                           #    between every difference treatment are plotted
                           # if false the parameters evolution  vs irradiance are plotted
    color = ['#0000A0','#1569C7','#78f89d','#FFEB3B','#E64A19'] # markers color
                                                                #  different color per irradiance
    marker = ["o", "+", "s", "<", ">", "p"]                     # maker symbol
                                                                #  different symbol per module type
        
    if dic_trt_meaning is None:
        dic_trt_meaning = {trt:trt for trt in TREATMENT_DEFAULT_LIST}
    
        
        
    list_irr = sorted(pd.unique(df_meta['irradiance'])) # List of different irradiances
    list_trt = pd.unique(df_meta['treatment'])         # list of different treatmments
    list_trt.sort()
    nbr_trt = len(list_trt)     # Number of different treatments
    
    if diff:
        list_trt_diff = list_diff
        dic_ax = {t:i for i,t in enumerate(list_diff)}
    else:
        list_trt_diff = list_trt
        dic_ax = {t:i for i,t in enumerate(list_trt)}
    
    #  Set y dynamic of the plots (enlarge the irradiance dynamic)
    dic_ylim = set_ymin_ymax_param(df_meta,params, list_modules_type,list_trt_diff,diff)
            
    #  Set x dynamic of the plots (enlarge the irradiance dynamic)
    (irr_min, irr_max) = set_xmin_xmax(list_irr)

    # Set the figure size and the subplots
    fig = plt.figure(figsize=(15,15) if len(params)>1 else (10,5))
    gs = fig.add_gridspec(
        len(params),
        len(list_trt_diff),
        hspace=0,
        wspace=0
    )

    ax = gs.subplots(sharex="col", sharey="row")
    if len(params) ==1 : # we transform a 1D array to a 2D array
        ax = ax.reshape((1,np.shape(ax)[0]))
    if len(list_trt_diff) ==1: # we transform a 1D array to a 2D array
        ax = ax.reshape((np.shape(ax)[0],1))
        
    # Loop over module, parameters, treatment/treatment differences and irradiance
    for idx_module, module_type in enumerate(list_modules_type): # Loop on the modules type
        for idx_param, param in enumerate(params): # Loop over the parameters
            for trt in list_trt_diff: # Loop over the treatmentS
                idx_trt = dic_ax[trt]
                x,y = construct_x_y(df_meta,module_type,trt,param,diff)
                for idx_irr,x_y in enumerate(zip(x,y)):
                    ax[idx_param, idx_trt].scatter(
                            x_y[0],
                            x_y[1],
                            c=color[idx_irr] ,
                            marker=marker[idx_module],
                            label=module_type+' '+str(x_y[0]))


                if diff: ax[idx_param, idx_trt].axhline(y=0, color="red", linestyle="--")
                if idx_param == 0:
                    title = f'{dic_trt_meaning[trt[0]]} - {dic_trt_meaning[trt[1]]}' if diff else dic_trt_meaning[trt]
                    ax[idx_param, idx_trt].set_title(title)
                ax[idx_param, idx_trt].set_xlabel("Irradiance ($W/{m^2}$)")
                if idx_trt == 0:
                    if diff:
                        ax[idx_param, idx_trt].set_ylabel("$\Delta$ " + param + " (%)")
                    else:
                        ax[idx_param, idx_trt].set_ylabel(f'{param} ({PARAM_UNIT_DIC[param]})')
                ax[idx_param, idx_trt].tick_params(axis="x", rotation=90)
                ax[idx_param, idx_trt].set_xticks(list_irr, minor=False)
                ax[idx_param, idx_trt].set_xticklabels(list_irr, fontsize=12)
                ax[idx_param, idx_trt].set_xlim([irr_min, irr_max])
                ax[idx_param, idx_trt].set_ylim(dic_ylim[param])
                for axis in ["top", "bottom", "left", "right"]:
                    ax[idx_param, idx_trt].spines[axis].set_linewidth(2)
                    
    labels_handles = {
                      label: handle for ax in fig.axes 
                      for handle, label in zip(*ax.get_legend_handles_labels())
                     }
    fig.legend(
               labels_handles.values(),
               labels_handles.keys(),
               loc='center right',
               bbox_to_anchor=(0.6,0, 0.5, 1),
               bbox_transform=plt.gcf().transFigure,
     )
    
def construct_x_y(df_meta,module_type,treatment,param,diff):
    
    '''Construct for the module type 'module_type', the parameter 'parameter' and the treatment 'treatment' the 
    list of abscissa x and ordonates y where y(x) corresponds to:
      to parameter_value(irradiance; module_type, treatment[0])
      to 100*(parameter_value(irradiance; module_type, treatment[0]) - parameter_value(irradiance; module_type, treatment[1]))/
                       parameter_value(irradiance; module_type, treatment[1])
                       
    Args:
       df_meta (dataframe): dataframe descrided in plot_params
       module_type (str): the module type
       treatment (tuple) : (T<end>,T<deb>) for relative variation of the parameter between T<end> and T<deb>
                           (T<i>,) for the parameter value for the treatment T<i>
       param (str): the parameter
       diff (bool): TRUE we work with parameter differences
                    FALSE we work with parameters
       
    Returns:
       (x,y) 
    '''
    
    
    import numpy as np
    
    if not diff:
        df_meta_cp = df_meta.query("module_type==@module_type & treatment==@treatment")
        y = df_meta_cp[param].tolist()
        x = df_meta_cp['irradiance'].astype(float).tolist()
    else:
        df_meta_cp_end = df_meta.query("module_type==@module_type & treatment==@treatment[0] ")
        df_meta_cp_deb = df_meta.query("module_type==@module_type & treatment==@treatment[1] ")
        val = np.array(df_meta_cp_end[param].tolist())
        ref = np.array(df_meta_cp_deb[param].tolist())
        x = df_meta_cp_end['irradiance'].tolist() 
        y = 100 * (val - ref) / ref
        
    return (x,y)
    
def set_ymin_ymax_param(df_meta,params, list_modules_type,list_trt_diff,diff):
    
    '''Build a dict keyed by the parameters and wich values are list [ymin, ymax] 
    '''
    
    min_max_param = {}
    
    for param in params: # Loop over the parameters
        val = []
        for  module_type in list_modules_type: # Loop on the modules type
            for trt in list_trt_diff: # Loop over the treatmentS
                _,y = construct_x_y(df_meta,module_type,trt,param,diff)
                val.extend(y)
        min_max_param [param] =[ min(val),max(val)] 
            
    min_max_param = {param:[y[0] - (ecart := (y[1]-y[0]))/2,y[1] + ecart]
                        for param,y in min_max_param.items()}
   
    return min_max_param

def set_xmin_xmax(list_irr):
    irr_add_nbr = 2
    irr_add = irr_add_nbr * (max(list_irr) - min(list_irr))
    irr_min, irr_max = (
        min(list_irr) -  irr_add,
        max(list_irr) +  irr_add,
    )
    
    return (irr_min, irr_max)

def init_plot_diff(df_meta):
    
    '''Interactivelly builds a list of tuples [(T<i>,T<j>),...] where i>j and T<i> is the ieme treatment.
    
    Args:
       df_meta (dataframe): the dataframe containing the experimental values.
       
    Returns:
       A list of tuples
    '''
    
    # Standard library imports
    from itertools import combinations

    #3rd party imports
    import pandas as pd

    mod_selected = df_meta['module_type'].unique()
    if len(mod_selected)>2: raise Exception("Sorry, the number of modules must be <=2 ") 
    list_setup = []
    name =[]
    if len(mod_selected)==2:
        for x in df_meta[['module_type','treatment','irradiance']].groupby('module_type'):
            name.append(x[0] )
            list_setup.append(set(zip(x[1]['treatment'].tolist(), x[1]['irradiance'].tolist())))
        if list_setup[1] - list_setup[0]:
            raise Exception(f'Cannot compare {name[0] } and {name[1]}')

    list_treatments = pd.unique(df_meta['treatment'])

    if len(list_treatments)==1: raise Exception("Sorry, the number of treatments must be >1 ") 

    list_combinations = list(combinations(list_treatments,2))

    list_diff = select_items(list_combinations,
                                'Select the difference',
                                mode = 'multiple')
    list_diff = [(x[1],x[0]) for x in list_diff]

    return list_diff

def plot_params_diff(df_meta,list_diff, list_params=None,dic_trt_meaning=None):
    
    
    #3rd party imports
    import pandas as pd

    list_allowed_params = list(USED_COLS)
    list_allowed_params.remove('Title')
    if list_params is None:
        list_params = select_items(list_allowed_params,
                                      'Select the params',
                                       mode = 'multiple')
    else:
        params_copy = list(list_params)
        unkown_params = set(params_copy).difference(set( list_allowed_params))
        for unknow_param in unkown_params:
            print(f'WARNING parameter {unknow_param} will be ignored')
            list_params.remove(unknow_param)

    list_modules = pd.unique(df_meta['module_type'])          # List of modules name (ID)
    plot_params(list_params,
                list_modules,
                df_meta,list_diff = list_diff,
                dic_trt_meaning=dic_trt_meaning) #None
    
def plot_iv_curves(irr_select,name_select,trt_select,data_folder):

    # Standard library imports
    import os
    from pathlib import Path

    # 3rd party import
    import matplotlib.pyplot as plt


    database_path = Path(data_folder) / Path(DATA_BASE_NAME)

    querries = sieve_files(irr_select,trt_select,name_select,database_path)

    for i, res in enumerate([read_flashtest_file(querry).IV0 for querry in querries]):
        plt.plot(res['Voltage'],res['Current'],label=trt_select[i])
        plt.scatter(res['Voltage'][::50],res['Current'][::50],s=10)
        plt.xlabel(res.columns[0] +'[V]')
        plt.ylabel(res.columns[1] +'[A]')
        plt.title(f'Irradiance: {str(irr_select[0])} {PARAM_UNIT_DIC["IrrCorr"]}')

    plt.legend()
    plt.show()
