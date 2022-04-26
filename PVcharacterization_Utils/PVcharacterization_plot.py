__all_ =['construct_x_y',
         'plot_params_diff',
         'select_diff_treatment',
         'select_params',
         ]

from .config import GLOBAL
from .PVcharacterization_GUI import (select_items,
                                     select_files)
from .PVcharacterization_flashtest import (correct_iv_curve,
                                           parse_filename,
                                           read_flashtest_file,)

from .PVcharacterization_database import sieve_files

def _plot_params(params,
                 list_modules_type,
                 df_meta,
                 plot_params_dict,
                 list_diff = [],
                 dic_trt_meaning=None,
                 long_label=False,):
    
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
       long_label (bool): if true long labels such moduletype_irradiance are plotted
                          if false short labels moduletype is plotted instead

    '''

    # Standard library imports
    from itertools import combinations

    # 3rd party import
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    params_nb = len(params)
    
    diff = bool(list_diff) # if true the parameters relative evolution in %, vs irradiance, 
                           #    between every difference treatment are plotted
                           # if false the parameters evolution  vs irradiance are plotted
    
        
        
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
    
    #  Set ordinates dynamic of the plots (enlarge the irradiance dynamic)
    dic_ylim = _set_ymin_ymax_param(df_meta,params, list_modules_type,list_trt_diff,diff,
                                    limit_type= plot_params_dict['y_limit_type'])
            
    #  Set abcissa dynamic of the plots (enlarge the irradiance dynamic)
    (irr_min, irr_max) = _set_xmin_xmax(list_irr,irr_add_nbr=plot_params_dict['irr_add_nbr'])

    # Set the figure size and the subplots
    fig = plt.figure(figsize=(plot_params_dict['fig_width'],
                              plot_params_dict['fig_height_unit']*params_nb+
                              plot_params_dict['fig_title_height']))
    gs = fig.add_gridspec(
                          params_nb,
                          len(list_trt_diff),
                          hspace=0,
                          wspace=0
                          )
    ax = gs.subplots(sharex="col", sharey="row")
    
    if params_nb==1 : # we transform a 1D array to a 2D array
        ax = ax.reshape((1,np.shape(ax)[0]))
    if len(list_trt_diff) ==1: # we transform a 1D array to a 2D array
        ax = ax.reshape((np.shape(ax)[0],1))
        
    # Loop over module, parameters, treatment/treatment differences and irradiance
    for idx_module, module_type in enumerate(list_modules_type): # Loop on the modules type
        for idx_param, param in enumerate(params): # Loop over the parameters
            for num_trt,trt in enumerate(list_trt_diff): # Loop over the treatmentS
                idx_trt = dic_ax[trt]
                x,y = construct_x_y(df_meta,module_type,trt,param,diff)
                for idx_irr,x_y in enumerate(zip(x,y)):
                    if long_label:
                        label = module_type+' '+str(x_y[0])
                    else:
                        label = module_type if num_trt==0 else ''  # We plot the label only once in the inner loop treatment 
                   
                    ax[idx_param, idx_trt].scatter(
                            x_y[0],
                            x_y[1],
                            edgecolors=plot_params_dict['marker_colors'][idx_irr] ,  #modif
                            facecolors='none', #modif
                            marker=plot_params_dict['markers'][idx_module],
                            label=label,
                            s=plot_params_dict['marker_size'])
                if diff: ax[idx_param, idx_trt].axhline(y=0, color="red", linestyle="--")
                if idx_param == 0:
                    title = f'{dic_trt_meaning[trt[0]]} - {dic_trt_meaning[trt[1]]}' \
                            if diff else dic_trt_meaning[trt]
                    ax[idx_param, idx_trt].set_title(title,fontsize=plot_params_dict['title_fontsize'])
                ax[idx_param, idx_trt].set_xlabel("Irradiance ($W/{m^2}$)",
                                                   fontsize=plot_params_dict['labels_fontsize'])
                if idx_module == 0:
                    ax[idx_param, idx_trt].grid() 
                    
                if idx_trt == 0:
                    if diff: # Plot the relative evolution of the parameters
                        if (param == "Fill Factor") or (param == "Fill Factor_corr"):
                            param_ = "FF"
                        elif param == "Isc_corr":
                            param_ = "Isc"
                        else :
                            param_ = param
                            
                        ax[idx_param, idx_trt].set_ylabel("$\Delta$ " + param_ + " (%)",# modif 
                                                          fontsize=plot_params_dict['labels_fontsize'])
                    else:
                        unit = GLOBAL['PARAM_UNIT_DIC'][param]
                        ax[idx_param, idx_trt].set_ylabel(f'{param} ({unit})',
                                                          fontsize=plot_params_dict['labels_fontsize'])
            
                    
                ax[idx_param, idx_trt].tick_params(axis="x", rotation=90)
                ax[idx_param, idx_trt].set_xticks(list_irr, minor=False)
                ax[idx_param, idx_trt].set_xticklabels(list_irr, fontsize=plot_params_dict['ticks_fontsize'])
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
               loc='center left',
               bbox_to_anchor=(plot_params_dict['bbox_x0'],
                               plot_params_dict['bbox_y0']),
                               #plot_params_dict['bbox_width'],
                               #plot_params_dict['bbox_height']),
               bbox_transform=plt.gcf().transFigure,
               fontsize=plot_params_dict['legend_fontsize']
     )
    plt.show()
    
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
        df_meta_cp_end = df_meta.query("module_type==@module_type & treatment==@treatment[0] ").sort_values('irradiance')
        df_meta_cp_deb = df_meta.query("module_type==@module_type & treatment==@treatment[1] ").sort_values('irradiance')
        val = np.array(df_meta_cp_end[param].tolist())
        ref = np.array(df_meta_cp_deb[param].tolist())
        try:
            x = df_meta_cp_end['irradiance'].tolist() 
            y = 100 * (val - ref) / ref
        except:
            x = []
            y = []
        
    return (x,y)
    
def _set_ymin_ymax_param(df_meta, params, list_modules_type, list_trt_diff, diff, limit_type=None):
    
    '''Build a dict keyed by the parameters and which values are list [ymin, ymax] 
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
    if limit_type == 'extremorum': 
        min_minimorum = min([y[0] for y in min_max_param.values()])
        max_maximorum = max([y[1] for y in min_max_param.values()])
    
        min_max_param = {param:[min_minimorum,max_maximorum]
                        for param,y in min_max_param.items()}
   
    return min_max_param

def _set_xmin_xmax(list_irr,irr_add_nbr=1):

    '''
    Set the min and max value use to set the limit of the x axis.
        args:
            list_irr (list or real): list of x values
            irr_add_nbr (real): we add/substract irr_add_nbr*(max(list_irr)-min(list_irr))
                               to max(list_irr)/min(list_irr).
        returns:
            The tuple (irr_min, irr_max)
    '''
    
    xmax = max(list_irr)
    xmin = min(list_irr)
    irr_add = irr_add_nbr * (xmax -xmin)
    if irr_add == 0 : irr_add = 100
    irr_min, irr_max = (
        xmin -  irr_add,
        xmax +  irr_add,
    )
    
    return (irr_min, irr_max)

    
def select_diff_treatment(working_dir,list_mod_selected):
    
    '''Interactivelly builds a list of tuples [(T<i>,T<j>),...] where i>j and T<i> is the ieme treatment.
    
    Args:
       working_dir (str): full path to the experimental values
       list_mod_selected (list of str) : list of the selectected modules
       
    Returns:
       A list of tuples
    '''
    
    # Standard library imports
    from itertools import combinations

    #3rd party imports
    import pandas as pd
    
    # Internal imports
    from .PVcharacterization_database import sqlite_to_dataframe

    df_files_descp = sqlite_to_dataframe(working_dir,GLOBAL['DATA_BASE_TABLE_FILE'])
    list_treatments = df_files_descp.query('module_type in @list_mod_selected').treatment.unique()
    list_treatments.sort()
    if len(list_treatments)==1: raise Exception("Sorry, the number of treatments must be >1 ") 
    list_combinations = list(combinations(list_treatments,2))

    list_diff_treatment = select_items(list_combinations,
                                          'Select the difference',
                                          mode = 'multiple')
    list_diff_treatment = [(x[1],x[0]) for x in list_diff_treatment]

    return list_diff_treatment
    
def select_params(list_params = None):

    '''Selection of param to be chosen. If list_mod_selected is None the parameter are selected
    interactivelly otherwise ist_params container the list of parameters.
    '''
    COL_NAMES = GLOBAL['COL_NAMES']

    list_allowed_params = list(COL_NAMES)
    list_allowed_params.insert(list_allowed_params.index("Isc")+1, 'Isc_corr')
    list_allowed_params.insert(list_allowed_params.index("Fill Factor")+1, 'Fill Factor_corr')
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
            
    return list_params


def plot_params_diff(df_meta,
                     list_diff,
                     list_params,
                     dic_trt_meaning=None,
                     long_label=False,
                     plot_params_dict=None):
                     
    '''Plot len(list_param) times len(list_diff graphics) of the parameters/parameters difference  values versus irradiance
    for each module type.
    
    Args:
       df_meta (datframe): dataframe containing all the information about the modules to deal with
       list_diff (list of tuples): list of tuples (Ti,Tj) we plot param((Ti)-param(Tj))/param(Tj)
       list_params (list of str):list of params to plot. If None the list of parameter is built interactivelly
       dic_trt_meaning (dict): {Ti: explicit value of Ti}
       long_label: False we truncate the labels
       plot_params_dict (dict): defines the ploting parameters
       
    
    '''
    
    
    #3rd party imports
    import pandas as pd
    
    PLOT_PARAMS_DICT = GLOBAL['PLOT_PARAMS_DICT']
    NBR_MAX_PARAMS_PLOT = len(GLOBAL['PLOT_PARAMS_DICT']['markers'])
    TREATMENT_DEFAULT_LIST = GLOBAL['TREATMENT_DEFAULT_LIST']
    
    assert len(list_params)>1, "The number of parameters must be greater than one"

    list_modules = pd.unique(df_meta['module_type'])          # List of modules name (ID)
    if len(list_modules)>NBR_MAX_PARAMS_PLOT:
        list_modules = list_modules[0:NBR_MAX_PARAMS_PLOT]
        print(f'WARNING: to much modules. Only the first {NBR_MAX_PARAMS_PLOT} will be plotted.')
        
    if plot_params_dict is None:
        plot_params_dict = PLOT_PARAMS_DICT
    else:
        plot_params_dict = plot_params_dict
        
    if dic_trt_meaning is None:
        dic_trt_meaning = {trt:trt for trt in TREATMENT_DEFAULT_LIST}
        
    _plot_params(list_params,
                list_modules,
                df_meta,
                plot_params_dict,
                list_diff = list_diff,
                dic_trt_meaning=dic_trt_meaning,
                long_label=long_label,) 
    
def plot_iv_curves(irr_select,name_select,trt_select,working_dir):

    '''
    Plot of the I/V curves of the modules type with: names in the list name_select,
    tratment in the list, trt_select and irradiance in the list irr_select.

    Args:
        irr_select (list of int): list of irradiance to be plotted
        name_select (list of str): list of module type names to be plotted
        trt_select (list of str): list of treatments to be plotted
        working_dir (str): the folder containing the database     '''
    
    # Standard library imports
    from pathlib import Path

    # 3rd party imports
    import pandas as pd
    import plotly.express as px
    
    DATA_BASE_NAME = GLOBAL['DATA_BASE_NAME']

    database_path = Path(working_dir) / Path(DATA_BASE_NAME)

    list_files_path = sieve_files(irr_select,trt_select,name_select,database_path)
    list_dataframe = []
    for file in list_files_path:
        parse_file = parse_filename(file)
        df_IV = read_flashtest_file(file).IV0
        df_IV['module'] = parse_file.module_type
        df_IV['treatment'] = parse_file.treatment
        df_IV['irradiance'] = parse_file.irradiance
        df_IV['exp'] = f'{parse_file.irradiance}-{parse_file.treatment}-{parse_file.module_type}'
        list_dataframe.append(df_IV)
    df_all_IV = pd.concat(list_dataframe)


    fig = px.line(df_all_IV,
                  x="Voltage",
                  y="Current",
                  color='exp',
                  labels={'Voltage':'Voltage (V)',
                          'Current':'Current (A)'})
    fig.show()
    
def plot_iv_power(file=None):

    # Standard library imports
    import os
    from pathlib import Path
    
    # 3rd party import
    import matplotlib.pyplot as plt
    import numpy as np
    
    PARAM_UNIT_DIC = GLOBAL['PARAM_UNIT_DIC']

    if file is None:
        file = select_files()
        file = file[0]
    else:
        file=file
        
    answ = read_flashtest_file(file, parse_all=True) # Parse= True to retrieve IV curves
    voltage = answ.IV0["Voltage"]
    current = answ.IV0["Current"]
    corrected_current = correct_iv_curve(voltage,current)

    power = voltage * current
    power_max = max(power)
    Vpm = voltage[np.argmax(power)]
    Ipm = current[np.argmax(power)]

    fig = plt.figure(figsize=(10,10))
    _ = plt.plot(voltage,current,label="Raw data")
    _ = plt.plot(voltage,corrected_current,label="Corrected data")
    _ = plt.scatter([0],[answ.meta_data['Isc']],c='r',marker='+',s=50,label="Isc(0) manufacturer")

    _ = plt.plot([0,Vpm],[Ipm,Ipm], color='red', linestyle='--')
    _ = plt.plot([Vpm,Vpm],[0,Ipm], color='red', linestyle='--')

    plt.xlim(-0.2,max(voltage))
    plt.ylim(0,1*current[0] )

    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.legend( bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.title(os.path.basename(file))

    ax = plt.gca()
    ax2 = ax.twinx()
    _ = ax2.plot(voltage,power,'--k',label='Maufacturer power')
    _ = ax2.scatter(voltage[::50] ,voltage[::50]*corrected_current[::50],label='PVcharacterization power')
    _ = ax2.plot([Vpm,max(voltage)],[power_max,power_max], color='red', linestyle='--')
    ax2.set_ylabel('Power (W)')
    ax2.set_ylim(0,1.2*power_max)

    ax2.legend( bbox_to_anchor=(1.05, 1), loc='lower left', borderaxespad=0.)
    
    print(f'Filename: {file}\n')
    print('PV characterization Data:')
    print(f'Pmax={power_max:.2f} W\nIsc={current[0]:.2f} A, Isc_corr={corrected_current[0]:.2f} A' )
    print(f'Voc={max(voltage):.2f} V' )
    print(f'Fill Factor={max(power)/(current[0]*max(voltage)):.3f}, corrected Fill Factor={max(power)/(corrected_current[0]*max(voltage)):.3f}')
    print(f'Vpm={Vpm:.2f} V' )
    print(f'Ipm={Ipm:.2f} V' )

    print('\nManufacturer data :')
    for param in  ['Pmax','Isc','Voc','Fill Factor','Vpm','Ipm']: 
        print(f'{param}={answ.meta_data[param] :.2f} {PARAM_UNIT_DIC[param]}')
        

