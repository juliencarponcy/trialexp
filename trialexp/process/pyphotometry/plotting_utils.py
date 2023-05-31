import matplotlib.pylab as plt
import numpy as np 

def plot_and_handler_error(plot_func, **kwargs):
    # helper function to handle the error when a certain facet is nil
    if len(kwargs['data'].dropna())>0:
        plot_func(**kwargs)
  
def annotate_trial_number(data, **kwargs):
    ax = plt.gca()
    total_trial_nb = len(data.dropna()['trial_nb'].unique())
    ax.text(0.8,0.9,f'n_trials={total_trial_nb}', transform=ax.transAxes)
    
    #add vertical line for the trigger
    ax.axvline(0,ls='--', color='k', alpha=0.5)
    
    
def plot_pyphoto_heatmap(dataArray):
    # calculate the proper color scale
    plt.figure(figsize=(4,4), dpi=300)
    x = dataArray.data
    x = x[~np.isnan(x)]
    vmax = np.percentile(x,99)
    vmin = np.percentile(x,1)
    
    quadMesh = dataArray.plot(vmax=vmax, vmin=-vmax, cmap='vlag')
    
    return quadMesh.figure