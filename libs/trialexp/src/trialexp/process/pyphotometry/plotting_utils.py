import matplotlib.pylab as plt
import numpy as np 
from trialexp.process.pycontrol.plot_utils import trial_outcome_palette

def plot_and_handler_error(plot_func, **kwargs):
    # helper function to handle the error when a certain facet is nil
    if len(kwargs['data'].dropna())>0:
        trial_outcome = kwargs['data']['trial_outcome'].iloc[0]
        kwargs['color'] = trial_outcome_palette[trial_outcome]
        plot_func(**kwargs)
  
def annotate_trial_number(data, **kwargs):
    ax = plt.gca()
    total_trial_nb = len(data.dropna()['trial_nb'].unique())
    ax.text(0.8,0.9,f'n_trials={total_trial_nb}', transform=ax.transAxes)
    
    #add vertical line for the trigger
    ax.axvline(0,ls='--', color='k', alpha=0.5)
    
    
def plot_pyphoto_heatmap(dataArray):
    # calculate the proper color scale
    fig = plt.figure(figsize=(4,4), dpi=300)
    x = dataArray.data
    x = x[~np.isnan(x)]
    ndim = np.squeeze(dataArray.data).ndim
    
    if len(x)>0 and ndim>1:
        vmax = np.percentile(x,99)
        vmin = np.percentile(x,1)
        
        quadMesh = dataArray.plot(vmax=vmax, vmin=-vmax, cmap='vlag')

        #NOTE xarray's built-in heatmap has pointy color map; replace this to a normal rectangle
        # Cannot directly change the `extend` attribute of colorbar from DataArray.plot()
        # So creating another one and removing the original

        cbar = quadMesh.colorbar

        ylabel = cbar.ax.get_ylabel()
        cbar.remove()

        new_cbar = plt.colorbar(quadMesh, ax=quadMesh.axes, extend='neither')
        new_cbar.set_label(ylabel)

        return quadMesh.figure
    else:
        return fig