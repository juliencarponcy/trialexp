import matplotlib.pylab as plt

def plot_and_handler_error(plot_func, **kwargs):
    # helper function to handle the error when a certain facet is nil
    if len(kwargs['data'].dropna())>0:
        plot_func(**kwargs)
  
def annotate_trial_number(data, **kwargs):
    ax = plt.gca()
    total_trial_nb = len(data['trial_nb'].unique())
    ax.text(0.1,0.9,f'n_trials={total_trial_nb}', transform=ax.transAxes)