import matplotlib.pylab as plt
import numpy as np

def draw_spike_waveform(wi, maxChan, xc, yc, ax=None, nc=16,n_chan=383,scale = 10, fs=30000, alpha=1, shift=0, color='k',**kwargs):
    """
    Draw spike waveforms on a probe layout.
    This function plots spike waveforms centered around the maximum channel, overlaying them
    on the physical coordinates of the recording channels. Waveforms are normalized and 
    scaled for visualization.
    Parameters
    ----------
    wi : ndarray
        Waveform data with shape (time x channel). Should contain the spike waveform
        sampled across multiple channels.
    maxChan : int
        Index of the channel with maximum spike amplitude (peak channel).
    xc : ndarray
        X-coordinates of all channels on the probe.
    yc : ndarray
        Y-coordinates of all channels on the probe.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure with 6x6 inch size.
    nc : int, optional
        Number of channels to display around the peak channel. Default is 16.
    n_chan : int, optional
        Total number of channels in the recording. Default is 383.
    scale : float, optional
        Scaling factor for waveform amplitude normalization. Default is 10.
        The actual voltage represented in the scale bar scale * 5 microvolts per unit.
    alpha : float, optional
        Transparency level for plotted waveforms (0-1). Default is 1.
    shift : int, optional
        Offset to apply to channel indices. Default is 0.
    color : str, optional
        Color for waveform traces (except peak channel which is red). Default is 'k' (black).
    **kwargs : dict
        Additional keyword arguments passed to matplotlib plot function.
    Returns
    -------
    None
        Modifies the axes object in place by adding waveform plots and a scale bar.
    Notes
    -----
    - The peak channel waveform is highlighted in red, others use the specified color.
    - A green scale bar is added showing the voltage scale (scale * 5 µV).
    - Waveforms are plotted in spatial coordinates matching the probe geometry.
    - Time axis is normalized to span the width of a contact (20 units).
    """
    '''
    wi should be in the shape (time x channel)
    Note: the spikes are first normalized to their peak-to-peak amp before plotting
    '''
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6,6))
    
    # location of the waveform
    ic0 = max(0, maxChan-nc//2)
    ic1 = min(n_chan, maxChan+nc//2)
    x0, y0 = xc[ic0:ic1], yc[ic0:ic1]
    x_max, y_max = xc[maxChan], yc[maxChan]

    idx = np.arange(ic0, ic1)+ shift
    wv = wi[:, idx]
    wv /= scale # the spike is already normalized before plotting
    
    for ii, (xi,yi) in enumerate(zip(x0,y0)):
        t = np.arange(-wv.shape[0]//2,wv.shape[0]//2,1,'float32')
        t /= wv.shape[0] / 20 #width of the contact
        if xi==x_max and yi==y_max:
          ax.plot(xi + t, yi + wv[:,ii], color='r',alpha=alpha,**kwargs)
        else:
          ax.plot(xi + t, yi + wv[:,ii], color=color,alpha=alpha,**kwargs)
        
    # ax.plot([x0[0], x0[0] + 20], [y0[-1] - 20, y0[-1] - 20], 'r-', linewidth=2, label='20 µm')
    xlim = ax.get_xlim()
    scale_bar_x = xlim[0]+5
    ylim = ax.get_ylim()
    scale_bar_y = ylim[0] + 5
    
    ax.plot([scale_bar_x, scale_bar_x], [scale_bar_y, scale_bar_y + 5], 'g-', linewidth=2)
    ax.text(scale_bar_x, scale_bar_y+6, f'{scale*5} uV', fontsize=10, va='center')
    
    # Draw horizontal scale bar for time (20 µm horizontal distance equals waveform width)
    # Horizontal scale bar for time
   
    time_bar_samples = int(fs/1000)  # number of samples for 1ms at 30kHz (fs/2)
    time_bar_um = time_bar_samples / (wv.shape[0] / 20)  # convert to spatial units
    # time_bar_ms = (time_bar_samples / fs) * 1000  # convert to milliseconds
    ax.plot([scale_bar_x, scale_bar_x + time_bar_um], [scale_bar_y, scale_bar_y], 'g-', linewidth=2)
    ax.text(scale_bar_x + time_bar_um/2, scale_bar_y - 2, f'1 ms', fontsize=10, ha='center')
    

def plot_template(template, max_chan, chan_locs):
    """
    Plot spike template waveforms in two different representations.
    This function creates a visualization of a spike template using two subplots:
    1. A heatmap/image representation of the template across all channels
    2. A spatial representation showing waveforms at their physical channel locations
    Parameters
    ----------
    template : numpy.ndarray
        2D array containing the spike template waveform data.
        Shape should be (n_timepoints, n_channels).
    max_chan : int
        Index of the channel with maximum amplitude in the template.
    chan_locs : numpy.ndarray
        2D array containing the physical locations of channels.
        Shape should be (n_channels, 2) where columns represent x and y coordinates.
    Returns
    -------
    None
        The function displays the plot using matplotlib but does not return any value.
    Notes
    -----
    - The right subplot shows the waveforms positioned at their physical locations.
    """
    
        
    fig, ax = plt.subplots(1,2, figsize=(4*2, 8))
    ax[0].imshow(template.T)
    ax[0].invert_yaxis()

    draw_spike_waveform(template, max_chan, chan_locs[:,0], chan_locs[:,1], ax[1])

    plt.tight_layout()
    
    return fig, ax