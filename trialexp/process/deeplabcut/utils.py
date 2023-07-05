# Utility functions for pycontrol and pyphotometry files processing

import numpy as np
import pandas as pd
from scipy import signal
import av
import matplotlib.pylab as plt
#----------------------------------------------------------------------------------
# Plotting
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
# Helpers
#----------------------------------------------------------------------------------

def get_regions_to_store(bodyparts_to_ave, names_of_ave_regions, bodyparts_to_store):
    '''
    determine which regions to store in a coordinates dict, based on the parameters
    used for get_deeplabcut_trials() method
    '''
    
    
    if names_of_ave_regions or bodyparts_to_store:
        if names_of_ave_regions and bodyparts_to_store:
            regions_to_store = names_of_ave_regions + bodyparts_to_store
        elif names_of_ave_regions and not bodyparts_to_store:
            regions_to_store = bodyparts_to_ave
        elif not names_of_ave_regions and bodyparts_to_store:
            regions_to_store = bodyparts_to_store
    return regions_to_store

#----------------------------------------------------------------------------------
# Processing helper
#----------------------------------------------------------------------------------

def normalize_coords(coord_dict, normalize_betwen=['Left_paw','spout'], bins_nb=200):
    '''
    Get the coordinates of maximum density of two regions in order to normalize trajectories.
    Only for 2D for now.
    coord_dict is a dictionary which keys are regions computed, and values are X-Y ndarrays
    return the coordinates normalized between the coords of max density of two regions.
    normalize_betwen is a 2 items list which state the start and stop region
    to normalize between.
    bins_nb is the number of bins used to compute the np.histogram2d functions.
    The trade-off for bins_nb: too high value will only have a few timestamps
    in a bin, leading to poor aggregation and then random-ish maximum coord.
    Values too low will lead to a good aggregation but much less pixel-wise
    precision.
    Used by session.get_deeplabcut_trials()
    '''
    if len(normalize_betwen) != 2:
        raise Exception('normalize_betwen must be a list of two regions (str)')
    
    min_max_coord = np.ndarray((2,2))
    for idx_r, region in enumerate(normalize_betwen):
        nan_free_coords = np.delete(coord_dict[region], np.isnan(coord_dict[region][:,0]),0)
        xmin = nan_free_coords[:,0].min()
        xmax = nan_free_coords[:,0].max()
        ymin = nan_free_coords[:,1].min()
        ymax = nan_free_coords[:,1].max()

        H, xedges, yedges = np.histogram2d(coord_dict[region][:,0],coord_dict[region][:,1], 
            bins=bins_nb , range=[[xmin, xmax], [ymin, ymax]])

        ind = np.unravel_index(np.argmax(H, axis=None), H.shape)
        min_max_coord[idx_r,:] = [xedges[ind[0]],yedges[ind[1]]]

    rangeXY = [min_max_coord[1,0] - min_max_coord[0,0], min_max_coord[1,1] - min_max_coord[0,1]]

    norm_coord_dict = dict()
    for region in coord_dict.keys():
        norm_coord_dict[region] = np.ndarray(shape=coord_dict[region].shape)
        norm_coord_dict[region][:,0] = (coord_dict[region][:,0]-min_max_coord[0,0]) / rangeXY[0]
        norm_coord_dict[region][:,1] = (coord_dict[region][:,1]-min_max_coord[0,1]) / rangeXY[1]

    return norm_coord_dict


def merge_marker(df):
    # merge marker and choose the one with max likelihood
    max_likehood_idx = df.groupby(level='coords').idxmax()['likelihood']
    # print(max_likehood_idx[0])
    return df[(max_likehood_idx[0], slice(None))]

def merge_marker_likelihood(df):
    # fast implementation using numpy to merge markers
    # other implementation using indexing in pd row by row is too slwo
    
    # find which marker has the larger likelihood
    likelihood = df.loc[:,(slice(None), 'likelihood')].values
    maxidx = np.argmax(likelihood,axis=1)
    
    # select data from that marker
    data = df.values.reshape(len(df),-1,3) # rearrange to (frame, marker, coords)
    data = data[np.arange(len(df)),maxidx,:]
    
    #recreate the dataframe
    return pd.DataFrame(data,columns=['x','y','likelihood'])


def interpolate_bad_points(df, threshold):
    df = df.copy()
    #likelihood value below the threshold will be removed and replaced by interpolation
    df.loc[df.likelihood<threshold,:] = None
    # return df.interpolate(method='nearest')
    return df.fillna(method='ffill')

def lowpass_coords(df, fs, corner_freq):
    df = df.copy()
    # use a low pass filter to filter the coordinates
    [b,a] = signal.butter(3, corner_freq/(fs/2))
    df['x'] = signal.filtfilt(b,a,df['x'])
    df['y'] = signal.filtfilt(b,a,df['y'])
    
    return df


# use pyav to extract the timestamp of each frame
def extract_video_timestamp(video: str, index: int = 0):
    """
    adapted from https://stackoverflow.com/questions/47743246/getting-timestamp-of-each-frame-in-a-video?utm_source=pocket_saves

    Parameters:
        video (str): Video path
        index (int): Stream index of the video.
    Returns:
        List of timestamps in ms
    """
    container = av.open(video)
    video = container.streams.get(index)[0]

    if video.type != "video":
            raise ValueError(
                f'The index {index} is not a video stream. It is an {video.type} stream.'
            )

    av_timestamps = [
        int(packet.pts * video.time_base * 1000) for packet in container.demux(video) if packet.pts is not None
    ]

    container.close()
    av_timestamps.sort()

    return av_timestamps


def plot_rsync(rsync_time):
    # rsync_time: timestamp in ms of the rsync signal
    plot_num = 4
    fps = 100

    fig,ax = plt.subplots(plot_num,2,figsize=(3*2,3*plot_num))

    sync_time = rsync_time[1:plot_num+1]

    for i,t in enumerate(sync_time):
        sync_pt = int(t/1000*fps)


        ax[i][0].imshow(frames[sync_pt-1,:,:],cmap='gray')
        ax[i][1].imshow(frames[sync_pt+1,:,:],cmap='gray')

        ax[i][0].axis('off')
        ax[i][1].axis('off')