# Utility functions for pycontrol and pyphotometry files processing

import numpy as np
import pandas as pd
from scipy import signal
import av
import matplotlib.pylab as plt
import xarray as xr
from trialexp.process.pyphotometry.utils import extract_event_data
from moviepy.editor import *
import threading
from moviepy.editor import *
from moviepy.video.io.bindings import mplfig_to_npimage
import subprocess
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


def interpolate_bad_points(df, threshold, max_correction_ratio=0.3):
    df = df.copy()
    
    # do a sanity check here, if more than max_correction_ratio needs to be removed, refuse the correction
    ratio2remove = (df.likelihood<threshold).mean()
    if ratio2remove < max_correction_ratio:
        #likelihood value below the threshold will be removed and replaced by interpolation
        df.loc[df.likelihood<threshold,:] = None
        # return df.interpolate(method='nearest')
        df =  df.fillna(method='ffill')
        df = df.fillna(method='bfill') #avoid error in lowpass filtering later
    
    return df

def lowpass_coords(df, fs, corner_freq):
    df = df.copy()
    # use a low pass filter to filter the coordinates
    [b,a] = signal.butter(3, corner_freq/(fs/2))
    df['x'] = signal.filtfilt(b,a,df['x'])
    df['y'] = signal.filtfilt(b,a,df['y'])
    
    return df

def preprocess_(df, threshold=0.7, dlc_fps=100, lowpass_freq=20):
    df = interpolate_bad_points(df,threshold)
    df = lowpass_coords(df,dlc_fps, lowpass_freq)
    return df

def filter_and_merge(df,threshold=0.7):
    markers = df.columns.get_level_values(0).unique()
    
    # remove all value lower than a certain threshold
    idx = df.loc[:, (slice(None), 'likelihood')]<threshold
    for m in markers:
        df.loc[idx[m]['likelihood'], (m, slice(None))] = None
    dfmean = df.groupby(level='coords', axis=1).mean()
    
    return dfmean


def copy_coords(df, new_df, marker_name):
    #copy coordinates from new df to old df with multindex column
    #note: this will modify the df inplace
    for c in ['x','y','likelihood']:
        df[(marker_name,c)] = new_df[c]
        
def preprocess_dlc(df):
    # df is the dataframe from DLC, with multiindex columns
    df  = df.copy()
    markers = df.columns.get_level_values(0).unique()
    
    for m in markers:
        # do the preprocessing on each marker, and copy the results back to the original
        # multindex dataframe
        marker_coords = df.xs(m,level='bodyparts',axis=1)
        marker_coords = preprocess_(marker_coords)
        copy_coords(df, marker_coords,m)
        
    return df

def dlc2xarray(df_dlc):
    #convert the DLC results structure to xarray
    data = df_dlc.values
    data = data.reshape(-1,data.shape[1]//3,3)
    bodyparts = df_dlc.columns.get_level_values(0).unique()
    
    da = xr.DataArray(
        data,
        dims = ('time','bodyparts','coords'),
        coords = {'bodyparts': bodyparts,
                  'time':df_dlc.index,
                  'coords':['x','y','likelihood']}
    )
    
    return da

def plot_event_video(t, starting_time, time_span, events2plot, clip, marker_signal:list, photo_signal):

    # events2plot should contains all the events that needs to be plotted
    
    x_start_time = starting_time + t//time_span*time_span
    x_end_time = x_start_time+time_span

    ax1 = plt.subplot2grid((3, 2), (0, 0))
    ax2 = plt.subplot2grid((3, 2), (1, 0), sharex=ax1)
    ax3 = plt.subplot2grid((3, 2), (2, 0), sharex=ax1)
    ax4 = plt.subplot2grid((3, 2), (0, 1), rowspan=3)

    plot_axes = [ax1,ax2,ax3]
    
    
    ## Event
    evt_colours =['r','g','b','w']
    for i, event in enumerate(events2plot.name.unique()):
        evt_time = events2plot[events2plot.name==event].time/1000
        evt_time = evt_time[(evt_time>x_start_time) & (evt_time<=x_end_time)]
        ax1.eventplot(evt_time, lineoffsets=80+20*i, linelengths=20,label=event, color=evt_colours[i])
    
    
    ## Speed data
    (signal_time, coords, likelihood, speed) = marker_signal[0] # only show the speed for the first marker
    
    
    ax1.plot(signal_time, speed, label='speed')
    ax1.legend(loc='upper left', prop = { "size": 7 }, ncol=4)
    ax1.set_ylim([0, 200])
    
    
    ## Marker coordinates
    idx2plot = (signal_time > x_start_time) & (signal_time<x_start_time+time_span)
    ax2.plot(signal_time[idx2plot], coords[idx2plot, 0], label='x')
    ax2.plot(signal_time[idx2plot], coords[idx2plot,1], label='y')
    ax2.legend(loc='upper left', prop = { "size": 7 }, ncol=4)
    ax2.set_ylabel('Tracker coords')
    ax2.set_ylim([200, 1200])

    
    ## Photometry signal
    ax3.plot(signal_time[idx2plot], photo_signal[idx2plot])
    ax3.set_ylabel('zscored df/f')

    ax1.set(xlim=(x_start_time,x_start_time+time_span))
    
    #plot the line for the curren time
    for ax in plot_axes:
        ax.axvline(starting_time+t,color='y')
    ax3.set_xlabel('Time (s)')
    
    ## Video
    ax4.imshow(clip.get_frame(starting_time+t))
    ax4.axis('off')
    
    ## Plot the marker location on the video too
    marker_list = ['ro','g*','w+']
    for i, (signal_time, coords,likelihood,speed) in enumerate(marker_signal):
        # print(likelihood)
        idx = np.argmin(abs(signal_time - (starting_time+t))) # use time to find the correct index to plot
        ax4.plot(coords[idx,0], coords[idx,1], marker_list[i%len(marker_list)], alpha=likelihood[idx])
    
    plt.subplots_adjust(hspace=0.3)
    



def make_sync_video(videofile_path:str, output_video_path:str, xr_session, df_pycontrol,
                    bodypart:list,
                    duration=20, start_time=80):
    # Create video with photometry data for verificationn

    plt.style.use("dark_background")

    marker = xr_session['dlc_markers'].sel(bodyparts=bodypart)
    marker_signal = marker2dataframe(marker)
    photo_signal = xr_session.zscored_df_over_f.data[0]
    
    events2plot = df_pycontrol[df_pycontrol.name.isin(['hold_for_water', 'spout','bar_off'])]

    def make_frame(t):
        plot_event_video(t, start_time, 20, events2plot, clip, marker_signal, photo_signal)
        return mplfig_to_npimage(plt.gcf())

    clip = VideoFileClip(videofile_path)

    plt.figure(figsize=(10,5), dpi=100)
    animation_fps = 25
    animation = VideoClip(make_frame, duration=20)
    animation.write_videofile(output_video_path, fps=animation_fps, threads=20)

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


def plot_rsync(rsync_time, frames):
    # write documentation
    
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
        
def get_marker_signal(marker_loc):    
    signal_time = marker_loc.time.data/1000
    coords = marker_loc.data
    speed = np.sqrt(np.sum(np.diff(coords,axis=0, prepend=[coords[0,:]])**2,axis=1))
    
    return (signal_time, coords, speed)


def marker2dataframe(marker_loc):
    
    df_list = []
    for part in marker_loc.bodyparts:
        loc = marker_loc.sel(bodyparts=part)
        # convert the xarray format to dataframe for easier manipulation
        signal_time = loc.time.data/1000
        coords = loc.data[:,[0,1]]
        likelihood = loc.data[:,2]
        speed = np.sqrt(np.sum(np.diff(coords,axis=0, prepend=[coords[0,:]])**2,axis=1))

        df_list.append((signal_time, coords, likelihood, speed))
    return df_list

def get_movement_metrics(marker_loc):
    signal_time = marker_loc.time.data/1000
    coords = marker_loc.data[:,:2]
    likelihood = marker_loc.data[:,2]
    speed = np.diff(coords,axis=0, prepend=[coords[0,:]])
    accel = np.diff(speed, axis=0, prepend=[speed[0,:]])
    
    return (signal_time, coords, speed, accel, likelihood)
    
def filter_init(df_move, move_init_idx, consec_rest, consec_move):
    # filter move init only if it is proceed by some amount of consec_rest
    # followed by consec_move of movement
    # allow for some tolerance as there may be tracking error
    
    valid_init = []
    for idx in move_init_idx:
        rest_cond = df_move.iloc[(idx-consec_rest):idx].is_rest.mean() > 0.9
        move_cond = df_move.iloc[idx:(idx+consec_move)].is_rest.mean() < 0.1
        if (rest_cond and move_cond):
            valid_init.append(idx)
            
    return valid_init
def add_video_timestamp(df,videofile):
    # add timestamp of the video file to the deeplabcut dataframe
    ts = extract_video_timestamp(videofile)
    df['time'] = ts
    df = df.set_index('time')
    return df


def extract_triggered_data(event_time, xr_session, trial_window, sampling_rate):
    event_period = (trial_window[1] - trial_window[0])/1000
    event_time_coord= np.linspace(trial_window[0], trial_window[1], int(event_period*sampling_rate)) #TODO
    
    data, event_found = extract_event_data(event_time, trial_window, xr_session['zscored_df_over_f'], sampling_rate)
    # print(data.shape)
    print(f'Extracted {np.sum(event_found)} events')
    da = xr.DataArray(
            data, coords={'event_time':event_time_coord,
                         'event_index': np.arange(data.shape[0])},
                            dims=('event_index','event_time'))


    df = da.to_dataframe(name='photometry').reset_index()
    return df



def accelerate_forward(df_move, accel_threshold, speed_threshold):
    # accelerate forward, from slow speed
    df = df_move[(df_move.accel>accel_threshold) & (df_move.speed_x<0) & (df_move.speed<speed_threshold)]    
    return df

def accelerate_backward(df_move, accel_threshold, speed_threshold):
    # accelerate forward, from slow speed
    df = df_move[(df_move.accel>accel_threshold) & (df_move.speed_x>0) & (df_move.speed<speed_threshold)]    
    return df

def deccelerate_forward(df_move, accel_threshold, speed_threshold):
    # accelerate forward, from slow speed
    df = df_move[(df_move.accel<accel_threshold) & (df_move.speed_x<0) & (df_move.speed<speed_threshold)]    
    return df

def deccelerate_backward(df_move, accel_threshold, speed_threshold):
    # accelerate forward, from slow speed
    df = df_move[(df_move.accel<accel_threshold) & (df_move.speed_x>0) & (df_move.speed<speed_threshold)]    
    return df

def extract_trigger_acceleration(df_move, accel_threshold, speed_threshold, direction, accel_type):
    if accel_type=='accel':
        if direction == "forward":
            df = df_move[(df_move.accel > accel_threshold) & (df_move.speed_x < 0) & (df_move.speed < speed_threshold)]
        else:
            df = df_move[(df_move.accel > accel_threshold) & (df_move.speed_x > 0) & (df_move.speed < speed_threshold)]
    else:
        if direction == "forward":
            df = df_move[(df_move.accel < -accel_threshold) & (df_move.speed_x < 0) & (df_move.speed > speed_threshold)]
        else:
            df = df_move[(df_move.accel < -accel_threshold) & (df_move.speed_x > 0) & (df_move.speed > speed_threshold)]
    return df

# plot a clip of the movie
def extract_sample_video(videofile, df, fn,num=5):
    for i in range(num):
        t = df.iloc[i].time/1000
        clip = VideoFileClip(videofile).subclip(t-1,t+1)
        clip.write_videofile(f'sample_video/{fn}_{i}.mp4',fps=60, 
                             threads=5)
        
        

def extract_video(videofile, fn_prefix, output_path,  t, video_type='mp4', resize_ratio=1,logger='bar', pretime =1 , posttime=1):
    # time should be in miliseconds
    
    t = t/1000
    clip = VideoFileClip(videofile).subclip(t-1,t+1).resize(resize_ratio)
    if video_type =='mp4':
        clip.write_videofile(f'{output_path}/{fn_prefix}_{int(t)}.mp4',fps=60)
    elif video_type =='gif':
        clip.write_gif(f'{output_path}/{fn_prefix}_{int(t)}.gif', fps=15,logger=logger)
    
    print(f'{fn_prefix}_{int(t)}')
    
def dlc2movementdf(xr_session, marker_loc):
    # convert marker location to speed and acceleration data

    signal_time, coords, speed, accel,likelihood = get_movement_metrics(marker_loc)
    speed_mag = np.linalg.norm(speed,axis=1)
    accel_mag = np.diff(speed_mag, prepend=speed_mag[0])

    f = xr_session.zscored_df_over_f.data[0]

    df_move = pd.DataFrame({
        'accel': accel_mag,
        'accel_x': accel[:,0],
        'accel_y': accel[:,1],
        'speed': speed_mag,
        'speed_x': speed[:,0],
        'speed_y': speed[:,1],
        'x' : coords[:,0],
        'y' : coords[:,1],
        'likelihood': likelihood,
        'time': xr_session.time,
        'df/f': f})
    
    is_moving = (df_move.speed>5)
    is_rest = ((df_move.speed<2) & (df_move.accel.abs()<3)).astype(np.int8)
    df_move['is_rest'] = is_rest
        
    return  df_move

def get_valid_init(df_move):
    # find the time for movement initiation

    move_init_idx = np.where(np.diff(df_move.is_rest, prepend=False)==-1)[0]
    valid_init = filter_init(df_move, move_init_idx,50, 10)
    valid_init_time = df_move.iloc[valid_init].time
    
    return valid_init, valid_init_time
            
def extract_sample_video_multi(videofile, fn_prefix, output_path, time, video_type='mp4',resize_ratio=1,  pretime =1 , posttime=1):
    # Use multi-threading to speed up extraction of videos
    threads = []
    for i in range(len(time)):
        thread = threading.Thread(target=extract_video, 
                                  args=(videofile, fn_prefix, output_path, time[i], video_type,resize_ratio,None,pretime, posttime))
        thread.start()
        threads.append(thread)
        
    for thread in threads:
        thread.join()
        

def get_direction(df_move, move_init_idx, window=10, win_dir = 'after'):
    direction = []
    for idx in move_init_idx:
        if win_dir == 'after':
            speed_x_mean =df_move.iloc[(idx):(idx+window)].speed_x.mean()
        else:
            speed_x_mean =df_move.iloc[(idx-window):(idx)].speed_x.mean()
        if speed_x_mean <0:
            d = 'forward'
        else:
            d = 'backward'
        
        direction.append(d)
    return direction

def get_average_speed(df_move, move_init_idx, window=10, win_dir = 'after'):
    speed = []
    for idx in move_init_idx:
        if win_dir == 'after':
            speed_mean =df_move.iloc[(idx):(idx+window)].speed.mean()
        else:
            speed_mean = df_move.iloc[(idx-window):(idx)].speed.mean()
        speed.append(speed_mean)
    return speed

def get_average_value(df_move,col_name, move_init_idx, window=10, win_dir='after'):
    data = []
    for idx in move_init_idx:
        if win_dir == 'after':        
            d =df_move.iloc[(idx):(idx+window)][col_name].mean()
        else:
            d =df_move.iloc[(idx-window):idx][col_name].mean()
        data.append(d)
    return data

def get_average_photom(df_move, move_init_idx, window=10, win_dir='after'):
    photom = []
    for idx in move_init_idx:
        if win_dir == 'after':        
            d =df_move.iloc[(idx):(idx+window)]['df/f'].mean()
        else:
            d =df_move.ilociloc[(idx-window):idx]['df/f'].mean()
        photom.append(d)
    return photom

def get_movement_type(df_move, move_init_idx, threshold, window=10, win_dir='after'):
    # get the starting pos and ending pos to find the displacement
    mov_type = []
    for idx in move_init_idx:
        if win_dir == 'after':
            start_pos = df_move.iloc[idx].x
            end_pos = df_move.iloc[idx+window].x
        else:
            start_pos = df_move.iloc[idx - window].x
            end_pos = df_move.iloc[idx].x
            
        displacement = end_pos - start_pos
        
        if abs(displacement)>threshold:
            mov_type.append('reach')
        else:
            mov_type.append('twitch')
        
    return mov_type



def rescale_video(
    video_path,
    output_path,
    width,
    height=-1,
    rotatecw="No",
    angle=0.0,
    frame_rate = 30,
    suffix="rescale",
):
    '''
    Adapted from DLC code, but force the frame rate to be the same
    '''
    command = (
        f"ffmpeg -n -loglevel fatal -i {video_path} -filter:v "
        f'"scale={width}:{height}{{}}" -r {frame_rate} -c:a copy {output_path}'
    )
    # Rotate, see: https://stackoverflow.com/questions/3937387/rotating-videos-with-ffmpeg
    # interesting option to just update metadata.
    if rotatecw == "Arbitrary":
        angle = np.deg2rad(angle)
        command = command.format(f", rotate={angle}")
    elif rotatecw == "Yes":
        command = command.format(f", transpose=1")
    else:
        command = command.format("")
    subprocess.call(command, shell=True)
    return output_path