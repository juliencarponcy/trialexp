import xarray as xr
import numpy as np

def get_first_bar_off(df_trial):
    #Find first bar off in trial
    # This function gets a dataframe of a trial, and need to return
    # the row for event of interest
    bar_off =  df_trial[df_trial['content']=='bar_off']
    
    if len(bar_off) >0:
        return bar_off.iloc[0]
    
    
def get_first_spout(df_trial):
    #Find spout touch
    spout =  df_trial[df_trial['content']=='spout']
    
    if len(spout) >0:
        return spout.iloc[0]

def get_last_bar_off_before_first_spout(df_trial):
    # Find the last bar_off before the first spout in trial
    # You can prepare df_trial for debugging/development as below:
    #
    # for i in range(1, np.max(df_event['trial_nb'])):
	#     df_trial = df_event[df_event['trial_nb'] == i]
    # df_trial = the row for event of interest

    bar_off =  df_trial[df_trial['content']=='bar_off']

    spout =  df_trial[df_trial['content']=='spout']
    if len(spout) > 0 and len(bar_off) > 0:
        spout1 = spout.iloc[0]

        filtered_df = bar_off[bar_off['trial_time'] < spout1['trial_time']]
        max_time = filtered_df['trial_time'].max()
        result = filtered_df[filtered_df['trial_time'] == max_time]

        if len(result) >0:
            return result.iloc[0]


def get_first_event_from_name(df_trial, evt_name):
    event =  df_trial[df_trial['content']==evt_name]
    
    if len(event) >0:    
        return event.iloc[0]
    
def get_first_after(df_trial, event1, event2):
    # get the first event1 after event2
    if len(event := df_trial[df_trial['content']==event2]) >0:
        df_window = df_trial[df_trial.time>=event.iloc[0].time]
        return get_first_event_from_name(df_window, event1)
        

def get_events_from_name(df_event, evt_name):
    # return all events with the name
    event =  df_event[df_event['content']==evt_name]
    return event
    
def extract_event_time(df_event, filter_func, filter_func_kwargs, groupby_col='trial_nb'):
    #extract event on a trial based on a filter function
    if groupby_col is not None:
        df_event_filtered = df_event.groupby(groupby_col,group_keys=True).apply(filter_func, **filter_func_kwargs)
    else:
        df_event_filtered = filter_func(df_event, **filter_func_kwargs)
        
    if len(df_event_filtered)>0:
        return df_event_filtered.time
    else:
        #No event found, but still need to return the trial nb info
        if groupby_col is not None:
            return df_event.groupby(groupby_col,group_keys=True)['time'].apply(lambda x: None)
        else:
            return []


# %%
def extract_clean_trigger_event(df_trial, target_event_name, clean_window, ignore_events=None):
    # This function will extract an event with nothing happening (except those in ignore_events) before and after the clean window
    
    # extract all event within the clean_window
    target_events = df_trial[df_trial['content'] == target_event_name]
    
    if len(target_events)>0:
        target_time = target_events.iloc[0].time
    
        idx = (df_trial.time > (target_time+clean_window[0])) & (df_trial.time <(target_time+clean_window[1]))
        
        # Disregard the ignored events
        if ignore_events is not None:
            idx = idx & ~df_trial['content'].isin(ignore_events)

        if sum(idx) ==1 and df_trial.loc[idx].iloc[0]['content'] == target_event_name:
            return df_trial.loc[idx].iloc[0] # must return a series
        
        
def get_reach_travel_time(df_trial):
    # Calculate time between the last bar off and spout touch
    #' to be used '
    # Note: the animal can start the bar off before the cue and then touch after the cue
    
    # only consider reach after the trial start
    df_trial = df_trial[df_trial.trial_time>=0]

    bar_off =  df_trial[df_trial['content']=='bar_off']
    spout =  df_trial[df_trial['content']=='spout']
    
    if len(spout) > 0 and len(bar_off) > 0:
        spout1 = spout.iloc[0]

        filtered_df = bar_off[bar_off['trial_time'] < spout1['trial_time']]
        last_bar_off_time = filtered_df['trial_time'].max()
    
        return spout1.trial_time - last_bar_off_time

def get_last_bar_off_time(df_trial):
    # last bar off before spout touch
    df_trial = df_trial[df_trial.trial_time>=0]

    bar_off =  df_trial[df_trial['content']=='bar_off']
    spout =  df_trial[df_trial['content']=='spout']
    
    if len(spout) > 0 and len(bar_off) > 0:
        spout1 = spout.iloc[0]

        filtered_df = bar_off[bar_off['trial_time'] < spout1['trial_time']]
        last_bar_off_time = filtered_df['trial_time'].max()
        return last_bar_off_time
    
def get_first_bar_off_speed(df_trial):
    # Calculate the time between the first bar off and the next bar on
    
    bar_off =  df_trial[df_trial['content']=='bar_off']
    bar_on =  df_trial[df_trial['content']=='bar_on']

    
    if len(bar_off) > 0 and len(bar_on) > 0:
        bar_off1 = bar_off.iloc[0]

        bar_on = bar_on[bar_on['trial_time'] > bar_off1['trial_time']]
        
        if len(bar_on)>0:
            return bar_on.iloc[0]['trial_time'] - bar_off1['trial_time']
        
def get_first_sig_bar_off_time(df_trial, min_off_time = 100):
    # Calculate the trial time of the first bar off, time between bar off and the next on 
    # must be larger than min_off_time
    
    df_trial = df_trial[df_trial.trial_time>=0]

    bar_off =  df_trial[(df_trial['content']=='bar_off')]
    bar_on =  df_trial[(df_trial['content']=='bar')]
    # print(bar_on)
    
    if len(bar_off) > 0 and len(bar_on) > 0:
        bar_off1 = bar_off.iloc[0]

        bar_on = bar_on[bar_on['trial_time'] > bar_off1['trial_time']]
        
        #make sure bar on and off are of the same size
        if len(bar_on) > len(bar_off):
            bar_on = bar_on.iloc[:len(bar_off)]
        else:
            bar_off = bar_off.iloc[:len(bar_on)]
        
        interval = bar_on.trial_time.values - bar_off.trial_time.values
        
        # filter out short bar off
        idx = np.where(interval>min_off_time)[0]
        if len(idx)>0:
            return bar_off.iloc[idx[0]].trial_time

def get_first_bar_off_time(df_trial):
    # Calculate the trial time of the first bar off, time between bar off and the next on 
    # must be larger than min_off_time
    df_trial = df_trial[df_trial.trial_time>=0]
    bar_off =  df_trial[df_trial['content']=='bar_off']
    if (len(bar_off))>0:
        return bar_off.iloc[0].trial_time
