import xarray as xr

def get_first_bar_off(df_trial):
    #Find first bar off in trial
    # This function gets a dataframe of a trial, and need to return
    # the row for event of interest
    bar_off =  df_trial[df_trial['name']=='bar_off']
    
    if len(bar_off) >0:
        return bar_off.iloc[0]
    
    
def get_first_spout(df_trial):
    #Find spout touch
    spout =  df_trial[df_trial['name']=='spout']
    
    if len(spout) >0:
        return spout.iloc[0]

def get_last_bar_off_before_first_spout(df_trial):
    # Find the last bar_off before the first spout in trial
    # You can prepare df_trial for debugging/development as below:
    #
    # for i in range(1, np.max(df_event['trial_nb'])):
	#     df_trial = df_event[df_event['trial_nb'] == i]
    # df_trial = the row for event of interest

    bar_off =  df_trial[df_trial['name']=='bar_off']

    spout =  df_trial[df_trial['name']=='spout']
    if len(spout) > 0 and len(bar_off) > 0:
        spout1 = spout.iloc[0]

        filtered_df = bar_off[bar_off['trial_time'] < spout1['trial_time']]
        max_time = filtered_df['trial_time'].max()
        result = filtered_df[filtered_df['trial_time'] == max_time]

        if len(result) >0:
            return result.iloc[0]


def get_first_event_from_name(df_trial, evt_name):
    event =  df_trial[df_trial['name']==evt_name]
    return event.iloc[0]
    
def extract_event_time(df_event, filter_func, filter_func_kwargs, groupby_col='trial_nb'):
    #extract event on a trial based on a filter function
    df_event_filtered = df_event.groupby(groupby_col,group_keys=True).apply(filter_func, **filter_func_kwargs)
    if len(df_event_filtered)>0:
        return df_event_filtered.time
    else:
        #No event found, but still need to return the trial nb info
        return df_event.groupby(groupby_col,group_keys=True)['time'].apply(lambda x: None)


# %%
def extract_clean_trigger_event(df_trial, target_event_name, clean_window, ignore_events=None):
    # This function will extract an event with nothing happening (except those in ignore_events) before and after the clean window
    
    # extract all event within the clean_window
    target_events = df_trial[df_trial['name'] == target_event_name]
    
    if len(target_events)>0:
        target_time = target_events.iloc[0].time
    
        idx = (df_trial.time > (target_time+clean_window[0])) & (df_trial.time <(target_time+clean_window[1]))
        
        # Disregard the ignored events
        if ignore_events is not None:
            idx = idx & ~df_trial['name'].isin(ignore_events)

        if sum(idx) ==1 and df_trial.loc[idx].iloc[0]['name'] == target_event_name:
            return df_trial.loc[idx].iloc[0] # must return a series