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

def get_us_timer_delay(df_trial):
    # df_trial is a dataframe of a single trial
    # it needs to return a row of the original dataframe
    us_timer_delay = df_trial[df_trial['name'] == 'US_end_timer']
    
    if len(us_timer_delay) > 0:
        return us_timer_delay.iloc[0]
    
def extract_event_time(df_event, filter_func, filter_func_kwargs, groupby_col='trial_nb'):
    #extract event on a trial based on a filter function
    df_event_filtered = df_event.groupby(groupby_col,group_keys=True).apply(filter_func, **filter_func_kwargs)
    if len(df_event_filtered)>0:
        return df_event_filtered.time
    else:
        #No event found, but still need to return the trial nb info
        return df_event.groupby(groupby_col,group_keys=True)['time'].apply(lambda x: None)

def extract_trial_time(df_event, filter_func, filter_func_kwargs, groupby_col='trial_nb'):
    #extract event on a trial based on a filter function
    df_event_filtered = df_event.groupby(groupby_col,group_keys=True).apply(filter_func, **filter_func_kwargs)
    return df_event_filtered.trial_time
