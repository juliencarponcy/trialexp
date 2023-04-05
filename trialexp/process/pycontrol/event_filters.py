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
    
def get_first_event_from_name(df_trial, evt_name):
    event =  df_trial[df_trial['name']==evt_name]
    return event.iloc[0]
    
def extract_event_time(df_event, filter_func, filter_func_kwargs, groupby_col='trial_nb'):
    #extract event on a trial based on a filter function
    df_event_filtered = df_event.groupby(groupby_col,group_keys=True).apply(filter_func, **filter_func_kwargs)
    if len(df_event_filtered)>0:
        return df_event_filtered.time
    else:
        print(f'No event for {filter_func.__name__} found')
        return []
