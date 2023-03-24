from trialexp.process.pyphotometry.utils import get_rel_time
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
    
def extract_event_time(df_event, filter_func, groupby_col='trial_nb'):
    #extract event on a trial based on a filter function
    
    df_event_filtered = df_event.groupby(groupby_col).apply(filter_func).dropna(how='all')
    return df_event_filtered.time
