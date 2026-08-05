
import numpy as np
import pandas as pd

def compute_trial_nb_by_day(row, max_trial_nb_by_day):
    '''
    Helper function to use with apply() on a metadata_df DataFrame.
    It re-computes the trial_nb based on total trial in a day in case
    of multiple sessions in one day.
    '''
    # list sessions numbers for the day for this subject

    # inefficient to put here, slow down computation
    sessions_nb = max_trial_nb_by_day.loc[
        (row.subject_ID, row.date, max_trial_nb_by_day.index.get_level_values(2)),:].index.get_level_values(2).values
    
    sessions_nb = list(sessions_nb)

    if row.session_nb == sessions_nb[0]:
        # print('row == session_nb')
        return row.trial_nb
    else:

        # return the number of trials before this session on the day
        prev_trial_nb = max_trial_nb_by_day.loc[(row.subject_ID, row.date, sessions_nb[:sessions_nb.index(row.session_nb)]),'trial_nb'].cumsum().values[-1]
        # print(prev_trial_nb)
        return row.trial_nb + prev_trial_nb


def trial_nb_normalization(metadata_df, by_day: bool = False):
    '''
    Compute trial_nb normalized position by session, or by day (if by_day = True) 

    Parameters:
    -----------
        metadata_df: pd.DataFrame
            The DataFrame containing metadata for each trial of the trial_dataset
        by_day: bool
            If True, aggregates trials from sessions performed on the same day to
            compute their normalized position
    Return:
    -------
        metadata_df: pd.DataFrame

    '''

    if by_day:
        metadata_df.loc[:,'date'] = metadata_df['datetime'].apply(lambda d: d.date())
        max_trial_nb_by_day = metadata_df.groupby(['subject_ID', 'date', 'session_nb']).agg({'trial_nb':'max'})

        metadata_df.loc[:,'trial_nb_day'] = metadata_df.apply(lambda x: compute_trial_nb_by_day(x, max_trial_nb_by_day), axis=1)
       
        max_trial_nb_by_day = max_trial_nb_by_day.groupby(['subject_ID', 'date']).agg({'trial_nb':'sum'})

        metadata_df.loc[:,'trial_nb_norm'] = metadata_df.apply(lambda x: x.trial_nb_day / max_trial_nb_by_day.loc[(x['subject_ID'], x['date'])], axis=1)

    else:
        max_trial_nb = metadata_df.groupby(['subject_ID','session_nb']).agg({'trial_nb':'max'})
        metadata_df.loc[:,'trial_nb_norm'] = metadata_df.apply(lambda x: x.trial_nb / max_trial_nb.loc[(x['subject_ID'], x['session_nb'])], axis=1)
    
    return metadata_df

def session_nb_normalization(metadata_df, by_day: bool = False):
    '''
    Compute session_nb normalized position by session, or by day (if by_day = True) 

    Parameters:
    -----------
        metadata_df: pd.DataFrame
            The DataFrame containing metadata for each trial of the trial_dataset
        by_day: bool
            If True, aggregates sessions performed on the same day to
            compute their normalized position
    Return:
    -------
        metadata_df: pd.DataFrame

    '''
    metadata_df.loc[:,'date'] = metadata_df['datetime'].apply(lambda d: d.date())

    if by_day:
        dates_norm_dict = dict()
        for subject_ID in metadata_df.subject_ID.unique():

            dates = metadata_df.loc[metadata_df.subject_ID == subject_ID,'date'].unique()
            dates_norm = np.linspace(0,1,len(dates))
            dates_norm_dict[subject_ID] = dict(zip(dates,dates_norm))

        metadata_df.loc[:,'session_nb_norm'] = metadata_df.apply(lambda x: dates_norm_dict[x.subject_ID][x.date], axis=1)

    else:
        dates_norm_dict = dict()
        for subject_ID in metadata_df.subject_ID.unique():

            sessions = metadata_df.loc[metadata_df.subject_ID == subject_ID, 'session_nb'].unique()
            sessions_norm = np.linspace(0,1,len(sessions))
            dates_norm_dict[subject_ID] = dict(zip(sessions,sessions_norm))

        metadata_df.loc[:,'session_nb_norm'] = metadata_df.apply(lambda x: dates_norm_dict[x.subject_ID][x.session_nb], axis=1)
    
    return metadata_df

# def trial_nb_normalization_by_day(metadata_df):

#     max_trial_nb = metadata_df.groupby(['subject_ID','session_nb']).agg({'trial_nb':'max'})
#     metadata_df['trial_nb_norm'] = metadata_df.apply(lambda x: x.trial_nb / max_trial_nb.loc[(x['subject_ID'], x['session_nb'])], axis=1)
#     return metadata_df

def trial_nb_quantilization(metadata_df: pd.DataFrame, quantiles: tuple):
    '''
    Assign quantile index to each trial based on the trial_nb_norm value
    computed by the trial_nb_normalization(metadata_df) method.

    Parameters:
    -----------
        metadata_df: pd.DataFrame
            The DataFrame containing metadata for each trial of the dataset
        quantiles: tuple
            A tuple of 2 entries tuples indicating the limits of the desired quantiles.
            The lower limit is not included while the upper limit is.
  
        example:

        >> metadata_df = trial_nb_normalization(metadata_df)
        # Break down the trial numbers into 3 thirds
        >> quantiles = ((0,0.33),(0.33,0.66),(0.66,1))
        >> metadata_df = trial_nb_quantilization(metadata_df, quantiles = quantiles)

    
    Return the same DataFrame with the correspondign quantile indices
    '''

    if 'trial_nb_norm' not in metadata_df.columns:
        raise Exception('You need to compute first trial_nb_normalization(metadata_df)')

    metadata_df['trial_nb_quantile'] = np.nan
    for q_nb, quantile_lim in enumerate(quantiles):
        metadata_df.loc[((metadata_df['trial_nb_norm'] > quantile_lim[0]) & (metadata_df['trial_nb_norm'] <= quantile_lim[1])) , 'trial_nb_quantile'] = q_nb

    metadata_df['trial_nb_quantile'] = metadata_df['trial_nb_quantile'].astype(int)

    return metadata_df

def session_nb_quantilization(metadata_df: pd.DataFrame, quantiles: tuple):
    '''
    Assign quantile index to each trial based on the trial_nb_norm value
    computed by the trial_nb_normalization(metadata_df) method.

    Parameters:
    -----------
        metadata_df: pd.DataFrame
            The DataFrame containing metadata for each trial of the dataset
        quantiles: tuple
            A tuple of 2 entries tuples indicating the limits of the desired quantiles.
            The lower limit is not included while the upper limit is.
  
        example:

        >> metadata_df = trial_nb_normalization(metadata_df)
        # Break down the trial numbers into 3 thirds
        >> quantiles = ((0,0.33),(0.33,0.66),(0.66,1))
        >> metadata_df = trial_nb_quantilization(metadata_df, quantiles = quantiles)

    
    Return the same DataFrame with the correspondign quantile indices
    '''

    if 'session_nb_norm' not in metadata_df.columns:
        raise Exception('You need to compute first session_nb_normalization(metadata_df)')

    metadata_df['session_nb_quantile'] = np.nan
    for q_nb, quantile_lim in enumerate(quantiles):
        metadata_df.loc[((metadata_df['session_nb_norm'] >= quantile_lim[0]) & (metadata_df['session_nb_norm'] <= quantile_lim[1])) , 'session_nb_quantile'] = q_nb

    metadata_df['session_nb_quantile'] = metadata_df['session_nb_quantile'].astype(int)

    return metadata_df