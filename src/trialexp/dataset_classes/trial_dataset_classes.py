# Python classes for storing, plotting and transform data by trials
import sys
from datetime import datetime, date
from typing import Iterable, Union, Optional, Tuple
import os
# from dataclasses import dataclass
# from abc import ABC, abstractmethod

import pickle
from re import search
import warnings

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.axes import Axes
import matplotlib

import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
from plotly.subplots import make_subplots

from trialexp.utils.cont_dataset_utlities import *

# To convert for ML with sktime.org
# make optional for now
try:
    from sktime.datatypes._panel import _convert as convert
except:
    print(
        '''Warning : sktime package not installed, this is `only an issue
        if you want to export datasets to perform Maching Learning tasks.
        To solve, type pip install sktime in your environment'''
    )

# Find use or remove
ConditionsType = Union[dict, Iterable[dict]]
VarsType = Union[str, int, Iterable[int], Iterable[str]]
LimitType = Tuple[int, int] 


# Custom exception
class NotPermittedError(Exception):
    def __init__(self, cls_method, attributes, values):
        self.message = f"Can't use {cls_method} method with {attributes} = {values}"
        super().__init__(self.message)

def load_dataset(fullpath:str):
    with open(fullpath, 'rb') as file:
        dataset = pickle.load(file)
        return dataset


class Trials_Dataset():
    """
    Trials_Dataset has subclasses Continuous_Dataset and Event_Dataset.
    Unlike trialexp.process.data_import.Session, which holds data of an entire session
    with a linear time vector, Trials_Dataset holds triggered trial data that extends to trial_window.
    In other words, each trial is provided the equal time vector whose 0 is at the trigger event.
    Attributes
    ----------
    data : 
        pandas.DataFrame (Event_Dataset) 
        or numpy.ndarray (Continuous_Dataset)
    metadata_df : DataFrame
        Rows for trials, holding colums:
            trial_nb : int
            trigger : str
            success : bool
            valid : bool
            condition_ID : int 
            condition : str
            group_ID : int
            session_nb : int
            subject_ID : int or str
            keep : bool
            trial_ID : int
    creation_date : datetime.datetime
    has_conditions : bool
    has_groups : bool
    groups : array
    conditions : list of dict
    cond_aliases : list of str
    trial_window : list
        eg [-2000, 6000]
        The window size relative to trigger for trial-based data fragmentation.  
    time_unit : str
        'ms' | 'milliseconds' | 's' | 'seconds'
    Methods
    -------
    export()
    filter_min()
        These filter_* and filterout_* methods are to modify the values of 'keep' column of metadata_df
    filter_reset()
        The values of 'keep' column of metadata_df are all set to True
    filterout_conditions()
    filterout_dates()
    filterout_groups()
    filterout_if_not_in_all_cond()
    filterout_subjects()
    get_groups()
    get_memory_size()
    get_session_files()
    save()
    set_conditions()
    set_groups()
    set_trial_window()
    """

    def __init__(self, data, metadata_df: pd.DataFrame):
        self.data = data
        self.metadata_df = metadata_df
        self.metadata_df['keep'] = True
        self.metadata_df['trial_ID'] = self.metadata_df.index.values
        self.creation_date = datetime.now()
        self._trial_window = []
        self._time_unit = ''
        
        if self.data.shape[0] != self.metadata_df.shape[0]:
            raise ValueError(' \
                Data and metadata does not have the same nb of trials')

        if 'condition_ID' in self.metadata_df.columns:
            if metadata_df['condition_ID'].nunique() > 1:
                self.has_conditions = True
        else:
            self.has_conditions = False
        
        if 'group_ID' in self.metadata_df.columns:
            self.has_groups = True
            _ = self.get_groups()
        else:
            self.has_groups = False

    @property
    def trial_window(self):
        return self._trial_window

    @property
    def time_unit(self):
        return self._time_unit

    @property
    def triggers(self):
        trg = self.metadata_df['trigger']
        trg = list(set(trg))
        return trg

    def set_conditions(self, conditions: ConditionsType, aliases: Iterable = None):
        if isinstance(conditions, list) and not all([isinstance(conditions[c],dict) for c in range(len(conditions))]):
            raise TypeError(f'conditions must be of any of the follwing types \
                {ConditionsType.__args__[:]}')
        
        elif isinstance(conditions, dict):
            self.conditions = (conditions)
        else:
            self.conditions = conditions

        if isinstance(aliases, str) and len(self.conditions) == 1:
            aliases = (aliases)
            self.cond_aliases = aliases
        elif isinstance(aliases, Iterable) and len(aliases) == len(self.conditions):
            self.cond_aliases = aliases
        elif not aliases:
            ...
        else:
            raise ValueError(
                'aliases must be of the same lenght than conditions')

    def get_groups(self):
        if not self.has_groups:
            raise NotPermittedError(self.get_groups.__name__, 'has_groups', self.has_groups)

        self.groups = self.metadata_df['subject_ID'].groupby(
            self.metadata_df['group_ID'].values).unique().values

        return self.groups
    
    # In progress
    def set_groups(self, groups: list = None):
        if not groups:
            self.metadata_df.loc['group_ID'] = 0
        else:
            for g_idx, group in enumerate(groups):

                self.metadata_df['group_ID'] = self.metadata_df.apply(
                    lambda x: [g_idx for g_idx, group in enumerate(groups) if x['subject_ID'] in group])
            # for g_idx, group in enumerate(groups):

                # for row in self.metadata_df.itertuples():
            #         print(row)
            #         if row.subject_ID in groups[g_idx]:
            #             row.group_ID = g_idx


    def set_trial_window(self, trial_window: Iterable, unit: str = None):
        self._trial_window = trial_window
        self._time_unit = unit

    def get_memory_size(self, verbose=True):
        mem_mb = sys.getsizeof(self.data)/1024/1024
        str_mem = f'data size is {mem_mb} Mb'
        if verbose:
            print(str_mem)
    
        return mem_mb

    def get_session_files(self, folder):
        # see whether it needs to import data_import 
        raise NotImplementedError
        
    def save(self, folder: str = None, name: str = None, verbose: bool = True):
        """
        Save Trials_Dataset to pickle file (.pkl)
        """
        if folder:
            fullpath = os.path.join(folder, name + '.pkl')
        else:
            fullpath = os.path.join(os.getcwd(), name + '.pkl')
        
        with open(fullpath, 'wb') as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

        if verbose:
            print(f'Dataset saved in {fullpath}')
            _ = self.get_memory_size()

    def export(self, format:str):
        '''
        Perhaps should be subclass specific,
        already implemented for sktime format
        '''
        raise NotImplementedError    

    def filterout_conditions(self, condition_IDs_to_exclude: list):
        """
        exclude one or several conditions of the dataset using integer condition_IDs_to_exclude
        the index (ID) starting from 0
        """
        if isinstance(condition_IDs_to_exclude, int):
            condition_IDs_to_exclude = [condition_IDs_to_exclude]
        filter = self.metadata_df['condition_ID'].apply(lambda x: x in condition_IDs_to_exclude)
        self.metadata_df.loc[filter,'keep'] = False

    def filterout_dates(self, days_to_exclude: list):
        """
        exclude one or several dates of the dataset 
        """
        if not isinstance(days_to_exclude, list):
            days_to_exclude = [days_to_exclude]

        if all([isinstance(d, date) for d in days_to_exclude]):
            days_to_exclude = days_to_exclude
        elif all([isinstance(d, datetime) for d in days_to_exclude]):
            days_to_exclude = [d.date() for d in days_to_exclude]
        else:
            raise TypeError("days_to_exclude has to be a list of datetime or date")
        filter = self.metadata_df['datetime'].apply(
            lambda x: x.date() in days_to_exclude)
        self.metadata_df.loc[filter,'keep'] = False

    def filterout_groups(self, group_IDs_to_exclude: list):
        """
        exclude one or several groups of the dataset
        """
        if isinstance(group_IDs_to_exclude, int):
            group_IDs_to_exclude = [group_IDs_to_exclude]
        filter = self.metadata_df['group_ID'].apply(lambda x: x in group_IDs_to_exclude)
        self.metadata_df.loc[filter,'keep'] = False

    def filterout_subjects(self, subject_IDs_to_exclude: list):
        """
        exclude one or several subjects of the dataset
        """
        if isinstance(subject_IDs_to_exclude, int):
            subject_IDs_to_exclude = [subject_IDs_to_exclude]
        filter = self.metadata_df['subject_ID'].apply(lambda x: x in subject_IDs_to_exclude)
        self.metadata_df.loc[filter,'keep'] = False
    
    def filter_min_by_condition(self, min_trials : int = 10):
        """
        filter subjects who do not have sufficient trials in a condition,
        NB: this is not removing the trials of this subject in the other
        conditions! 
        If you want to exclude animals which have less than x trials in
        a condition from the full dataset, use sequentially:
        <trial_dataset>.filter_min(min_trials = x)
        <trial_dataset>.filterout_if_not_in_all_cond()
        """

        nb_trials = self.metadata_df.groupby(['condition_ID', 'group_ID','subject_ID']).agg(len)['trial_ID']
        trials_idx = self.metadata_df.groupby(['condition_ID', 'group_ID','subject_ID']).agg(list)['trial_ID']
        discarded = nb_trials[nb_trials < min_trials].index
        if len(discarded) > 0:
            trials_idx[discarded]
            idx_filter = np.concatenate(trials_idx[discarded].values)
            self.metadata_df.loc[idx_filter,'keep'] = False

    def filter_min_by_session(self, min_trials : int = 5):
        """
        filter subjects who do not have sufficient trials in a condition,
        NB: this is not removing the trials of this subject in the other
        conditions! 
        If you want to exclude animals which have less than x trials in
        a condition from the full dataset, use sequentially:
        <trial_dataset>.filter_min(min_trials = x)
        <trial_dataset>.filterout_if_not_in_all_cond()
        """

        nb_trials = self.metadata_df.groupby(['condition_ID', 'group_ID','subject_ID','session_nb']).agg(len)['trial_ID']
        trials_idx = self.metadata_df.groupby(['condition_ID', 'group_ID','subject_ID','session_nb']).agg(list)['trial_ID']
        discarded = nb_trials[nb_trials < min_trials].index
        if len(discarded) > 0:
            trials_idx[discarded]
            idx_filter = np.concatenate(trials_idx[discarded].values)
            self.metadata_df.loc[idx_filter,'keep'] = False

    def filter_lastNsessions(self, n : int):
        """
        Only keep the last n sessions for each animal.
        The sessions are counted for the sessions with 'keep' == True
        """
        metadata_df = self.metadata_df[self.metadata_df['keep'] == True].copy()
        subject_IDs = metadata_df['subject_ID'].unique()

        idx_reject = list()
        for subject_ID in subject_IDs:
            session_nbs = metadata_df[metadata_df['subject_ID'] == subject_ID].session_nb.unique()
            session_nbs = np.flip(np.sort(session_nbs))

            if len(session_nbs) > n:
                invalid_sessions_nb = session_nbs[n:]

                for inv_s in invalid_sessions_nb:    
                    idx_reject = idx_reject + list(metadata_df[(metadata_df['subject_ID'] == subject_ID) & (metadata_df['session_nb'] == inv_s)].index.values)

        self.metadata_df.loc[(idx_reject), 'keep'] = False

    def filter_lastNdays(self, n : int):
        """
        Only keep the last n sessions for each animal.
        The sessions are counted for the sessions with 'keep' == True
        """
        metadata_df = self.metadata_df[self.metadata_df['keep'] == True].copy()
        subject_IDs = metadata_df['subject_ID'].unique()

        idx_reject = list()
        for subject_ID in subject_IDs:
            dates = metadata_df[metadata_df['subject_ID'] == subject_ID].datetime.apply(lambda d: d.date()).unique()
            dates = np.flip(np.sort(dates))

            if len(dates) > n:
                invalid_dates = dates[n:]

                for inv_d in invalid_dates:    
                    idx_reject = idx_reject + list(metadata_df[(metadata_df['subject_ID'] == subject_ID) & (metadata_df.datetime.apply(lambda d: d.date()) == inv_d)].index.values)

        self.metadata_df.loc[(idx_reject), 'keep'] = False

    def filter_firstNsessions(self, n : int):
        """
        Only keep the first n sessions for each animal.
        """
        metadata_df = self.metadata_df[self.metadata_df['keep'] == True].copy()
        subject_IDs = metadata_df['subject_ID'].unique()

        idx_reject = list()
        for subject_ID in subject_IDs:
            session_nbs = metadata_df[metadata_df['subject_ID'] == subject_ID].session_nb.unique()
            session_nbs = np.sort(session_nbs)

            if len(session_nbs) > n:
                invalid_sessions_nb = session_nbs[n:]

                for inv_s in invalid_sessions_nb:    
                    idx_reject = idx_reject + list(metadata_df[(metadata_df['subject_ID'] == subject_ID) & (metadata_df['session_nb'] == inv_s)].index.values)

        self.metadata_df.loc[(idx_reject), 'keep'] = False

    def filterout_if_not_in_all_cond(self):
        """
        To remove subjects who do not have
        trials in all the conditions.
        Can be used after <trial_dataset>.filter_min()
        """
        

        filtered_df = self.metadata_df[self.metadata_df['keep'] == True]
        pre_gby = filtered_df.groupby(['condition_ID','subject_ID'])
        pre_gby = pre_gby.aggregate({'trial_ID' : np.count_nonzero})
        
        nb_cond = pre_gby.index.get_level_values('condition_ID').nunique()

        to_remove = pre_gby.reset_index(level=0).index.value_counts() < nb_cond

        to_remove = to_remove.index[to_remove.values]

        idx_to_remove = filtered_df['subject_ID'].isin(to_remove) == True
        idx_to_remove = idx_to_remove[idx_to_remove == True]

        self.metadata_df.loc[idx_to_remove.index,'keep'] = False

    def filter_flip(self):

        tf = pd.Series([True] * self.metadata_df.shape[0])
        
        tf.loc[self.metadata_df['keep']] = False

        self.metadata_df['keep'] = tf

    def filter_reset(self):
        """
        reset filters to include all trials as
        at the creation of the dataset
        The values of 'keep' column of metadata_df are all set to True
        """

        self.metadata_df['keep'] = True

    def set_keep(self, tfkeep: pd.Series):
        """
        Set the value of 'keep' column (used as a filter) of metadata_df
        self.metadata_df['keep'] = tfkeep
        See also:
        get_tfkeep_subjects()
        get_tfkeep_dates()
        get_tfkeep_conditions()
        get_tfkeep_lastNsessions()
        get_tfkeep_groups()
        """
        self.metadata_df['keep'] = tfkeep

    def get_tfkeep_subjects(self, subject_IDs_to_include: list) -> pd.Series:
        """
        return a bool vector (pd.Series) that have True for one or several subjects of the dataset.
        The vector tfkeep can be used for boolean operations 
        eg. tfkeep3 = (tfkeep1) & (tfkeep2) 
        and then used to set the value of obj.set_keep(tfkeep3)
        """
        if isinstance(subject_IDs_to_include, int):
            subject_IDs_to_include = [subject_IDs_to_include]
        tfkeep = self.metadata_df['subject_ID'].apply(lambda x: x in subject_IDs_to_include)
        return tfkeep

    def get_tfkeep_dates(self, days_to_include: list) -> pd.Series:
        """
        return a bool vector (pd.Series) that have True for one or several dates of the dataset 
        days_to_include
            list of datetime.datetime or list of datetime.date
        The vector tfkeep can be used for boolean operations 
        eg. tfkeep3 = (tfkeep1) & (tfkeep2) 
        and then used to set the value of obj.set_keep(tfkeep3)
        https://docs.python.org/3/library/datetime.html#datetime.datetime
        https://docs.python.org/3/library/datetime.html#datetime.date
        """
        if not isinstance(days_to_include, list):
            days_to_include = [days_to_include]

        if all([isinstance(d, datetime) for d in days_to_include]):
            days_to_include = [d.date() for d in days_to_include]
        elif all([isinstance(d, date) for d in days_to_include]):
            days_to_include = days_to_include
        else:
            raise TypeError(
                "days_to_include has to be a list of datetime.datetime or datetime..date")
        tfkeep = self.metadata_df['datetime'].apply(
            lambda x: x.date() in days_to_include)
        return tfkeep

    def get_tfkeep_conditions(self, condition_IDs: list) -> pd.Series:
        """
        return a bool vector (pd.Series) that have True for one or several conditions of the dataset using integer condition_IDs
        the index (ID) starting from 0
        The vector tfkeep can be used for boolean operations 
        eg. tfkeep3 = (tfkeep1) & (tfkeep2) 
        and then used to set the value of obj.set_keep(tfkeep3)
        """
        if isinstance(condition_IDs, int):
            condition_IDs = [condition_IDs]
        tfkeep = self.metadata_df['condition_ID'].apply(
            lambda x: x in condition_IDs)
        return tfkeep

    def get_tfkeep_lastNsessions(self, n : int) -> pd.Series:
        """
        return a bool vector (pd.Series) that have True for the last n sessions for each animal.
        The five sessions are counted for the sessions with 'keep' == True
        The vector tfkeep can be used for boolean operations 
        eg.tfkeep3 = (tfkeep1) & (tfkeep2) 
        and then used to set the value of obj.set_keep(tfkeep3)
        """
        subject_IDs = list(set(self.metadata_df['subject_ID']))
        subject_IDs.sort()

        tfkeep = pd.Series([False] * self.metadata_df.shape[0])
        for s in subject_IDs:
            # NOTE copy() is needed
            session_nbs = self.metadata_df.loc[:, 'session_nb'].copy()

            session_nbs.loc[(
                (self.metadata_df['subject_ID'] != s)
                | (self.metadata_df['keep'] != True)
            )] = -1

            smallestNs = list(set(session_nbs))
            smallestNs.sort(reverse=True)

            smallestNs = smallestNs[0:n]

            if -1 in smallestNs:
                smallestNs.remove(-1)

            for k in smallestNs:
                tfkeep.loc[session_nbs == k] = True

        return tfkeep

    def get_tfkeep_groups(self, group_IDs_to_exclude: list) -> pd.Series:
        """
        return a bool vector (pd.Series) that have True for one or several groups of the dataset
        The vector tfkeep can be used for boolean operations 
        eg. tfkeep3 = (tfkeep1) & (tfkeep2) 
        and then used to set the value of obj.set_keep(tfkeep3)
        """
        if isinstance(group_IDs_to_exclude, int):
            group_IDs_to_exclude = [group_IDs_to_exclude]
        tfkeep = self.metadata_df['group_ID'].apply(lambda x: x in group_IDs_to_exclude)
        return tfkeep  


class Continuous_Dataset(Trials_Dataset):
    """
    Subclass of Trials_Dataset.
    Attributes
    ----------
    colnames_dict : dict
    Methods
    -------
    scatterplot()
        to be implemented
    export_to_sktime()
    transform_variables()
        to be implemented
    get_time_vector(self, unit: str = None)
    set_fs(self, fs: int):
    lineplot(...)
        The main plotting method averaging for sessions/subjects/groups
    heatmap(...)
        Plotting method to see the detail of every trial/session
    get_col_names()
    """
    def __init__(self, data: np.ndarray, metadata_df: pd.DataFrame, colnames_dict: dict):
        super().__init__(data, metadata_df)
        """
        Arguments
        ---------
        self
        data : numpy.ndarray
            Has to be 3d?
            What are dimensions? #TODO
        metadata_df : pandas.DataFrame
        colnames_dict : dict
        """
        # TODO: Consider inputing colnames only as list or tuple
        # and compute the dictionary '<names>': <idx_col(int)> in __init__
        self.colnames_dict = colnames_dict
        

    # is the following necessary?
    # will be if attributes made private
    def get_col_names(self) -> list:
        return [key for key in self.colnames_dict.keys()]

    # useless given length of trials and trial_window
    def set_fs(self, fs: int):
        self.sampling_rate = fs

    def set_trial_window(self, trial_window: Iterable, unit: str = None):
        self._trial_window = trial_window
        self._time_unit = unit
        if hasattr(self, 'sampling_rate'):
            self.time_vector = self.get_time_vector(unit)

    def get_time_vector(self, unit: str = None):
        
        if not hasattr(self, 'trial_window'):
            raise AttributeError(
                "You must define trial window \n\r \
                by using set_trial_window()")     
        
        if hasattr(self, 'trial_window'):
            
            time_vector = np.linspace(self.trial_window[0],
                self.trial_window[1], self.data.shape[2])

        if hasattr(self, 'time_unit') and unit is not None \
            and self.time_unit in ['ms', 'milliseconds'] \
            and unit in ['s', 'seconds']:

            time_vector = time_vector / 1000

        elif hasattr(self, 'time_unit') and unit is not None \
            and self.time_unit in ['s', 'seconds'] \
            and unit in ['ms', 'milliseconds']:

            time_vector = time_vector * 1000
        
        elif hasattr(self, 'time_unit') and unit is not None \
            and self.time_unit in ['s', 'seconds'] \
            and unit in ['s', 'seconds']:

            pass
                
        elif hasattr(self, 'time_unit') and unit is not None \
            and self.time_unit in ['ms', 'milliseconds'] \
            and unit in ['ms', 'milliseconds']:

            pass

        elif self.time_unit == None and unit == None:
            
            pass
            
        else:
            
            raise AttributeError(
                'if you want a specific time unit you must first\n\r \
                specify it (re)using set_trial_window or set_time_vector')

        return time_vector

    def cluster_trials(
            self,
            vars_to_cluster_on: str = 'analog_2',  # only work with one var for now.
            dim_reduc_type: str = 'ICA',
            eps: float =  0.005,
            min_samples_by_cluster: int = 20, 
            plot: bool = False,
            plot_lim_pctile: float = 0.1
            ):
        
        """
        Cluster trials based on computation over a single channel (for now).
        The features extracted ar in the base_feature_extraction() method,
        in the cont_dataset_utilities.py file. (in which other features could
        be implemented)
        Once this feature extracted, the method perform dimensionality reduction
        by PCA or ICA, then run the DBSCAN algorithm to group the trials in
        differnt clusters, to identify noise (cluster: -1) and/or other
        abnormal trials.
        Beside optional plotting, the results are stored in the 
        Arguments
        ---------
        vars_to_cluster_on: str = 'analog_2',  # only work with one var for now.
            variable from which to extract feature and perform clustering on
        dim_reduc_type: str = 'ICA',
            can be either 'PCA' or 'ICA'
        eps: float =  0.005, 
            The most crucial parameter, as it will determine the level
            of acceptable noise/artifacts in your remaining trials
        min_samples_by_cluster: int = 20,
            The minimum number of samples that could constitute a cluster for
            the DBSCAN algorithm
        plot: bool = False,
            Whether to plot the dimensionality reduction and the clustering
        plot_lim_pctile: float = 0.1
            use to scale the plot as artifacted, very far way trials, if all
            included, may entrain crowding of useful samples in a tiny area 
            of the plot. 
            plot_lim_pctile will scale in x and y to display all but 
            plot_lim_pctile portion of your data, e.g. if 0.1, the scaling
            of the plot will include 100 - 0.1% (99.8% of the trials) in x/y
        Returns
        -------
        None
            just store the nb of the cluster in the <cont_dataset>.metadata_df
            in the cluster column. 
        """
        # TODO, transform feature extraction to not have to transform the data
        # make available to cluster on more than one variable
        # 
        df_panel, _ = self.export_to_sktime()

        df_components = extract_feature_components(
            df_panel, 
            variable = vars_to_cluster_on, 
            type = dim_reduc_type, 
            plot = False)

        labels  = DBSCAN_cluster_on_compoonents(
            np_comp = df_components, 
            eps = eps,
            min_samples_by_cluster = min_samples_by_cluster, 
            plot = plot,
            plot_lim_pctile = plot_lim_pctile)


        self.metadata_df['cluster'] = labels


    #TODO adapt to be able to work with only one variable, unlikely for now (axes issue)
    def plot_clustered_trials(self,
            vars_to_plot: list = ['analog_2','analog_1_df_over_f'],
            min_cluster_size_to_plot: int = 20,
            ylims: list = [[0.2,0.8],[-0.1,0.4]],
            figsize: tuple = (15,10)
            ):
        """
        Plot all trials (and average) by cluster as defined by the cluster_trials()
        method.
        Arguments:
        ----------
            vars_to_plot: list = ['analog_2','analog_1_df_over_f'],
                the variables to plot
            min_cluster_size_to_plot: int = 20,
                do not create a new plot for clusters smaller than this nb
            ylims: list = [[0.2,0.8],[-0.1,0.4]],
                allow you to specify the ylim property of axes, otherwise the
                plots will be scaled to the most artifacted trials
            figsize: tuple = (15,10)
                figure size, so far to manually adapt
            ):
        """
        if 'cluster' in self.metadata_df.columns:
            labels = self.metadata_df['cluster'].values
        else:
            raise Exception('Clustering not performed on this dataset, use <cont_dataset>.cluster_trials() method')

        pd_labels = pd.Series(labels)
        clusters_size = pd_labels.value_counts().to_dict()
        clusters_size = {k: v for (k, v) in clusters_size.items() if (v >= min_cluster_size_to_plot)}


        fig, axs = plt.subplots(nrows = len(vars_to_plot), ncols = clusters_size.__len__(), sharey = 'row', figsize = figsize)
        timevec_trial = np.linspace(self.trial_window[0], self.trial_window[1], self.data.shape[2])

        for row, var in enumerate(vars_to_plot):

            for col, clust_nb in enumerate(clusters_size.keys()):

                _ = axs[row,col].plot(
                    
                    timevec_trial, self.data[labels == clust_nb, self.colnames_dict[var],:].T, alpha=0.3)
                _ = axs[row,col].plot(timevec_trial, self.data[labels == clust_nb, self.colnames_dict[var],:].mean(0), c='k', alpha=1)


                axs[row,col].set_title(f'{var} cluster: {clust_nb}, trials: {clusters_size[clust_nb]}')
                axs[row,col].set_ylim(ylims[row])

    def delete_artifact_trials(self):
        """
        Removes permanently from the dataset the cluster of outliers trials
        as classified by cluster_trials() (metadata_df.clusters == -1)
        """

        outliers_idx = self.metadata_df.index[self.metadata_df.cluster == -1].values

        self.metadata_df.drop(outliers_idx, axis=0, inplace=True)
        self.metadata_df.reset_index(inplace=True)
        self.metadata_df['trial_ID'] = self.metadata_df.index.values
        self.data = np.delete(self.data, outliers_idx, 0)


        print(f'removed {len(outliers_idx)} outliers trials from the dataset \r\n \
            {self.data.shape[0]} trials remaining')

    def filterout_clusters(self, clusters_to_exclude: list):
        """
        exclude one or several cluster of trials of the dataset
        """
        if isinstance(clusters_to_exclude, int):
            subject_IDs_to_exclude = [subject_IDs_to_exclude]
        filter = self.metadata_df['cluster'].apply(lambda x: x in clusters_to_exclude)
        self.metadata_df.loc[filter,'keep'] = False

    def export_to_sktime(self,
            folder: str = None,
            name: str = None,
            vars_to_export: list = 'all',
            verbose: bool = True):
        '''
        Convert 3D numpy array into nested dataframe (with series in each cell)
        This is the standard format for multivariate timeseries in sktime.org
        sktime is used to perform classification and regression time on uni or
        multi-variate timeseries (unlike scikit-learn, more suited for tabular data)
        if a name is specified (and optionally a folder), the tranformed DataFrame
        will be stored as a pickle file.
        '''
        cols_idx, cols_names = self.checker_vars_to_export(vars_to_export)

        if len(cols_idx) > 1:

            data = self.data[:,cols_idx,:]

            X = convert.from_3d_numpy_to_nested(
                data,
                column_names = cols_names,
                cells_as_numpy=False)
        else:
            # if only one columns, output a 2D numpy array
            # dimensions = samples x timepoints 
            data = self.data[:,cols_idx,:].squeeze()
            X = data
            # X = convert.from_2d_array_to_nested(
            #     data, 
            #     index = self.metadata_df.index.values,
            #     time_index = self.time_vector,
            #     cells_as_numpy = False)
            
        if hasattr(self, 'cond_aliases'):
            y = np.array([self.cond_aliases[cond] for cond in 
                self.metadata_df['condition_ID'].values])    
        else:
            y = self.metadata_df['condition_ID'].values

        if name:
            if folder:
                fullpath = os.path.join(folder, name + '.pkl')
            else:
                fullpath = os.path.join(os.getcwd(), name + '.pkl')
            
            with open(fullpath, 'wb') as file:
                pickle.dump([X,y], file, protocol=pickle.HIGHEST_PROTOCOL)

            if verbose:
                print(f'Nested Dataframe saved in {fullpath}')
                mem_mb = sys.getsizeof(X)/1024/1024
                str_mem = f'sktime-format data size: {mem_mb} Mb'
                print(str_mem)

        return X, y

    def checker_vars_to_export(self, vars_to_export):
        if vars_to_export == 'all':
            cols_idx = [i for i in range(self.data.shape[1])]
            cols_names = list(self.colnames_dict.keys())

        # to check vars_to_export as a list of column names
        elif isinstance(vars_to_export, list) and all(
            [var in self.colnames_dict.keys() for var in vars_to_export]):
            cols_idx = [self.colnames_dict[var] for var in vars_to_export]
            cols_names = vars_to_export
        
        # to check vars_to_export as a list of column numbers
        elif isinstance(vars_to_export, list) and all(
            [var in self.colnames_dict.values() for var in vars_to_export]):
            cols_idx = vars_to_export
            cols_names = [list(self.colnames_dict.keys()
                )[list(self.colnames_dict.values()).index(col_idx)] for col_idx in cols_idx] 

        elif isinstance(vars_to_export, str) and vars_to_export in self.colnames_dict.keys():
            cols_idx = [self.colnames_dict[vars_to_export]]
            cols_names = [vars_to_export]

        else:
            raise ValueError('vars_to_export appears not correct')
        return cols_idx,cols_names

    def transform_variables(self, on_vars: VarsType, function: callable, deriv_name: str):
        ...

    # TODO: Won't probably adapt for only one condition (axes will not be np.arrays)
    def lineplot(
            self,
            vars: VarsType = 'all',
            time_lim: Optional[list] = None,
            time_unit: str = None,
            error: bool = None, # only for group plot
            is_x_vs_y: bool = False, # implement here or not?
            plot_subjects: bool = True,
            plot_groups: bool = True,
            ylim: list = None, 
            colormap: str = 'jet',
            figsize: tuple = (20, 10),
            dpi: int = 100,
            box: bool = False,
            linex0: bool = True,
            liney0: bool = True, # draw horizontal gray dashed line at y = 0
            legend: bool = True,
            axs: Axes = None,
            verbose: bool = False):
        """
        Arguments
        ---------
        self
        vars: VarsType = 'all',
            #TODO what to expect?
        time_lim: Optional[list] = None,
            #TODO better called xlim?
            Specify the xlim of Axes
        time_unit: str = None,
            's', 'seconds', 'ms', or 'milliseconds'
        error: bool = None, 
            Only for group plot
        is_x_vs_y: bool = False, 
            implement here or not?
        plot_subjects: bool = True,
            Whether to plot individual subject data.
        plot_groups: bool = True,
            Whether to plot group data.
        ylim: list = None, 
        colormap: str = 'jet',
            trialexp.utils.pycontrol_utilities.cmap10 provides the 10 default plotting colors in matplotlib as a colormap
        figsize: tuple = (20, 10),
        dpi: int = 100,
        box: bool = False,
        liney0:bool = True, 
            draw horizontal gray dashed line at y = 0
        linex0:bool = True, 
            draw vertical gray dashed line at x = 0
        #TODO linex0 :bool = True, 
        legend: bool = True,
        verbose: bool = False
        Returns
        _______
        fig, 
        axs, 
        out_df : DataFrame
            A summary table to show the counting sample numbers.
            Can be improved.
            #TODO Is this format useful? I am not sure! You need to dig a lot.
            Instead of nesting, we can have dataframes and list of dataframes etc to be joined when necessary
            Or to have already joined, flat big table???      
        """
        plt.ion()
        plt.rcParams["figure.dpi"] = dpi
        plt.rcParams['font.family'] = ['Arial']
        
        # vars, time_vec, vars_idx = self.check_vars_and_times_input(vars, time_unit)
        vars_dict, time_vec = self.check_vars_and_times_input(vars, time_unit)

        if ylim != None and isinstance(ylim, list):
            if isinstance(ylim[0],list) and \
                all([len(yvar) == 2 for yvar in ylim]) and \
                len(ylim) == len(vars):

                ...
            else:
                raise ValueError(
                    'ylim must be a list of 2 items lists e.g.: [[0.19, 0.2],[-0.01, 0.01]]')
            
        # Do not take trials/subjects filtered out
        filtered_df = self.metadata_df[self.metadata_df['keep'] == True]
        # Perform grouping, this could be customized later
        gby = filtered_df.groupby(['condition_ID', 'group_ID','subject_ID'])
        gby = gby.agg({'trial_ID': list})

        # Get conditions and groups from filtered metadata
        condition_IDs = gby.index.get_level_values('condition_ID').unique()
        group_IDs = gby.index.get_level_values('group_ID').unique()
        # Compute color maps and assign colors to subjects
        group_colors = cm.get_cmap(colormap, self.metadata_df['group_ID'].nunique())
        subj_colors = cm.get_cmap(colormap, self.metadata_df['subject_ID'].nunique())
        subj_dict = {subject: subject_idx for (subject_idx, subject) 
            in enumerate(filtered_df['subject_ID'].unique())}
       
        # Adapt layout, this could be in a separate class/method
        if axs is None:
            # Set title as condition on the first line
            if len(group_IDs) == 1:
                group_colors = cm.get_cmap(colormap, self.metadata_df['condition_ID'].nunique())
                if plot_groups and not plot_subjects:
                    
                    fig, axs = plt.subplots(len(vars), 1, sharex= 'all',
                        sharey = 'row', squeeze = False , figsize = figsize)
                else:
                    fig, axs = plt.subplots(len(vars), len(condition_IDs)+1, sharex= 'all',
                        sharey = 'row', squeeze = False , figsize = figsize)
            else:
                group_colors = cm.get_cmap(colormap, len(self.groups))

                fig, axs = plt.subplots(len(vars), len(condition_IDs), sharex= 'all',
                    sharey = 'row', squeeze = False , figsize = figsize)
        else:
            if axs.ndim == 1:
                axs = axs.reshape(1, -1)

            for a in axs:
                for b in a:
                    assert isinstance(b, matplotlib.axes.Axes), \
                        f'axs must be an array (like) of matplotlib.axes.Axes '

            fig = None
            assert axs.shape[0] == len(vars), \
                f'The row numner of axs {axs.shape[0]} must match length of vars {len(vars)}'

            if len(group_IDs) == 1:
                group_colors = cm.get_cmap(colormap, len(self.conditions))

                if plot_groups and not plot_subjects:
                        
                    assert axs.shape[1] == len(vars), \
                        f'The column numner of axs {axs.shape[0]} must be 1'

                else:

                    assert axs.shape[1] == len(condition_IDs)+1, \
                        f'The column numner of axs {axs.shape[1]} must be {len(condition_IDs)+1}'
                   
            else:
                group_colors = cm.get_cmap(colormap, len(self.groups))

                assert axs.shape[1] == len(condition_IDs)+1, \
                    f'The column numner of axs {axs.shape[1]} must be {len(condition_IDs)}'

      
        group_dfs = [0] * len(condition_IDs)
        cond_n = [0] * len(condition_IDs) # trial counts per condition
        
        # Replace cond_aliases with condition_IDs if not provided
        if not hasattr(self, 'cond_aliases'):
            self.cond_aliases = [cond for cond in condition_IDs]
        
        for cond_idx, cond_ID in enumerate(condition_IDs):

            # Set title as condition on the first line
            if len(group_IDs) == 1:
                if plot_groups and not plot_subjects:
                    ...
                else:
                    axs[0, cond_idx+1].set_title(str(self.cond_aliases[cond_idx]))
            else:
                    axs[0, cond_idx].set_title(str(self.cond_aliases[cond_idx]))

                    
            # Compute means and sems
            group_n = [0] * len(group_IDs)
            subj_dfs = [None] * len(group_IDs)
            for g_idx, group_ID in enumerate(group_IDs):
                subj_subset = gby.loc[cond_ID, group_ID ,:].index.values
                # used to work using the following, possibly previous for behaviour only?
                # subj_subset = gby.loc[cond_ID, group_ID ,:].index.get_level_values(2).values

                subj_n = [0] * len(subj_subset)
                for subj_idx, subject in enumerate(subj_subset):
                    if verbose:
                        print(f'cond_ID: {cond_ID}, group_idx {group_ID}, subj {subject}')

                    trial_idx = np.array(gby.loc[(cond_ID, group_ID, subject),'trial_ID'])
                    np_val = self.data[trial_idx,:,:]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        mean_subj = np.nanmean(np_val,axis=0)
                    
                    mean_subj = mean_subj[list(vars_dict.values()), :]
                    # sem_subj = np_val.std(axis=0) / np.sqrt(len(trial_idx)-1)
                    # sem_subj = sem_subj[vars_idx, :]
                    
                    cond_n[cond_idx] = cond_n[cond_idx] + len(trial_idx)
                    group_n[g_idx] = group_n[g_idx] + len(trial_idx)
                    subj_n[subj_idx] = len(trial_idx)

                    # Plot
                    if plot_subjects:
                        for ax_idx, (var_name, col_idx) in enumerate(vars_dict.items()):
                            if len(group_IDs) == 1:
                                axs[ax_idx, cond_idx+1].plot(
                                    time_vec, mean_subj[ax_idx, :],
                                    alpha = 0.7,
                                    label = f'{subject} (n = {len(trial_idx)})',
                                    color = subj_colors(subj_dict[subject]))
                            else:
                                axs[ax_idx, cond_idx].plot(
                                    time_vec, mean_subj[ax_idx, :],
                                    alpha = 0.7,
                                    label = f'{subject} (n = {len(trial_idx)})',
                                    color = subj_colors(subj_dict[subject]))


                    if subj_idx == 0:
                        
                        mean_group = mean_subj
                  
                    elif subj_idx == 1:
                        mean_group = np.stack([mean_group, mean_subj], axis=0)
                    else:
                        mean_group = np.concatenate([mean_group, np.expand_dims(mean_subj, axis=0)], axis=0)
                
                # subj_dfs[g_idx] = pd.DataFrame(
                #     [[cond_ID] * len(subj_subset), [group_ID] * len(subj_subset), subj_subset, subj_n])
                # subj_dfs[g_idx] = subj_dfs[g_idx].transpose()
                # subj_dfs[g_idx].columns = ['cond_ID', 'group_ID', 'subject_ID', 'subject_trial_n']

                subj_dfs[g_idx] = pd.DataFrame(list(zip([group_ID] * len(subj_subset), subj_subset, subj_n)))
                subj_dfs[g_idx].columns = ['group_ID', 'subject_ID', 'subject_trial_n']
                # Group computations
                if plot_groups:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        sem_group = np.nanstd(mean_group, axis=0) / np.sqrt(mean_group.shape[0]-1)
                        mean_group = np.nanmean(mean_group, axis=0) # the mean of the mean per subject(animal), rather than the mean of all trials

                    # Group plotting
                    group_lw = 1    
                    for ax_idx, (var_name, col_idx) in enumerate(vars_dict.items()):
                        if len(group_IDs) == 1:
                            if g_idx == 0 and cond_idx == 0:
                                axs[ax_idx, 0].set_ylabel(vars[ax_idx])
                            # if a group is more than a single subject
                            if len(mean_group.shape) > 1:
                                axs[ax_idx, 0].plot(time_vec, mean_group[ax_idx, :], lw=group_lw,
                                    color = group_colors(cond_ID), label=self.cond_aliases[cond_idx])
                
                                if error is not None:
                                    # fill sem
                                    axs[ax_idx, 0].fill_between(time_vec, 
                                        mean_group[ax_idx, :] - sem_group[ax_idx, :],
                                        mean_group[ax_idx, :] + sem_group[ax_idx, :],
                                        alpha=0.2, color=group_colors(cond_ID), lw=0)

                            # if single subject in the group
                            else:
                                axs[ax_idx, 0].plot(time_vec, mean_group, lw=group_lw,
                                color = group_colors(cond_ID), label=self.cond_aliases[cond_idx])

                        else:
                            if g_idx == 0 and cond_idx == 0:
                                axs[ax_idx, 0].set_ylabel(vars[ax_idx])
                            
                            # if a group is more than a single subject
                            if len(mean_group.shape) > 1:    
                                # plot mean
                                axs[ax_idx, cond_idx].plot(time_vec, mean_group[ax_idx, :],
                                    lw=group_lw, color=group_colors(group_ID),
                                    label = group_ID)                    
                                
                                if error is not None:
                                    # fill sem
                                    axs[ax_idx, cond_idx].fill_between(time_vec, 
                                        mean_group[ax_idx, :] - sem_group[ax_idx, :],
                                        mean_group[ax_idx, :] + sem_group[ax_idx, :],
                                        alpha=0.3, color=group_colors(group_ID), lw=0)  
                            # if single subject in the group
                            else:
                                axs[ax_idx, cond_idx].plot(time_vec, mean_group,
                                    lw=group_lw, color=group_colors(group_ID),
                                    label = group_ID)    

                        if ax_idx == len(vars_dict.values())-1:
                            if plot_groups and not plot_subjects:
                                ...
                            else:
                                axs[ax_idx, cond_idx].set_xlabel(time_unit)
                            if len(self.groups) == 1:
                                if plot_groups and not plot_subjects:
                                    [ax[0].set_xlabel(time_unit) for ax in axs]
                                else:
                                    axs[ax_idx, cond_idx+1].set_xlabel(time_unit)

            subj_dfs = pd.concat(subj_dfs)

            group_dfs[cond_idx] = pd.DataFrame(list(zip([cond_ID] * len(group_IDs),  
                [str(self.cond_aliases[cond_idx])] * len(group_IDs),  group_IDs, group_n)))

            group_dfs[cond_idx].columns = ['condition_ID', 'condition_alias', 'group_ID', 'group_trial_n']

            group_dfs[cond_idx] = pd.merge(group_dfs[cond_idx], subj_dfs, 'outer') 

        out_df = pd.DataFrame(list(zip(condition_IDs, cond_n)))
        out_df.columns=['condition_ID', 'condition_trial_n']

        group_dfs = pd.concat(group_dfs)

        out_df = pd.merge(out_df, group_dfs, 'outer')
        #TODO Is this format useful? I am not sure! You need to dig a lot.
        # Instead of nesting, we can have dataframes and list of dataframes etc to be joined when necessary
        # Or to have already joined, flat big table???


                 
        if time_lim:
            axs[0,0].set_xlim([time_lim[0], time_lim[1]])
        else:        
            axs[0,0].set_xlim([time_vec[0], time_vec[-1]])
        

        if ylim:
            for r, var in enumerate(vars):
                axs[r,0].set_ylim(ylim[r][0], ylim[r][1])
        
        if legend:
            for r in range(axs.shape[0]):
                if len(axs.shape) > 1:
                    for c in range(axs.shape[1]):
                        axs[r,c].legend()
                else:
                    axs[r].legend()
        
        if not box:
            for r in range(axs.shape[0]):
                if len(axs.shape) > 1:
                    for c in range(axs.shape[1]):
                        axs[r, c].spines['top'].set_visible(False)
                        axs[r, c].spines['right'].set_visible(False)
                else:
                    axs[r].spines['top'].set_visible(False)
                    axs[r].spines['right'].set_visible(False)

        if liney0:
            for r in range(axs.shape[0]):
                if len(axs.shape) > 1:
                    for c in range(axs.shape[1]):
                        axs[r, c].plot([time_vec[0], time_vec[-1]],
                            [0, 0], 
                            color=(0.7, 0.7, 0.7),
                            linestyle='--',
                            zorder=0.5, # send to back
                            )
                else:
                    axs[r].plot([time_vec[0], time_vec[-1]],
                        [0, 0],
                        color=(0.8, 0.8, 0.8),
                        linestyle=':',
                        zorder=0.5,  # send to back
                        lw=0.5,
                        )
        #plt.show()
        if linex0:
            for r in range(axs.shape[0]):
                if len(axs.shape) > 1:
                    for c in range(axs.shape[1]):
                        axs[r, c].plot([0, 0],
                            axs[r, c].get_ylim(), 
                            color=(0.7, 0.7, 0.7),
                            linestyle='--',
                            zorder=0.5, # send to back
                            )
                else:
                    axs[r].plot([0, 0],
                        axs[r].get_ylim(),
                        color=(0.8, 0.8, 0.8),
                        linestyle=':',
                        zorder=0.5,  # send to back
                        lw=0.5,
                        )
        #plt.show()
        return fig, axs, out_df

    def check_vars_and_times_input(self, vars, time_unit):
        if time_unit == None and not hasattr(self, 'time_unit'):
            time_vec = self.get_time_vector()
        else:
            time_vec = self.get_time_vector(unit = time_unit)

        if vars == 'all':
            vars = self.get_col_names()
            vars_idx = list(self.colnames_dict.values())

        if isinstance(vars, str):
            vars = [vars]
        
        if any([var not in self.colnames_dict.keys() for var in vars]):
            wrong_var_idx = [var not in self.colnames_dict.keys() for var in vars]
            raise ValueError(
                f'Variable(s) not in the dataset: {np.array(vars)[wrong_var_idx]}')
        else:
            vars_dict = dict() # with keys as var names and values as columns indices
            for var in vars:
                vars_dict[var] = [col_value for col_key, col_value in self.colnames_dict.items() if col_key == var][0]
        
        return vars_dict, time_vec

    def scatterplot(self, vars: VarsType, groupby: Optional[list] = ['group_ID', 'subject_ID'], \
            timelim: Optional[list] = None):
        ...

    def heatmap(self,
            vars: VarsType = 'all',
            time_lim: Optional[list] = None,
            time_unit: str = None,
            plot_by_conditions: bool = True,
            plot_by_sessions: bool = False,
            plot_by_subjects: bool = True,
            plot_by_groups: bool = True,
            ylim: list = None,
            clim_pctile: float = None, # pctile of values that will have min/max color    
            colormap: str = 'jet',
            figsize: tuple = (20, 10),
            dpi: int = 100,
            box: bool = False,
            liney0:bool = True, # draw horizontal gray dashed line at y = 0
            legend: bool = True,
            verbose: bool = False):
        """
        This function could share most of the comuputation with Continous_Dataset.lineplot()
        Heatmap representation, rather than line plot, of multiple continuous data, 
        typically representing individual mice or neurons.

        return a list of figures and a list of subject_IDs
        """
        
        vars_dict, time_vec = self.check_vars_and_times_input(vars, time_unit)


        if not hasattr(self, 'cond_aliases'):
            present_cond_IDs = list(set(self.metadata_df[self.metadata_df.keep == True].condition_ID.values))
            self.cond_aliases = present_cond_IDs


        # Do not take trials/subjects filtered out
        filtered_df = self.metadata_df[self.metadata_df['keep'] == True].copy()
        # Perform grouping, this could be customized later
        gby = filtered_df.groupby(['group_ID','subject_ID','condition_ID'])
        gby = gby.agg({'trial_ID': list})

        # Get conditions and groups from filtered metadata
        condition_IDs = gby.index.get_level_values('condition_ID').unique()
        group_IDs = gby.index.get_level_values('group_ID').unique()
        # Compute color maps and assign colors to subjects
        # group_colors = cm.get_cmap(colormap, self.metadata_df['group_ID'].nunique())
        # subj_colors = cm.get_cmap(colormap, self.metadata_df['subject_ID'].nunique())
        subj_dict = {subject: subject_idx for (subject_idx, subject) 
            in enumerate(filtered_df['subject_ID'].unique())}
       
        plt.rcParams["figure.dpi"] = dpi
        plt.rcParams['font.family'] = ['Arial']

        # create empty list of figures
        figs = ['empty fig' for subject in subj_dict.keys()]
        # basic loop for plot
        for group_ID in group_IDs:
            for subject_ID, subject_idx in subj_dict.items():

                fig_title = f'{subject_ID}'
                figs[subject_idx], axs = plt.subplots(len(vars), len(condition_IDs), sharex= 'all',
                    squeeze = False , figsize = figsize)
                    # sharey = 'row', squeeze = False , figsize = figsize)
                figs[subject_idx].suptitle(fig_title)
                for row_idx, (var_name, col_idx) in enumerate(vars_dict.items()):

                    for cond_idx, condition_ID in enumerate(condition_IDs):
                        
                        subplot_title = f'{var_name} - {self.cond_aliases[cond_idx]}'
                        data = self.data[gby.loc[(group_ID, subject_ID, condition_ID), 'trial_ID'], col_idx, :]
                        if clim_pctile:
                            vmin = np.percentile(data, clim_pctile)
                            vmax = np.percentile(data, 100-clim_pctile)
                        else:
                            vmin = data.min()
                            vmax = data.max()

                        axs[row_idx, cond_idx].set_title(subplot_title)

                        axs[row_idx, cond_idx].pcolormesh(
                            self.get_time_vector(),
                            range(data.shape[0]), #gby.loc[(group_ID, subject_ID, condition_ID), 'trial_ID'],
                            data,
                            cmap = colormap,
                            vmin=vmin,
                            vmax=vmax)
                        
                        if time_lim:
                            axs[row_idx, cond_idx].set_xlim([time_lim[0], time_lim[1]])

        # return a 
        return figs, list(subj_dict.values())

class Event_Dataset(Trials_Dataset):
    """
    Subclass of Trials_Dataset.
    Methods
    -------
    raster()
        to be implemented
    peth()
        to be implemented, can be integrated with raster
    compute_distribution(...)
        Compute distribution of events for each session.
    """
    def __init__(self, data: pd.DataFrame, metadata_df: pd.DataFrame):
            super().__init__(data, metadata_df)
            # TODO: Consider inputing colnames only as list or tuple
            # and compute the dictionary '<names>': <idx_col(int)> in __init__

    def raster(self, kvargs):
        ...
        #TODO this is important

    

    def _checker_events_request(self, events_to_plot: list = 'all'):
        """
        Helper function that helps to validate and match events requested as an argument of 
        a particular function to the proper dataframe columns of the dataset
        
        """

        event_cols = [event_col for event_col in self.data.columns if '_trial_time' in event_col]
        event_names = [event_col.split('_trial_time')[0] for event_col in event_cols]

        if events_to_plot == 'all':
            pass
        
        elif isinstance(events_to_plot, list):

            # check if events requested exist
            check = all(ev in event_names for ev in events_to_plot)

            if not check:
                raise Exception('Check your list of requested events, event not found')
            
            event_cols = [ev + '_trial_time' for ev in events_to_plot]
            event_names = events_to_plot

        elif isinstance(events_to_plot, str):
            if events_to_plot not in event_names:
                raise Exception('Check the name of your requested event, event not found')
            
            event_cols = [events_to_plot + '_trial_time']
            event_names = [events_to_plot]

        else:
            raise Exception('bad format for requesting plot_trials events')
        
        return event_cols, event_names

    def plot_trials(self, events_to_plot:list = 'all',  sort:bool = False):

        # I dont get that K, review symbol selection? 
        raw_symbols  = SymbolValidator().values
        symbols = [raw_symbols[i+2] for i in range(0, len(raw_symbols), 12)]

        event_cols, event_names = self._checker_events_request(events_to_plot)

        # Implement this as abstract method to check requested arguments (events) match the session obj.

        plot_names =  [trig + ' ' + event for event in event_cols for trig in self.data.trigger.unique()]

        # https://plotly.com/python/subplots/
        # https://plotly.com/python/line-charts/
        fig = make_subplots(
            rows= len(event_cols), 
            cols= len(self.data.trigger.unique()), 
            shared_xaxes= True,
            shared_yaxes= True,
            subplot_titles= plot_names
        )

        for trig_idx, trigger in enumerate(self.data.trigger.unique()):
            
            # sub-selection of df_events based on trigger, should be condition for event_dataset class
            df_subset = self.data[self.data.trigger == trigger]


            for ev_idx, event_col in enumerate(event_cols):
                # if sort:
                #     min_times = df_subset[event_cols[ev_idx]].apply(lambda x: find_min_time_list(x))
                #     min_times = np.sort(min_times)

                ev_times = df_subset[event_cols[ev_idx]].apply(lambda x: np.array(x)).values
                ev_trial_nb = [np.ones(len(array)) * df_subset.index[idx] for idx, array in enumerate(ev_times)]

                ev_trial_nb = np.concatenate(ev_trial_nb)
                ev_times =  np.concatenate(ev_times)

                fig.add_shape(type="line",
                    x0=0, y0=1, x1=0, y1= ev_trial_nb.max(),
                    line=dict(
                    color="Grey",
                    width=2,
                    dash="dot"
                    ),
                    row= ev_idx+1,
                    col = trig_idx+1)

                fig.add_trace(
                    go.Scatter(
                        x= ev_times/1000,
                        y= ev_trial_nb,
                        name= event_names[ev_idx],
                        mode='markers',
                        marker_symbol=symbols[ev_idx % 40]
                        ),
                        row= ev_idx+1,
                        col = trig_idx+1)

                    

                fig.update_xaxes(
                    title_text = 'time (s)',
                    ticks = 'outside',
                    ticklen = 6,
                    tickwidth = 2,
                    tickfont_size = 12,
                    showline = True,
                    linecolor = 'black',
                    # range=[self.trial_window[0]/1000, self.trial_window[1]/1000]
                    autorange = True,
                    row = ev_idx+1,
                    col = trig_idx+1
                    )
                
                fig.update_yaxes( 
                    title_text = 'trial nb', 
                    ticks = 'outside',
                    ticklen = 6,
                    tickwidth = 2,   
                    tickfont_size = 12,
                    showline = True,
                    linecolor = 'black',
                    range = [0, ev_trial_nb.max()+1],
                    showgrid=True,
                    row = ev_idx+1,
                    col = trig_idx+1
                    )

        fig.update_layout(
            # get metadata right
            # title_text= f'Events Raster plot, ID:{self.subject_ID} / {self.task_name} / {self.datetime_string}',
            height=800,
            width=800
                        
        )

        fig.show()

        return fig


    def compute_distribution(
            self,
            trial_window: Iterable = None, 
            bin_size: int = 100, # by default in ms, not adapted yet for seconds
            normalize: bool = True, # normalize the count of events according to bin_size
            per_session: bool = False, # if false, compute distribution per animal for all its sessions
            out_as_continuous: bool = False, # if true, output a continuous dataset
            verbose: bool = False):
        """
        Compute distribution of events for each session.
        Output a continuous_dataset instance if out_as_continuous = True
        Arguments
        ---------
        self
        trial_window: Iterable = None,
        bin_size: int = 100, 
            by default in ms, not adapted yet for seconds
        normalize: bool = True, 
            normalize the count of events according to bin_size
        per_session: bool = False, 
            if False, compute distribution per animal for all its sessions
        out_as_continuous: bool = False, 
            if True, output a Continuous_Dataset object
        verbose: bool = False
        Returns
        -------
        grouped_df : DataFrame
            default
            only if not out_as_continuous
        dist_as_continuous : Continuous_Dataset
            only if out_as_continuous
        """

        if trial_window == None and hasattr(self, 'trial_window'):
            trial_window = self.trial_window
        elif trial_window == None and not hasattr(self, 'trial_window'):
            if verbose:
                print('please set trial_window using <event_dataset>.set_trial_window([tlim0, tlim1])')
            raise NotPermittedError('compute_distribution',
                ['trial_window', 'Event_Dataset.trial_window'], ['None', 'None'])
        
        # select data if filtered
        filtered_meta_df = self.metadata_df[self.metadata_df['keep'] == True]
        filtered_ev_df = self.data[self.metadata_df['keep'] == True]
        
        # TODO: Merge instead of concatenate behavioural data and conditions DataFrames
        full_df = filtered_ev_df.merge(filtered_meta_df, on='uid')
    
        # Extract behavioural times columns names and create new ones for the distribution
        ev_times_cols = [col for col in self.data.columns if search('trial_time', col)]
        dist_col_names = [col.split('_trial_time',1)[0] + '_dist' for col in ev_times_cols]
        
        # define function for aggregating event times
        # func_agg_event_times = [np.hstack for i in range(len(ev_times_cols))]
        # # define
        
        # agg_dict = dict(zip(ev_times_cols, func_agg_event_times))
        agg_dict = dict(zip(ev_times_cols, [np.hstack for i in range(len(ev_times_cols))]))
        # To compute nb of trials in the groupby
        agg_dict['trial_ID'] = len
        agg_dict['datetime'] = lambda x: x.iloc[0]

        # from functools import reduce
        # def func_get_first_from_agg(serie: pd.Series = None):
        #     return reduce(lambda x: x.iloc[0])

        if per_session:
            grouped_df = full_df.groupby(
                ['condition_ID', 'group_ID', 'subject_ID', 'session_nb']).agg(agg_dict)
        else:
            grouped_df = full_df.groupby(
                ['condition_ID', 'group_ID', 'subject_ID']).agg(agg_dict)

        grouped_df.reset_index()

        for c_idx, col in enumerate(ev_times_cols):
            
            # Compute histogram without returning bin edges
            grouped_df[dist_col_names[c_idx]] = grouped_df[col].apply(
                lambda x: histo_only(x, trial_window, bin_size))

            # Normalize by trial number and histogram size
            # Will only work now if all data in milliseconds
            if normalize:
                grouped_df[dist_col_names[c_idx]] = \
                    ((grouped_df[dist_col_names[c_idx]] * (1000 / bin_size)) / grouped_df['trial_ID'])
                      # will onl
        
        if out_as_continuous:
            # build a 3d numpy array with dimensions:
            # n sessions/trials/subjects X m variables X t time bins
            colnames_dict = dict(zip(dist_col_names,range(len(dist_col_names))))
            for var_idx, col in enumerate(dist_col_names):
                if var_idx == 0:    
                    np_3d = np.expand_dims(np.vstack(grouped_df[col].values),1)
                else:
                    np_3d = np.concatenate(
                        [np_3d, np.expand_dims(np.vstack(grouped_df[col].values),1)], axis = 1)

            # Dataset instance creation
            dist_as_continuous = Continuous_Dataset(
                np_3d, grouped_df.reset_index(), colnames_dict)

            # Pass groups of the Event_Dataset to the
            # Continuous_Dataset instance
            dist_as_continuous.groups = self.groups

            # Adapt and set the trial window considering
            # the binning of the histogram
            dist_as_continuous.set_trial_window(
                [trial_window[0] + bin_size / 2, trial_window[1] - bin_size / 2])

            return dist_as_continuous
        else:
            return grouped_df

    def analyse_successrate(self, 
        conditions_succcess: list,
        conditions_failure : list,
        subject_IDs: list = None, 
        group_IDs : list = None, 
        bywhat: str = 'sessions', 
        ax = None):
        """
        - susscess is alreday computed by Experiment.compute_success
            but you still need to specify conditions from conditions_list
        - failure, on the other hand, is less clear
        - Sessions with 0 trial (success or failure) are ignored.
        conditions_succcess : list
            list of integers
                eg. [0]
                To specify conditions for success from self.conditions and self.cond_aliases
                The index starts from 0.
            or (list or Series of bool)
                To directly specifiy which trials to be considered success.
                The length must match the height of the self.metadata_df.
        conditions_failure : list
            (list of integers) 
                eg. [1]
                To specify conditions for success from self.conditions and self.cond_aliases
                The index starts from 0.
            or (list or Series of bool)
                To directly specifiy which trials to be considered success.
                The length must match the height of the self.metadata_df.
        subject_IDs: list = None
        group_IDs : list = None
        bywhat : str = 'sessions'
            'sessions', 'sessions_with_gaps', 'days', 'days_with_gaps', 'dates'
            'sessions' will renumber sessions so there will be no gaps shown.
            'days' will recount days so there will be no gaps shown.
            'session_with_gaps' 'days_with_gaps', and 'dates'will include gaps.
        ax: matplotlib.axes.Axes = None
        #TODO to support the removal of human interventions
        """
     
        assert bywhat in ['sessions', 'sessions_with_gaps', 'days', 'days_with_gaps', 'dates'], "bywhat is invalid"

        def get_gr_df(self, conditions_succcess, conditions_failure, group_IDs, subject_IDs):

           # parse conditions_succcess and conditions_failure
            if type(conditions_succcess) is list:
                if all([type(c) is int for c in conditions_succcess]):
                    is_cond_success_int = True
                elif all([type(c) is bool for c in conditions_succcess]):
                    assert len(conditions_succcess) == self.metadata_df.shape[0]
                    is_cond_success_int = False
                    conditions_succcess = pd.Series(conditions_succcess)
                else:
                    raise Exception('conditions_succcess is invalid')
            elif type(conditions_succcess) is pd.core.series.Series:
                is_cond_success_int = False
            else:
                raise Exception('conditions_succcess is invalid')

            if type(conditions_failure) is list:
                if all([type(c) is int for c in conditions_failure]):
                    is_cond_failure_int = True
                elif all([type(c) is bool for c in conditions_failure]):
                    assert len(
                        conditions_failure) == self.metadata_df.shape[0]
                    is_cond_failure_int = False
                    conditions_failure = pd.Series(conditions_failure)
                else:
                    raise Exception('conditions_failure is invalid')
            elif type(conditions_failure) is pd.core.series.Series:
                is_cond_failure_int = False
            else:
                raise Exception('conditions_failure is invalid')


            metadata_df = self.metadata_df.loc[self.metadata_df['keep'] 
                & self.metadata_df['valid'], :].copy()

            if group_IDs is None:
                group_IDs = list(set(metadata_df['group_ID']))

            for g in group_IDs:
                if subject_IDs is None:
                    subject_IDs_ = list(
                        set(metadata_df.loc[metadata_df.group_ID == g, 'subject_ID']))
                else:
                    subject_IDs_ = subject_IDs

                ss_dfs = [0] * len(subject_IDs_)
                for s_idx, s in enumerate(subject_IDs_):
                    session_nbs = list(set(metadata_df.loc[
                        (metadata_df['group_ID'] == g)
                        & (metadata_df['subject_ID'] == s),
                        'session_nb'])).copy()

                    ss_sc = [np.nan] * len(session_nbs)
                    ss_fl = [np.nan] * len(session_nbs)
                    ss_tn = [np.nan] * len(session_nbs)
                    ss_sr = [np.nan] * len(session_nbs)
                    ss_dt = [None] * len(session_nbs) # import datetime; datetime.time()

                    for ss_idx, ss in enumerate(session_nbs):
                        if is_cond_success_int:
                            tf1 = pd.concat([metadata_df['condition_ID'] == c 
                                for c in conditions_succcess], axis=1).any(axis=1)
                        else:
                            tf1 = conditions_succcess

                        TF_success = (metadata_df['group_ID'] == g) & (metadata_df['subject_ID'] == s) \
                            & (metadata_df['session_nb'] == ss) \
                            & (tf1)

                        ss_sc[ss_idx] = np.count_nonzero(metadata_df.loc[
                            TF_success, 'success'])

                        if is_cond_failure_int:
                            tf2 = pd.concat([metadata_df['condition_ID'] == c
                                             for c in conditions_failure], axis=1).any(axis=1)
                        else:
                            tf2 = conditions_failure
                        TF_failure = (metadata_df['group_ID'] == g) & (metadata_df['subject_ID'] == s) \
                            & (metadata_df['session_nb'] == ss) \
                            & (tf2)

                        ss_fl[ss_idx] = np.count_nonzero(~metadata_df.loc[
                            TF_failure, 'success'])

                        ss_tn[ss_idx] = ss_sc[ss_idx] + ss_fl[ss_idx]
                        # if ss_tn[ss_idx] == 0, then this session is invalid. We should not count this.

                        dt = (metadata_df.loc[(TF_success) | (TF_failure), 'datetime']).copy()
                        if not dt.empty:
                            ss_dt[ss_idx] = dt.iloc[0]
                      
                    np.seterr(divide='ignore', invalid='ignore')
                    # https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
                    ss_sr = (np.array(ss_sc)/np.array(ss_tn)).tolist()

                    ss_df = pd.DataFrame(list(zip(session_nbs, ss_sc, ss_fl, ss_tn, ss_sr, ss_dt)))
                    ss_df.columns = ['session_nb', 'success_n', 'failure_n','trial_n', 'success_rate', 'datetime']

                    ss_df.astype({'session_nb': 'int', 'success_n': 'int', 'failure_n': 'int', 'trial_n': 'int'})

                    ss_df['subject_ID'] = pd.Series([s] * len(session_nbs))

                    ss_df.drop(labels = [ i for i, x in enumerate([np.isnan(sr) for sr in ss_sr]) if x],
                        axis='index', inplace=True) # drop the rows with sr being NaN (also meaning trial_n == 0)

                    ss_dfs[s_idx] = ss_df

            ss_dfs = pd.concat(ss_dfs)
            gr_df = pd.DataFrame(list(zip([g] * len(subject_IDs_), subject_IDs_)),
                                columns=['group_ID', 'subject_ID'])

            gr_df = pd.merge(gr_df, ss_dfs, 'outer')
            gr_df['date'] = gr_df['datetime'].dt.date  # TODO
            return gr_df
        

        def get_list_df_success_rate(gr_df: pd.DataFrame, bywhat: str, 
            group_IDs, subject_IDs):
            if group_IDs is None:
                group_IDs = list(set(gr_df['group_ID']))

            out_list = [None] * len(group_IDs)
            for g_idx, g in enumerate(group_IDs):
                if subject_IDs is None:
                    subject_IDs_ = list(
                        set(gr_df.loc[gr_df['group_ID'] == g, 'subject_ID']))
                else:
                    subject_IDs_ = subject_IDs
                
                out_listlist = [None] * len(subject_IDs_)
                for s_idx, s in enumerate(subject_IDs_):
                    thismouse_df = gr_df.loc[(gr_df['group_ID'] == g)
                        & (gr_df['subject_ID'] == s), :].copy()

                    if bywhat == 'days':
                        # gaps are ignored
                        # success rate is computed daily
                        dates = list(set(thismouse_df['datetime'].dt.date)) 

                        # delete NaT from the list because sort() doesn't like it
                        # But why there are NaT for dates????
                        for idx, tf in enumerate([pd.isnull(d) for d in dates]):
                            if tf:
                                del dates[idx]

                        dates.sort()
                        sr = [None] * len(dates)
                        for d_idx, d in enumerate(dates):
                            X = thismouse_df.loc[thismouse_df['datetime'].dt.date == d,
                                [ 'success_n', 'trial_n']].sum(axis=0)

                            sr[d_idx] = X.success_n/X.trial_n

                        df1 = pd.DataFrame(list(zip(range(1, len(dates)+1), sr)), columns=['dayN', str(s)])
                        df1 = df1.drop(df1[np.isnan(df1[str(s)])].index) # drop NaN rows
                        df1['dayN'] = range(1, df1.shape[0]+1)

                        out_listlist[s_idx] = df1.set_index('dayN')

                    elif bywhat == 'days_with_gaps':
                        # gaps are considered
                        # success rate is computed daily
                        dates = list(set(thismouse_df['datetime'].dt.date)) #TODO
                        dates.sort()
                        sr = [None] * len(dates)
                        for d_idx, d in enumerate(dates):
                            X = thismouse_df.loc[thismouse_df['datetime'].dt.date == d, #TODO
                                                ['success_n', 'trial_n']].sum(axis=0)

                            sr[d_idx] = X.success_n/X.trial_n

                        #TODO get rid of NaT from dates
                        out_listlist[s_idx] = pd.DataFrame(
                            list(zip([d - dates[0] for d in dates], sr)), columns=['dayN', str(s)])
                        # unsupported operand type(s) for -: 'datetime.date' and 'NaTType'

                        out_listlist[s_idx] = out_listlist[s_idx].set_index('dayN')

                    elif bywhat == 'dates':
                        # gaps are considered
                        # success rate is computed daily
                        # use dates instead of days
                        dates = list(set(thismouse_df['datetime'].dt.date)) #TODO
                        dates.sort()
                        sr = [None] * len(dates)
                        for d_idx, d in enumerate(dates):
                            X = thismouse_df.loc[thismouse_df['datetime'].dt.date == d,  # TODO
                                [ 'success_n', 'trial_n']].sum(axis=0)

                            sr[d_idx] = X.success_n/X.trial_n

                        out_listlist[s_idx] = pd.DataFrame(list(zip(dates, sr)), columns=['dates', str(s)])
                        out_listlist[s_idx] = out_listlist[s_idx].set_index('dates')

                    elif bywhat == 'sessions':
                        # gaps are ignored
                        # session with NaN will be ignored
                        # session number always starts with 1
                        # success rate is computed by session

                        df1 = thismouse_df.loc[:, ['session_nb', 'success_rate']].copy()
                        df1 = df1.drop(df1[np.isnan(df1['success_rate'])].index) # drop NaN rows
                        df1.sort_values(by=['session_nb'],inplace=True)
                        df1 = df1.reset_index(drop=True)
                        df1['session_nb'] = pd.Series(range(1, df1.shape[0]+1), dtype='int64')

                        out_listlist[s_idx] = df1.set_index('session_nb')
                        out_listlist[s_idx].columns = [str(s)]
                    
                    elif bywhat == 'sessions_with_gaps':
                        # gaps are maintained
                        # success rate is computed by session
                        out_listlist[s_idx] = thismouse_df.loc[:, ['session_nb', 'success_rate']].copy() 
                        out_listlist[s_idx] = out_listlist[s_idx].set_index('session_nb')
                        out_listlist[s_idx].columns = [str(s)]

                    else:
                        raise 'bywhat must be  ''sessions'', ''sessions_with_gaps'' ,''days'', ''days_with_gap'', or ''dates'''

                    #out_list[g_idx] = 1 = pd.concat(out_listlist) # TODO this is wrong
                    # session_nb as index, success_rate as value, subject_ID as columns
                    out_list[g_idx] = pd.concat(out_listlist, axis=1)

                    out_list[g_idx] = out_list[g_idx].sort_index()

            return out_list

        gr_df = get_gr_df(self, conditions_succcess, conditions_failure,
                        group_IDs, subject_IDs)

        out_list = get_list_df_success_rate(
            gr_df, bywhat, group_IDs, subject_IDs)

        # Plotting
        #TODO deal with groups

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        else:
            assert isinstance(ax, Axes)
            # may not work for subplotds
        nml = colors.Normalize(0, 1)

        im1 = ax.imshow(out_list[0].transpose(), norm=nml)

        if bywhat == 'sessions':
            ax.set_xlabel('Sessions')
        elif bywhat == 'sessions_with_gaps':
            ax.set_xlabel('Sessions')

        elif bywhat == 'days':
            ax.set_xlabel('Days')
        elif bywhat == 'days_with_gaps':
            ax.set_xlabel('Days')

        elif bywhat == 'dates':
            ax.set_xlabel('Dates')
            xticks = ax.get_xticks()

            ax.set_xticks(range(0, len(out_list[0].index),  int(xticks[1] - xticks[0])))
            ax.set_xticklabels(out_list[0].index[range(0, len(out_list[0].index),  int(xticks[1] - xticks[0]))])

        ax.set_ylabel('Mice')
        ax.set_yticks(range(0,out_list[0].shape[1]))
        ax.set_yticklabels(out_list[0].columns)
        ax.set_facecolor('k')
        ax.tick_params(axis='both', which='major', labelsize=8)

        return gr_df, out_list, ax, im1
    
    def plot_raster(
            self, keys: list = None, 
            conditions_IDs : list = None, 
            separate: bool = True, 
            colors : list = 'default',
            module: str = 'matplotlib',
            raster_y: str = 'trial', 
            target: np.array = None,
            figsize: tuple = None):
        """
        Raster plot for Event_Dataset
        # TODO introduce separation between conditions instead of/alternative to triggers
        keys : list = None
            what to plot
            Must match the names containing '*_trial_time' in the colums of self.data
        cond_ID : list = None
            what to use as cond_ID
            Must be subset of the self.cond_ID
        
        separate : bool = True
            If false, overlaid with different colors
        raster_y : char = 'trial' #TODO
            'trial' or 'time'
            Event_Dataset probably doesn't hold the timestamps of the trigger events.
            #Julien: Yes it does in ev_dataset.data['timestamp']
        colors : list = 'default'
            Specify plot colors for keys as list.
            By default, 'k' is used for separate = True, while 'C1', 'C2', ... is used for separate = False
            https://matplotlib.org/stable/tutorials/colors/colors.html
        module : str = 'matplotlib' (default) or 'plotly'
            'matplotlib' is suitable for publication/presentation.
            'plotly' provides scrollabbility. Note that the figures created by Plotly is pretty heavy 
            and not suitable for version control of Jupyter notebook. It is recommened to delete the figure before commiting.
        rastertype: str = 'lines' (default ) or 'markers' #TODO
            Sometimes 'markers' can be useful for visibility.
        target: np.array for matplotlib, an instance of plotly.graph_objects.Figure for plotly
            If separate is True, ax is an m by n np.arary of matplotlib.axes.Axes. 
            m is the number of columns with the name '*_trial_time' in self.data, 
            if keys is None or the length of keys if keys is not None.
            n is the number of cond_ID.
            If separate is False, ax is a matplotlib.axes.Axes object
        """

        if module == 'matplotlib':
            plt.rcParams['font.family'] = ['Arial']
            plt.ion
        elif module == 'plotly':
            from plotly.validators.scatter.marker import SymbolValidator
            from plotly.subplots import make_subplots

        event_cols = [
            event_col for event_col in self.data.columns if '_trial_time' in event_col]

        
        if not figsize:
            cm = 1/2.54  # centimeters in inches
            figsize = (21.0*cm, 29.7*cm)

        # easily done by set().intersection
        def intersection(lst1, lst2):
            lst3 = [value for value in lst1 if value in lst2]
            return lst3

        if keys is not None:

            event_cols = intersection(event_cols, keys)

            if not event_cols:
                print('No keys were found in self.data')
                return None

        if conditions_IDs is None:
            conditions_IDs = [i for i, _ in enumerate(self.conditions)]

            # detect conditions_IDs present in the dataset and remove the conditions if
            # the condition_ID is not present in the dataset
            present_cond_IDs = set(self.metadata_df[self.metadata_df.keep == True].condition_ID.values)
            new_cond_ID_list = conditions_IDs.copy()
            for condition_ID in conditions_IDs:
                if condition_ID not in present_cond_IDs:
                    new_cond_ID_list.remove(condition_ID)
            conditions_IDs = new_cond_ID_list

        if self.cond_aliases is None:
            present_cond_IDs = list(set(self.metadata_df[self.metadata_df.keep == True].condition_ID.values))
            self.cond_aliases = [set(self.metadata_df[(self.metadata_df.keep == True) & (self.metadata_df.condition_ID == present_cond_IDs[present_cond_idx])].trigger.values) for present_cond_idx, _ in enumerate(present_cond_IDs)]
        
        if self.time_unit == 's' or self.time_unit == 'seconds': 
            trial_window_s = [ float(x) for x in self.trial_window]
        elif self.time_unit == 'ms' or self.time_unit == 'milliseconds': 
            trial_window_s = [ float(x)/1000 for x in self.trial_window]

        if separate:

            if target is None:

                if module == 'matplotlib':
                    fig, ax = plt.subplots(len(event_cols), len(conditions_IDs),
                        sharex=True, sharey=False, figsize=figsize)

                    if len(conditions_IDs) == 1:
                        ax = ax[:, np.newaxis]

                    if len(event_cols) == 1:
                        ax = ax[np.newaxis, :]
                        
                elif module == 'plotly':
                    fig = make_subplots(
                        rows= len(event_cols), 
                        cols= len(conditions_IDs), 
                        shared_xaxes= True,
                        horizontal_spacing = 0.05,
                        vertical_spacing = 0.05,
                        subplot_titles=tuple(
                            self.cond_aliases + [""] * (len(event_cols) - 1) * len(conditions_IDs))
                        )
                    fig.update_layout(
                        autosize=False,
                        width = 720,
                        height = 960
                    )

            else:
                if module == 'matplotlib':
                    assert isinstance(ax, Axes)
                    assert len(event_cols) == ax.shape[0]
                    assert len(conditions_IDs) == ax.shape[1]
                elif module == 'plotly':
                    fig = make_subplots(
                        rows=len(event_cols),
                        cols=len(conditions_IDs),
                        shared_xaxes=True,
                        horizontal_spacing = 0.05,
                        vertical_spacing = 0.05,
                        subplot_titles=tuple(
                            self.cond_aliases + [""] * (len(event_cols) - 1) * len(conditions_IDs))
                        )
                    fig.update_layout(
                        autosize=False,
                        width = 720,
                        height = 960
                    )

        else:
            if target is None:
                if module == 'matplotlib':
                    cm = 1/2.54  # centimeters in inches
                    fig, ax = plt.subplots(1, len(conditions_IDs), figsize=(21.0*cm, 29.7*cm))

                    for axi in ax:
                        box = axi.get_position()
                        axi.set_position([box.x0, box.y0 + box.height * 0.15,
                            box.width, box.height * 0.85])
                elif module == 'plotly':
                    ...
                    #TODO
                
            else:
                if module == 'matplotlib':
                    assert isinstance(ax, Axes)
                    assert ax.shape[0] == 1
                    assert len(conditions_IDs) == ax.shape[1]
                elif module == 'plotly':
                    ...
                    #TODO              

            if module == 'matplotlib':
                ax = ax.reshape(1,len(ax))

        if colors == 'default':
            if separate:
                colors = ['k'] * len(event_cols)
            else:
                if module == 'matplotlib':
                    colors = [ 'C' + str(i)  for i, _ in enumerate(event_cols)]
                elif module == 'plotly':
                    ...
                    #TODO
        else:
            assert isinstance(colors, list)
            assert len(colors) == len(event_cols)


        for cond_idx, condition_ID in enumerate(conditions_IDs):


            df_subset = self.data.loc[(self.metadata_df['condition_ID'] == condition_ID) & (
                self.metadata_df['keep']), :]  # only include keep

            df_subset = df_subset.reset_index()

            L = [None] * len(event_cols) # placeholder for the first line of each event_col

            for ev_idx_, event_col in enumerate(event_cols):
                if separate:
                    ev_idx = ev_idx_ # you cannot override the iterator
                else:
                    ev_idx = 0

                X = [None] * df_subset.shape[0]
                Y = [None] * df_subset.shape[0]
                for r in range(0, df_subset.shape[0]):

                    ev_times = df_subset.at[r, event_col]

                    X_ = np.array(ev_times)
                    X_.shape = (1, len(X_))
                    X[r] = np.tile(X_, (2, 1))/1000  # ms

                    Y_ = np.array([r, r+1])
                    Y_.shape = (2, 1)
                    Y[r] = np.tile(Y_, (1, X_.shape[1]))

                    # plot_names = trigger + ' ' + event_col

                    event_name_stem = event_col.split('_trial_time')[0]

                # after for r

                X = np.concatenate(X, axis=1)
                Y = np.concatenate(Y, axis=1)

                if module == 'matplotlib':
                    L_ = ax[ev_idx][cond_idx].plot(
                        X, Y, '-', color=colors[ev_idx_], linewidth=0.5, label = event_col)

                    if (L[ev_idx_] is None) & (bool(L_)):
                        L[ev_idx_] = L_[0]
                    
                    ax[ev_idx][cond_idx].set_ylim(0,df_subset.shape[0] )
                    ax[ev_idx][cond_idx].set_xlim(trial_window_s)

                    if separate:
                        ax[ev_idx][cond_idx].set_ylabel('Trials: ' + event_name_stem)
                    else:
                        ax[ev_idx][cond_idx].set_ylabel('Trials')

                    ax[ev_idx][cond_idx].spines['top'].set_visible(False)
                    ax[ev_idx][cond_idx].spines['right'].set_visible(False)
                elif module == 'plotly':
                    #see https://plotly.com/python/line-charts/#connect-data-gaps
                    # convert 2 x n array to 1 x 3n vector with NaN as a separater 

                    X_ = np.zeros((X.shape[1]*3))
                    X_[:] = np.nan
                    Y_ = np.zeros((Y.shape[1]*3))
                    Y_[:] = np.nan

                    for c in range(0, X.shape[1]):
                        X_[c*3:c*3+2] = np.transpose(X[:,c])
                        Y_[c*3:c*3+2] = np.transpose(Y[:,c])

                    fig.add_trace(
                        go.Scatter(
                            x=X_,
                            y=Y_,
                            name=event_col,
                            mode='lines',
                            connectgaps=False
                        ), row= ev_idx+1, col = condition_ID+1)

                    fig.update_yaxes(
                        ticks="outside",
                        ticklen=5,
                        tickwidth=1,   
                        tickfont_size=12,
                        showline=True,
                        linecolor='black',
                        showticklabels =False
                        )

                    fig.update_yaxes(
                        title_text = 'Trials',
                        showticklabels = True,
                        col=1
                        )                        

                if module == 'matplotlib':
                    ax[0][cond_idx].set_title(self.cond_aliases[cond_idx])

                    ax[ev_idx][cond_idx].set_xlabel('Time (s)')

                    if not separate:
                        ax[ev_idx][cond_idx].legend(handles = [ln for ln in L if ln is not None], 
                            bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                            mode='expand', ncol=1)
                elif module == 'plotly':
                    
                    fig.update_xaxes(type='linear')
                    fig.update_yaxes(type='linear')

                    fig.update_xaxes(
                        ticks="outside",
                        ticklen=5,
                        tickwidth=1,
                        tickfont_size=12,
                        # range=[session.trial_window[0]/1000, session.trial_window[1]/1000],
                        showline=True,
                        linecolor='black',
                        range=trial_window_s
                        )


        if module == 'plotly':
            fig.update_xaxes(
                title_text = 'Time (s)',
                row=ev_idx+1
                )
            fig.show()

        return fig

# TODO: store into helper files
def histo_only(x: np.array, trial_window: list, bin_size: int):
    histo, _ = np.histogram(x,
        bins = range(trial_window[0], trial_window[1]+1, bin_size))
    return histo

def load_sktime_dataset(fullpath: str = None):
    """
    Load dataset previously prepared for timeseries ML with sktime
    Returns
    _______
    X : DataFrame
        panel format of timeseries for sktime dependant variables
    y: np.array 
        array of strings corresponding to the condition aliases
    """

    with open(fullpath, 'rb') as file:
        X_y = pickle.load(file)     

    return X_y[0], X_y[1]

