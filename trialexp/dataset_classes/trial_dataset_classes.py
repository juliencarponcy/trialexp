# Python classes for storing, plotting and transform data by trials
import sys
from datetime import datetime
from typing import Iterable, Union, Optional, Tuple
import os
import pickle
from re import search
import warnings

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import cm

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

    def __init__(self, data, metadata_df: pd.DataFrame):
        self.data = data
        self.metadata_df = metadata_df
        self.metadata_df['keep'] = True
        self.metadata_df['trial_ID'] = self.metadata_df.index.values
        self.creation_date = datetime.now()
        
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
        self.trial_window = trial_window
        self.time_unit = unit

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
        '''
        Save Trials_Dataset to pickle file (.pkl)
        '''
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
        '''
        exclude one or several conditions of the dataset
        '''
        if isinstance(condition_IDs_to_exclude, int):
            condition_IDs_to_exclude = [condition_IDs_to_exclude]
        filter = self.metadata_df['condition_ID'].apply(lambda x: x in condition_IDs_to_exclude)
        self.metadata_df.loc[filter,'keep'] = False

    def filterout_groups(self, group_IDs_to_exclude: list):
        '''
        exclude one or several groups of the dataset
        '''
        if isinstance(group_IDs_to_exclude, int):
            group_IDs_to_exclude = [group_IDs_to_exclude]
        filter = self.metadata_df['group_ID'].apply(lambda x: x in group_IDs_to_exclude)
        self.metadata_df.loc[filter,'keep'] = False

    def filterout_subjects(self, subject_IDs_to_exclude: list):
        '''
        exclude one or several subjects of the dataset
        '''
        if isinstance(subject_IDs_to_exclude, int):
            subject_IDs_to_exclude = [subject_IDs_to_exclude]
        filter = self.metadata_df['subject_ID'].apply(lambda x: x in subject_IDs_to_exclude)
        self.metadata_df.loc[filter,'keep'] = False

    def filter_min(self, min_trials : int = 5):
        '''
        filter subjects who do not have sufficient trials in a condition,
        NB: this is not removing the trials of this subject in the other
        conditions! 
        If you want to exclude animals which have less than x trials in
        a condition from the full dataset, use sequentially:

        <trial_dataset>.filter_min(min_trials = x)
        <trial_dataset>.filterout_if_not_in_all_cond()
        '''

        nb_trials = self.metadata_df.groupby(['condition_ID', 'group_ID','subject_ID']).agg(len)['trial_ID']
        trials_idx = self.metadata_df.groupby(['condition_ID', 'group_ID','subject_ID']).agg(list)['trial_ID']
        discarded = nb_trials[nb_trials < min_trials].index
        if len(discarded) > 0:
            trials_idx[discarded]
            idx_filter = np.concatenate(trials_idx[discarded].values)
            self.metadata_df.loc[idx_filter,'keep'] = False

    def filterout_if_not_in_all_cond(self):
        '''
        To remove subjects who do not have
        trials in all the conditions.
        Can be used after <trial_dataset>.filter_min()
        '''
        

        filtered_df = self.metadata_df[self.metadata_df['keep'] == True]
        pre_gby = filtered_df.groupby(['condition_ID','subject_ID'])
        pre_gby = pre_gby.aggregate({'trial_ID' : np.count_nonzero})
        
        nb_cond = pre_gby.index.get_level_values('condition_ID').nunique()

        to_remove = pre_gby.reset_index(level=0).index.value_counts() < nb_cond

        to_remove = to_remove.index[to_remove.values]

        idx_to_remove = filtered_df['subject_ID'].isin(to_remove) == True
        idx_to_remove = idx_to_remove[idx_to_remove == True]

        self.metadata_df.loc[idx_to_remove.index,'keep'] = False

    def filter_reset(self):
        '''
        reset filters to include all trials as
        at the creation of the dataset
        '''

        self.metadata_df['keep'] = True

class Continuous_Dataset(Trials_Dataset):

    def __init__(self, data: np.ndarray, metadata_df: pd.DataFrame, colnames_dict: dict):
        super().__init__(data, metadata_df)
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
        self.trial_window = trial_window
        self.time_unit = unit
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
        '''
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
            
            cols_idx = self.colnames_dict[vars_to_export]
            cols_names = [vars_to_export]

        else:
            raise ValueError('vars_to_export appears not correct')

        if len(cols_idx) > 1:

            data = self.data[:,cols_idx,:]

            X = convert.from_3d_numpy_to_nested(
                data,
                column_names = cols_names,
                cells_as_numpy=False)
        else:

            data = self.data[:,cols_idx,:].squeeze

            X = convert.from_2d_array_to_nested(
                data, 
                index = self.metadata_df.index.values,
                time_index = self.time_vector,
                cells_as_numpy = False)
            
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

    def transform_variables(self, on_vars: VarsType, function: callable, deriv_name: str):
        ...

    # TODO: Won't probably adapt for only one condition (axes will not be np.arrays)
    def lineplot(
            self,
            vars: VarsType = 'all',
            time_lim: Optional[list] = None,
            time_unit: str = None,
            error: str = None,
            is_x_vs_y: bool = False, # implement here or not?
            plot_subjects: bool = True,
            plot_groups: bool = True,
            ylim: list = None, 
            colormap: str = 'jet',
            figsize: tuple = (20, 10),
            dpi: int = 100,
            legend: bool = True,
            verbose: bool = False):

        plt.rcParams["figure.dpi"] = dpi
        
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
            vars_idx = [self.colnames_dict[var] for var in vars]


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
        if len(group_IDs) == 1:
            group_colors = cm.get_cmap(colormap, len(self.conditions))
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
        
        
        for c_idx, cond_ID in enumerate(condition_IDs):

            # Set title as condition on the first line
            if len(group_IDs) == 1:
                if plot_groups and not plot_subjects:
                    ...

                elif hasattr(self, 'cond_aliases'):
                    axs[0, c_idx+1].set_title(str(self.cond_aliases[cond_ID]))
                else:
                    axs[0, c_idx+1].set_title(str(self.conditions[cond_ID]))
            else:
                if hasattr(self, 'cond_aliases'):
                    axs[0, c_idx].set_title(str(self.cond_aliases[cond_ID]))
                else:
                    axs[0, c_idx].set_title(str(self.conditions[cond_ID]))

            # Compute means and sems
            for g_idx, group_ID in enumerate(group_IDs):
                subj_subset = gby.loc[cond_ID, group_ID ,:].index.values
                # used to work using the following, possibly previous for behaviour only?
                # subj_subset = gby.loc[cond_ID, group_ID ,:].index.get_level_values(2).values

                for subj_idx, subject in enumerate(subj_subset):
                    if verbose:
                        print(f'cond_ID: {cond_ID}, group_idx {group_ID}, subj {subject}')

                    trial_idx = gby.loc[(cond_ID, group_ID, subject),'trial_ID']
                    np_val = self.data[trial_idx,:,:]
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        mean_subj = np.nanmean(np_val,axis=0)
                    
                    mean_subj = mean_subj[vars_idx, :]
                    # sem_subj = np_val.std(axis=0) / np.sqrt(len(trial_idx)-1)
                    # sem_subj = sem_subj[vars_idx, :]
                    
                    # Plot
                    if plot_subjects:
                        for ax_idx, var in enumerate(vars_idx):
                            if len(group_IDs) == 1:
                                axs[ax_idx, c_idx+1].plot(
                                    time_vec, mean_subj[ax_idx, :],
                                    alpha = 0.7,
                                    label = f'{subject} #{len(trial_idx)}',
                                    color = subj_colors(subj_dict[subject]))
                            else:
                                axs[ax_idx, c_idx].plot(
                                    time_vec, mean_subj[ax_idx, :],
                                    alpha = 0.7,
                                    label = f'{subject} #{len(trial_idx)}',
                                    color = subj_colors(subj_dict[subject]))

                    if subj_idx == 0:
                        
                        mean_group = mean_subj
                  
                    elif subj_idx == 1:
                        mean_group = np.stack([mean_group, mean_subj], axis=0)
                    else:
                        mean_group = np.concatenate([mean_group, np.expand_dims(mean_subj, axis=0)], axis=0)
                
                # Group computations
                if plot_groups:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        sem_group = np.nanstd(mean_group, axis=0) / np.sqrt(mean_group.shape[0]-1)
                        mean_group = np.nanmean(mean_group, axis=0)

                    # Group plotting    
                    for ax_idx, var_idx in enumerate(vars_idx):
                        if len(group_IDs) == 1:
                            if g_idx == 0 and c_idx == 0:
                                axs[ax_idx, 0].set_ylabel(vars[ax_idx])
                            # if a group is more than a single subject
                            if len(mean_group.shape) > 1:
                                axs[ax_idx, 0].plot(time_vec, mean_group[ax_idx, :], lw=3,
                                    color = group_colors(cond_ID),
                                    label = self.cond_aliases[cond_ID])                    
                                
                                if error is not None:
                                    # fill sem
                                    axs[ax_idx, 0].fill_between(time_vec, 
                                        mean_group[ax_idx, :] - sem_group[ax_idx, :],
                                        mean_group[ax_idx, :] + sem_group[ax_idx, :],
                                        alpha=0.2, color=group_colors(cond_ID), lw=0)

                            # if single subject in the group
                            else:
                                axs[ax_idx, 0].plot(time_vec, mean_group, lw=3,
                                color = group_colors(cond_ID),
                                label = self.cond_aliases[cond_ID])     

                        else:
                            if g_idx == 0 and c_idx == 0:
                                axs[ax_idx, 0].set_ylabel(vars[ax_idx])
                            
                            # if a group is more than a single subject
                            if len(mean_group.shape) > 1:    
                                # plot mean
                                axs[ax_idx, c_idx].plot(time_vec, mean_group[ax_idx, :],
                                    lw=3, color=group_colors(group_ID),
                                    label = group_ID)                    
                                
                                if error is not None:
                                    # fill sem
                                    axs[ax_idx, c_idx].fill_between(time_vec, 
                                        mean_group[ax_idx, :] - sem_group[ax_idx, :],
                                        mean_group[ax_idx, :] + sem_group[ax_idx, :],
                                        alpha=0.3, color=group_colors(group_ID), lw=0)  
                            # if single subject in the group
                            else:
                                axs[ax_idx, c_idx].plot(time_vec, mean_group,
                                    lw=3, color=group_colors(group_ID),
                                    label = group_ID)    

                        if ax_idx == len(vars_idx)-1:
                            if plot_groups and not plot_subjects:
                                ...
                            else:
                                axs[ax_idx, c_idx].set_xlabel(time_unit)
                            if len(self.groups) == 1:
                                if plot_groups and not plot_subjects:
                                    [ax[0].set_xlabel(time_unit) for ax in axs]
                                else:
                                    axs[ax_idx, c_idx+1].set_xlabel(time_unit)
        
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
        
        plt.show()

    def scatterplot(self, vars: VarsType, groupby: Optional[list] = ['group_ID', 'subject_ID'], \
            timelim: Optional[list] = None):
        ...




class Event_Dataset(Trials_Dataset):

    def __init__(self, data: pd.DataFrame, metadata_df: pd.DataFrame):
            super().__init__(data, metadata_df)
            # TODO: Consider inputing colnames only as list or tuple
            # and compute the dictionary '<names>': <idx_col(int)> in __init__

    def raster(self, kvargs):
        ...

    def compute_distribution(
            self,
            trial_window: Iterable = None,
            bin_size: int = 100, # by default in ms, not adapted yet for seconds
            normalize: bool = True, # normalize the count of events according to bin_size
            per_session: bool = False, # if false, compute distribution per animal for all its sessions
            out_as_continuous: bool = False, # if true, output a continuous dataset
            verbose: bool = False):
        '''
        Compute distribution of events for each session.
        Output a continuous_dataset instance if out_as_continuous = True

        '''

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
        
        # Merged behavioural data and conditions DataFrames
        full_df = pd.concat([filtered_ev_df, filtered_meta_df], axis=1)
        
        # Extract behavioural times columns names and create new ones for the distribution
        ev_times_cols = [col for col in self.data.columns if search('trial_time', col)]
        dist_col_names = [col.split('_trial_time',1)[0] + '_dist' for col in ev_times_cols]
        
        # define column and function for named aggregate
        func_iter = [np.hstack for i in range(len(ev_times_cols))]
        agg_dict = dict(zip(ev_times_cols, func_iter))
        agg_dict['trial_ID'] = len

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
                lambda x: histo_only(x, self.trial_window, bin_size))

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

# TODO: store into helper files
def histo_only(x: np.array, trial_window: list, bin_size: int):
    histo, _ = np.histogram(x,
        bins = range(trial_window[0], trial_window[1]+1, bin_size))
    return histo
