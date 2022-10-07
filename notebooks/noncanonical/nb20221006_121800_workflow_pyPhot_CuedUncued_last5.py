#!/usr/bin/env python
# coding: utf-8

# ## Workflow to analyze Photometry data
# 
# ```bash
# jupyter nbconvert "D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical\nb20221006_121800_workflow_pyPhot_CuedUncued_last5.ipynb" --to="python" --output-dir="D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical" --output="nb20221006_121800_workflow_pyPhot_CuedUncued_last5"
# ```

# ### Imports

# In[1]:


# allow for automatic reloading of classes and function when updating the code
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Import Session and Experiment class with helper functions
from trialexp.process.data_import import *


# ### Variables

# In[2]:


import pandas as pd

trial_window = [-2000, 6000] # in ms

# time limit around trigger to perform an event
# determine successful trials
timelim = [0, 2000] # in ms

# Digital channel nb of the pyphotometry device
# on which rsync signal is sent (from pycontrol device)
rsync_chan = 2

basefolder, _ = os.path.split(os.path.split(os.getcwd())[0])

# These must be absolute paths
# use this to use within package tasks files (in params)
tasksfile = os.path.join(basefolder,'params\\tasks_params.csv')
# use this to put a local full path
#tasksfile = -r'C:/.../tasks_params.csv' 

# photometry_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\test_folder\photometry'
photometry_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\kms_pyphotometry'
video_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\videos'


# ### Tasks
# - A tasks definition file (.csv) contains all the information to perform the extractions of behaviorally relevant information from **PyControl** files, for each **task** file. It includes what are the **triggers** of different trial types, what **events** to extract (with time data), and what are events or printed lines that could be relevant to determine the **conditions** (e.g: free reward, optogenetic stimulation type, etc.)
# - To analyze a new task you need to append task characteristics like **task** filename, **triggers**, **events** and **conditions**

# In[3]:


tasks = pd.read_csv(tasksfile, usecols = [1,2,3,4], index_col = False)
tasks


# ### Optional
# 
# Transfer Files from hierarchical folders by tasks to flat folders, for photometry and behaviour files
# 
# 2m 13.9s
# 
# If we obtain list of files in source and dest at first and then only perform comparison on them,
# This should be much faster

# In[4]:


photo_root_dir = 'T:\\Data\\head-fixed\\pyphotometry\\data'
pycontrol_root_dir = 'T:\\Data\\head-fixed\\pycontrol'

root_folders = [photo_root_dir, pycontrol_root_dir]
horizontal_folder_pycontrol = 'T:\\Data\\head-fixed\\test_folder\\pycontrol'
horizontal_folder_photometry = 'T:\\Data\\head-fixed\\test_folder\\photometry'

copy_files_to_horizontal_folders(root_folders, horizontal_folder_pycontrol, horizontal_folder_photometry)


# ### Create an experiment object
# 
# This will include all the pycontrol files present in the folder_path directory (do not include subdirectories)

# In[5]:


# Folder of a full experimental batch, all animals included

# Enter absolute path like this
# pycontrol_files_path = r'T:\Data\head-fixed\test_folder\pycontrol'

# or this if you want to use data from the sample_data folder within the package
pycontrol_files_path = os.path.join(basefolder,'sample_data/pycontrol')
pycontrol_files_path = r'T:\Data\head-fixed\kms_pycontrol'

# Load all raw text sessions in the indicated folder or a sessions.pkl file
# if already existing in folder_path
exp_cohort = Experiment(pycontrol_files_path)

# Only use if the Experiment cohort as been processed by trials before
# TODO: assess whether this can be removed or not
exp_cohort.by_trial = True


# ### Perform extraction of behavioural information by trial
# 
# 5m55.4s

# In[6]:


# Process the whole experimental folder by trials
exp_cohort.process_exp_by_trial(trial_window, timelim, tasksfile, blank_spurious_event='spout', blank_timelim=[0, 65])

# Save the file as sessions.pkl in folder_path
# exp_cohort.save() # Do I need to save this???


# ### Match with photometry, videos, and DeepLabCut files
# 
# The following Warning : 
# 
# KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads...
# 
# is due to rsync function for photometry-pycontrol alignment
# 
# 2m10.9s
# 

# In[7]:


# Find if there is a matching photometry file and if it can be used:
# rsync synchronization pulses matching between behaviour and photometry
from copy import deepcopy

exp_cohort.match_to_photometry_files(photometry_dir, rsync_chan, verbose=False)

# Find matching videos
exp_cohort.match_sessions_to_files(video_dir, ext='mp4')

# FInd matching DeepLabCut outputs files
exp_cohort.match_sessions_to_files(video_dir, ext='h5')

exp_cohort.save()

exp_cohort_copy = deepcopy(exp_cohort)


# ### Define conditions and groups for extraction

# Example in progress for Cued-Uncued

# In[8]:


# List of uncued conditions as listed on the tasks .csv file for task reaching_go_spout_cued_uncued:
# free_reward_timer; reward spout cued; reward bar cued; reward bar_off; reward spout uncued; reward bar uncued; reward free; reward free_uncued

# Many combinations possible
conditions_dict0 = {'trigger': 'cued', 'valid': True, 'reward spout cued': True, 'free_reward_timer': False, 'success': True}
conditions_dict1 = {'trigger': 'cued', 'valid': True, 'reward bar cued': True, 'free_reward_timer': False, 'success': True}
conditions_dict2 = {'trigger': 'cued', 'valid': True, 'reward free': True, 'success': True}
conditions_dict3 = {'trigger': 'cued', 'valid': True, 'success': False}
conditions_dict4 = {'trigger': 'uncued', 'valid': True, 'reward spout uncued': True, 'free_reward_timer': False, 'success': True}
conditions_dict5 = {'trigger': 'uncued', 'valid': True, 'reward bar uncued': True, 'free_reward_timer': False, 'success': True}
conditions_dict6 = {'trigger': 'uncued', 'valid': True, 'reward free_uncued': True} # reward after [20, 30] s

# Aggregate all condition dictionaries in a list
condition_list = [conditions_dict0, conditions_dict1, conditions_dict2, conditions_dict3,                   conditions_dict4, conditions_dict5, conditions_dict6]
# Aliases for conditions
cond_aliases = [
    'Cued, reward at spout, hit', 
    'Cued, reward at bar release, hit', 
    'Cued, Pavlovian, hit', 
    'Cued, miss', \
    'Uncued, reward at spout, hit', 
    'Uncued, reward at bar release, hit',
    'Uncued, miss']

# Groups as a list of lists
groups = None

# right_handed = [281]
# groups = [[280, 282, 299, 300, 301],\
#     [284, 285, 296, 297, 306, 307]]
# Window to exctract (in ms)


# ### Extract Photometry trials and create a Continuous_Dataset

# Example data filtering, only needed if you want to separate days or aninals or else
# 
# https://github.com/ThomasAkam/photometry_preprocessing/blob/master/Photometry%20data%20preprocessing.ipynb

# # Cued, Cue onset, last five sessions
# 
# - Still this contains a lot of sessions with bad performance
# - How to narrow this down to good performance only?
# - **Need to combine this analysis with pyControl analysis**
# 

# In[9]:



exp_cohort = deepcopy(exp_cohort_copy)  # copy back to recover

exp_cohort.sessions = [session for session in exp_cohort.sessions
                       if (session.subject_ID in [47, 48, 49, 51, 53]) and (session.number > 2) 
                       and (session.task_name == 'reaching_go_spout_cued_uncued')]

cont_dataset = exp_cohort.get_photometry_groups(
    groups=None,  # or use groups variable defined above
    conditions_list=condition_list,
    cond_aliases=cond_aliases,
    when='all',
    task_names='reaching_go_spout_cued_uncued',  # 'reaching_go_nogo',
    # align to the first event of a kind e.g. None (meaning CS_Go onset), 'bar_off', 'spout'
    trig_on_ev=None,
    high_pass=None,  # analog_1_df_over_f doesn't work with this
    low_pass=45,
    median_filt=3,
    motion_corr=True,
    df_over_f=True,
    downsampling_factor=10,
    export_vars=['analog_1', 'analog_1_filt', 'analog_2',
                 'analog_2_filt', 'analog_1_df_over_f'],
    verbose=False)




cont_dataset.filterout_conditions([1, 2, 4, 5, 6]) # Cued 0 hit and 3 miss
cont_dataset.filter_lastNsessions(5)
cont_dataset.set_trial_window([t/1000 for t in trial_window], 's')


fig, axs, df1 = cont_dataset.lineplot(
    vars=['analog_1_df_over_f'],
    time_lim=[-2, 2],
    time_unit='s',
    # [[-0.1, 0.6]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    ylim=None,
    error=True,
    colormap=cmap10(),
    legend=True,
    plot_subjects=True,
    plot_groups=True,
    figsize=(25, 5),
    dpi=200,
    verbose=True)

for r in range(axs.shape[0]):
    if len(axs.shape) > 1:
        for c in range(axs.shape[1]):
            axs[r, c].set_xlabel('Relative to Cue onset (s)', fontsize=14)
            axs[r, c].set_title(axs[r, c].get_title('center'), fontsize=14)

    else:
        axs[r].set_xlabel('Relative to Cue onset (s)', fontsize=14)

axs[0, 0].set_ylabel('\u0394F/F', fontsize=14)

# Return a count of overall number of trials
cont_dataset.metadata_df['keep'].value_counts()

df1


# # Cued, spout touch, last five sessions
# 
# - Still this contains a lot of sessions with bad performance
# - How to narrow this down to good performance only?
# - **Need to combine this analysis with pyControl analysis**

# In[10]:



exp_cohort = deepcopy(exp_cohort_copy)  # copy back to recover

exp_cohort.sessions = [session for session in exp_cohort.sessions
                       if (session.subject_ID in [47, 48, 49, 51, 53]) and (session.number > 2)
                       and (session.task_name == 'reaching_go_spout_cued_uncued')]

cont_dataset = exp_cohort.get_photometry_groups(
    groups=None,  # or use groups variable defined above
    conditions_list=condition_list,
    cond_aliases=cond_aliases,
    when='all',
    task_names='reaching_go_spout_cued_uncued',  # 'reaching_go_nogo',
    # align to the first event of a kind e.g. None (meaning CS_Go onset), 'bar_off', 'spout'
    trig_on_ev='spout',
    high_pass=None,  # analog_1_df_over_f doesn't work with this
    low_pass=45,
    median_filt=3,
    motion_corr=True,
    df_over_f=True,
    downsampling_factor=10,
    export_vars=['analog_1', 'analog_1_filt', 'analog_2',
                 'analog_2_filt', 'analog_1_df_over_f'],
    verbose=False)

cont_dataset.filterout_conditions([1, 2, 3, 4, 5, 6])  # Cued 0 hit 
cont_dataset.filter_lastNsessions(5)

cont_dataset.set_trial_window([t/1000 for t in trial_window], 's')


fig, axs, df2 = cont_dataset.lineplot(
    vars=['analog_1_df_over_f'],
    time_lim=[-2, 2],
    time_unit='s',
    # [[-0.1, 0.6]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    ylim=None,
    error=True,
    colormap=cmap10(),
    legend=True,
    plot_subjects=True,
    plot_groups=True,
    figsize=(25, 5),
    dpi=200,
    verbose=True)

for r in range(axs.shape[0]):
    if len(axs.shape) > 1:
        for c in range(axs.shape[1]):
            axs[r, c].set_xlabel('Relative to spout touch (s)', fontsize=14)
            axs[r, c].set_title(axs[r, c].get_title('center'), fontsize=14)

    else:
        axs[r].set_xlabel('Relative to spout touch (s)', fontsize=14)

axs[0, 0].set_ylabel('\u0394F/F', fontsize=14)

# Return a count of overall number of trials
cont_dataset.metadata_df['keep'].value_counts()

df2


# # Cued, bar release, last five sessions
# 
# - Still this contains a lot of sessions with bad performance
# - How to narrow this down to good performance only?df
# - **Need to combine this analysis with pyControl analysis**

# In[11]:



exp_cohort = deepcopy(exp_cohort_copy)  # copy back to recover

exp_cohort.sessions = [session for session in exp_cohort.sessions
                       if (session.subject_ID in [47, 48, 49, 51, 53]) and (session.number > 2)
                       and (session.task_name == 'reaching_go_spout_cued_uncued')]

cont_dataset = exp_cohort.get_photometry_groups(
    groups=None,  # or use groups variable defined above
    conditions_list=condition_list,
    cond_aliases=cond_aliases,
    when='all',
    task_names='reaching_go_spout_cued_uncued',  # 'reaching_go_nogo',
    # align to the first event of a kind e.g. None (meaning CS_Go onset), 'bar_off', 'spout'
    trig_on_ev='bar_off',
    high_pass=None,  # analog_1_df_over_f doesn't work with this
    low_pass=45,
    median_filt=3,
    motion_corr=True,
    df_over_f=True,
    downsampling_factor=10,
    export_vars=['analog_1', 'analog_1_filt', 'analog_2',
                 'analog_2_filt', 'analog_1_df_over_f'],
    verbose=False)


cont_dataset.filterout_conditions([1,2,3,4,5,6])  # Cued only
cont_dataset.filter_lastNsessions(5)

cont_dataset.set_trial_window([t/1000 for t in trial_window], 's')

fig, axs, df4 = cont_dataset.lineplot(
    vars=['analog_1_df_over_f'],
    time_lim=[-2, 2],
    time_unit='s',
    # [[-0.1, 0.6]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    ylim=None,
    error=True,
    colormap=cmap10(),
    legend=True,
    plot_subjects=True,
    plot_groups=True,
    figsize=(25, 5),
    dpi=200,
    verbose=True)

for r in range(axs.shape[0]):
    if len(axs.shape) > 1:
        for c in range(axs.shape[1]):
            axs[r, c].set_xlabel('Relative to bar release (s)', fontsize=14)
            axs[r, c].set_title(axs[r, c].get_title('center'), fontsize=14)

    else:
        axs[r].set_xlabel('Relative to bar release (s)', fontsize=14)

axs[0, 0].set_ylabel('\u0394F/F', fontsize=14)

# Return a count of overall number of trials
cont_dataset.metadata_df['keep'].value_counts()

df4


# # Uncued, spout touch, last five sessions
# 
# - Still this contains a lot of sessions with bad performance
# - How to narrow this down to good performance only?
# - **Need to combine this analysis with pyControl analysis**

# In[12]:




exp_cohort = deepcopy(exp_cohort_copy)  # copy back to recover

exp_cohort.sessions = [session for session in exp_cohort.sessions
                       if (session.subject_ID in [47, 48, 49, 51, 53]) and (session.number > 2)
                       and (session.task_name == 'reaching_go_spout_cued_uncued')]

cont_dataset = exp_cohort.get_photometry_groups(
    groups=None,  # or use groups variable defined above
    conditions_list=condition_list,
    cond_aliases=cond_aliases,
    when='all',
    task_names='reaching_go_spout_cued_uncued',  # 'reaching_go_nogo',
    # align to the first event of a kind e.g. None (meaning CS_Go onset), 'bar_off', 'spout'
    trig_on_ev='spout',
    high_pass=None,  # analog_1_df_over_f doesn't work with this
    low_pass=45,
    median_filt=3,
    motion_corr=True,
    df_over_f=True,
    downsampling_factor=10,
    export_vars=['analog_1', 'analog_1_filt', 'analog_2',
                 'analog_2_filt', 'analog_1_df_over_f'],
    verbose=False)


cont_dataset.filterout_conditions([0, 1, 2, 3, 5, 6])  # Uncued 4
cont_dataset.filter_lastNsessions(5)

cont_dataset.set_trial_window([t/1000 for t in trial_window], 's')


fig, axs, df3 = cont_dataset.lineplot(
    vars=['analog_1_df_over_f'],
    time_lim=[-2, 2],
    time_unit='s',
    # [[-0.1, 0.6]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    ylim=None,
    error=True,
    colormap=cmap10(),
    legend=True,
    plot_subjects=True,
    plot_groups=True,
    figsize=(25, 5),
    dpi=200,
    verbose=True)

for r in range(axs.shape[0]):
    if len(axs.shape) > 1:
        for c in range(axs.shape[1]):
            axs[r, c].set_xlabel('Relative to spout touch (s)', fontsize=14)
            axs[r, c].set_title(axs[r, c].get_title('center'), fontsize=14)

    else:
        axs[r].set_xlabel('Relative to spout touch (s)', fontsize=14)

axs[0, 0].set_ylabel('\u0394F/F', fontsize=14)

# Return a count of overall number of trials
cont_dataset.metadata_df['keep'].value_counts()

df3


# # Cued/Uncued, spout

# In[13]:



exp_cohort = deepcopy(exp_cohort_copy)  # copy back to recover

exp_cohort.sessions = [session for session in exp_cohort.sessions
                       if (session.subject_ID in [47, 48, 49, 51, 53]) and (session.number > 2)
                       and (session.task_name == 'reaching_go_spout_cued_uncued')]

cont_dataset = exp_cohort.get_photometry_groups(
    groups=None,  # or use groups variable defined above
    conditions_list=condition_list,
    cond_aliases=cond_aliases,
    when='all',
    task_names='reaching_go_spout_cued_uncued',  # 'reaching_go_nogo',
    # align to the first event of a kind e.g. None (meaning CS_Go onset), 'bar_off', 'spout'
    trig_on_ev='spout',
    high_pass=None,  # analog_1_df_over_f doesn't work with this
    low_pass=45,
    median_filt=3,
    motion_corr=True,
    df_over_f=True,
    downsampling_factor=10,
    export_vars=['analog_1', 'analog_1_filt', 'analog_2',
                 'analog_2_filt', 'analog_1_df_over_f'],
    verbose=False)


cont_dataset.filterout_conditions([1, 2,3, 5, 6])  # Uncued 4
cont_dataset.filter_lastNsessions(5)

cont_dataset.set_trial_window([t/1000 for t in trial_window], 's')


fig, axs, df3 = cont_dataset.lineplot(
    vars=['analog_1_df_over_f'],
    time_lim=[-2, 2],
    time_unit='s',
    # [[-0.1, 0.6]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    ylim=None,
    error=True,
    colormap=cmap10(),
    legend=True,
    plot_subjects=True,
    plot_groups=True,
    figsize=(25, 5),
    dpi=200,
    verbose=True)

for r in range(axs.shape[0]):
    if len(axs.shape) > 1:
        for c in range(axs.shape[1]):
            axs[r, c].set_xlabel('Relative to spout touch (s)', fontsize=14)
            axs[r, c].set_title(axs[r, c].get_title('center'), fontsize=14)

    else:
        axs[r].set_xlabel('Relative to spout touch (s)', fontsize=14)

axs[0, 0].set_ylabel('\u0394F/F', fontsize=14)

# Return a count of overall number of trials
cont_dataset.metadata_df['keep'].value_counts()

df3


# # Uncued, bar release, last five sessions
# 
# - Still this contains a lot of sessions with bad performance
# - How to narrow this down to good performance only?
# - **Need to combine this analysis with pyControl analysis**

# In[14]:



exp_cohort = deepcopy(exp_cohort_copy)  # copy back to recover

exp_cohort.sessions = [session for session in exp_cohort.sessions
                       if (session.subject_ID in [47, 48, 49, 51, 53]) and (session.number > 2)
                       and (session.task_name == 'reaching_go_spout_cued_uncued')]

cont_dataset = exp_cohort.get_photometry_groups(
    groups=None,  # or use groups variable defined above
    conditions_list=condition_list,
    cond_aliases=cond_aliases,
    when='all',
    task_names='reaching_go_spout_cued_uncued',  # 'reaching_go_nogo',
    # align to the first event of a kind e.g. None (meaning CS_Go onset), 'bar_off', 'spout'
    trig_on_ev='bar_off',
    high_pass=None,  # analog_1_df_over_f doesn't work with this
    low_pass=45,
    median_filt=3,
    motion_corr=True,
    df_over_f=True,
    downsampling_factor=10,
    export_vars=['analog_1', 'analog_1_filt', 'analog_2',
                 'analog_2_filt', 'analog_1_df_over_f'],
    verbose=False)

cont_dataset.filterout_conditions([0, 1, 2, 3, 5, 6])  # Uncued 4
cont_dataset.filter_lastNsessions(5)

cont_dataset.set_trial_window([t/1000 for t in trial_window], 's')


fig, axs, df5 = cont_dataset.lineplot(
    vars=['analog_1_df_over_f'],
    time_lim=[-2, 2],
    time_unit='s',
    # [[-0.1, 0.6]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    ylim=None,
    error=True,
    colormap=cmap10(),
    legend=True,
    plot_subjects=True,
    plot_groups=True,
    figsize=(25, 5),
    dpi=200,
    verbose=True)

for r in range(axs.shape[0]):
    if len(axs.shape) > 1:
        for c in range(axs.shape[1]):
            axs[r, c].set_xlabel('Relative to bar release (s)', fontsize=14)
            axs[r, c].set_title(axs[r, c].get_title('center'), fontsize=14)

    else:
        axs[r].set_xlabel('Relative to bar release (s)', fontsize=14)

axs[0, 0].set_ylabel('\u0394F/F', fontsize=14)

# Return a count of overall number of trials
cont_dataset.metadata_df['keep'].value_counts()

df5


# # Cued/Uncued, bar release

# In[15]:



exp_cohort = deepcopy(exp_cohort_copy)  # copy back to recover

exp_cohort.sessions = [session for session in exp_cohort.sessions
                       if (session.subject_ID in [47, 48, 49, 51, 53]) and (session.number > 2)
                       and (session.task_name == 'reaching_go_spout_cued_uncued')]

cont_dataset = exp_cohort.get_photometry_groups(
    groups=None,  # or use groups variable defined above
    conditions_list=condition_list,
    cond_aliases=cond_aliases,
    when='all',
    task_names='reaching_go_spout_cued_uncued',  # 'reaching_go_nogo',
    # align to the first event of a kind e.g. None (meaning CS_Go onset), 'bar_off', 'spout'
    trig_on_ev='bar_off',
    high_pass=None,  # analog_1_df_over_f doesn't work with this
    low_pass=45,
    median_filt=3,
    motion_corr=True,
    df_over_f=True,
    downsampling_factor=10,
    export_vars=['analog_1', 'analog_1_filt', 'analog_2',
                 'analog_2_filt', 'analog_1_df_over_f'],
    verbose=False)

cont_dataset.filterout_conditions([1, 2, 3, 5, 6])  # Uncued 4
cont_dataset.filter_lastNsessions(5)

cont_dataset.set_trial_window([t/1000 for t in trial_window], 's')


fig, axs, df5 = cont_dataset.lineplot(
    vars=['analog_1_df_over_f'],
    time_lim=[-2, 2],
    time_unit='s',
    # [[-0.1, 0.6]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    ylim=None,
    error=True,
    colormap=cmap10(),
    legend=True,
    plot_subjects=True,
    plot_groups=True,
    figsize=(25, 5),
    dpi=200,
    verbose=True)

for r in range(axs.shape[0]):
    if len(axs.shape) > 1:
        for c in range(axs.shape[1]):
            axs[r, c].set_xlabel('Relative to bar release (s)', fontsize=14)
            axs[r, c].set_title(axs[r, c].get_title('center'), fontsize=14)

    else:
        axs[r].set_xlabel('Relative to bar release (s)', fontsize=14)

axs[0, 0].set_ylabel('\u0394F/F', fontsize=14)

# Return a count of overall number of trials
cont_dataset.metadata_df['keep'].value_counts()

df5


# # Cued/Uncued, bar holding

# In[16]:



exp_cohort = deepcopy(exp_cohort_copy)  # copy back to recover

exp_cohort.sessions = [session for session in exp_cohort.sessions
                       if (session.subject_ID in [47, 48, 49, 51, 53]) and (session.number > 2)
                       and (session.task_name == 'reaching_go_spout_cued_uncued')]

cont_dataset = exp_cohort.get_photometry_groups(
    groups=None,  # or use groups variable defined above
    conditions_list=condition_list,
    cond_aliases=cond_aliases,
    when='all',
    task_names='reaching_go_spout_cued_uncued',  # 'reaching_go_nogo',
    # align to the first event of a kind e.g. None (meaning CS_Go onset), 'bar_off', 'spout'
    trig_on_ev=None, # CS_Go = end of hold_start
    high_pass=None,  # analog_1_df_over_f doesn't work with this
    low_pass=45,
    median_filt=3,
    motion_corr=True,
    df_over_f=True,
    downsampling_factor=10,
    export_vars=['analog_1', 'analog_1_filt', 'analog_2',
                 'analog_2_filt', 'analog_1_df_over_f'],
    verbose=False)


cont_dataset.filterout_conditions([1, 2, 5, 6])  # Cued 0 hit and 3 miss
cont_dataset.filter_lastNsessions(5)

cont_dataset.set_trial_window([t/1000 for t in trial_window], 's')


fig, axs, df5 = cont_dataset.lineplot(
    vars=['analog_1_df_over_f'],
    time_lim=[-2, 2],
    time_unit='s',
    # [[-0.1, 0.6]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    ylim=None,
    error=True,
    colormap=cmap10(),
    legend=True,
    plot_subjects=True,
    plot_groups=True,
    figsize=(25, 5),
    dpi=200,
    verbose=True)

for r in range(axs.shape[0]):
    if len(axs.shape) > 1:
        for c in range(axs.shape[1]):
            axs[r, c].set_xlabel('Relative to the end of bar holding (s)', fontsize=14)
            axs[r, c].set_title(axs[r, c].get_title('center'), fontsize=14)

    else:
        axs[r].set_xlabel('Relative to the end of bar holding (s)', fontsize=14)

axs[0, 0].set_ylabel('\u0394F/F', fontsize=14)

# Return a count of overall number of trials
cont_dataset.metadata_df['keep'].value_counts()


# # Uncude, hold start
# 
# data_import.py
# 
# line 1175 error
# 
# get_photometry_groups 1937
# get_photometry_trials 908
# get_trials_times_from_conditions 1175
# 
# 
# self.evettoprocess = ['bar', 'bar_off', 'spout']
# 
# ```python
# elif trig_on_ev not in self.events_to_process:
#             raise Exception('trig_on_ev not in events_to_process')
# ```

# In[17]:




exp_cohort = deepcopy(exp_cohort_copy)  # copy back to recover

exp_cohort.sessions = [session for session in exp_cohort.sessions
                       if (session.subject_ID in [47, 48, 49, 51, 53]) and (session.number > 2)
                       and (session.task_name == 'reaching_go_spout_cued_uncued')]

cont_dataset = exp_cohort.get_photometry_groups(
    groups=None,  # or use groups variable defined above
    conditions_list=condition_list,
    cond_aliases=cond_aliases,
    when='all',
    task_names='reaching_go_spout_cued_uncued',  # 'reaching_go_nogo',
    # align to the first event of a kind e.g. None (meaning CS_Go onset), 'bar_off', 'spout'
    trig_on_ev=None,
    high_pass=None,  # analog_1_df_over_f doesn't work with this
    low_pass=45,
    median_filt=3,
    motion_corr=True,
    df_over_f=True,
    downsampling_factor=10,
    export_vars=['analog_1', 'analog_1_filt', 'analog_2',
                 'analog_2_filt', 'analog_1_df_over_f'],
    verbose=False)

cont_dataset.filterout_conditions([0, 1, 2, 3, 5, 6])  # Uncued 4 only
cont_dataset.filter_lastNsessions(5)

cont_dataset.set_trial_window([t/1000 for t in trial_window], 's')


fig, axs, df5 = cont_dataset.lineplot(
    vars=['analog_1_df_over_f'],
    time_lim=[-2, 2],
    time_unit='s',
    # [[-0.1, 0.6]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    ylim=None,
    error=True,
    colormap=cmap10(),
    legend=True,
    plot_subjects=True,
    plot_groups=True,
    figsize=(25, 5),
    dpi=200,
    verbose=True)

for r in range(axs.shape[0]):
    if len(axs.shape) > 1:
        for c in range(axs.shape[1]):
            axs[r, c].set_xlabel('Relative to the end of bar holding (s)', fontsize=14)
            axs[r, c].set_title(axs[r, c].get_title('center'), fontsize=14)

    else:
        axs[r].set_xlabel('Relative to the end of bar holding (s)', fontsize=14)

axs[0, 0].set_ylabel('\u0394F/F', fontsize=14)

# Return a count of overall number of trials
cont_dataset.metadata_df['keep'].value_counts()

