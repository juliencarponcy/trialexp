#!/usr/bin/env python
# coding: utf-8

# # Delayed tasks analysis
# 

# ### Convert notebook to python
# 

# In[1]:


import os

nb_name = "nb20230222_092800_poster_delayed_tasks_photometry_23apr.ipynb" #TODO change this

basename, ext = os.path.splitext(nb_name)
input_path = os.path.join(os.getcwd(), nb_name)

get_ipython().system('jupyter nbconvert "{input_path}" --to="python" --output="{basename}"')


# Quick analysis of instrumental reaching

# 

# In[2]:


# allow for automatic reloading of classes and function when updating the code
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Import Session and Experiment class with helper functions
from trialexp.process.data_import import *



# ### Variables

# In[3]:


import pandas as pd
from pathlib import Path
trial_window = [-2000, 4000] # in ms

# time limit around trigger to perform an event
# determine successful trials
# timelim = [1000, 4000] # in ms

# Digital channel nb of the pyphotometry device
# on which rsync signal is sent (from pycontrol device)
rsync_chan = 2

basefolder = Path(os.getcwd()).parent.parent

# These must be absolute paths
# use this to use within package tasks files (in params)
tasksfile = Path(basefolder,'params','tasks_params.csv')
# use this to put a local full path
#tasksfile = -r'C:/.../tasks_params.csv' 

# from sample_data

# # From jade
# photometry_dir = Path('/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/photometry')
# pycontrol_dir = Path('/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/pycontrol')


video_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\videos'
tasks = pd.read_csv(tasksfile, usecols=[1, 2, 3, 4], index_col=False)


# ### Create an experiment object
# 

# In[5]:


# From julien-pc
photometry_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\_Other\test_folder\delayed_go\pyphotometry\delayed_go_dual_2022'
pycontrol_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\_Other\test_folder\delayed_go\pycontrol\delayed_go_dual_2022'
trial_window=[-3000,6000]
exp_cohort_mixed = Experiment(path=pycontrol_dir, int_subject_IDs=True, update=True, verbose=True)


# In[6]:


exp_cohort_mixed.process_exp_by_trial(trial_window, timelim=None, tasksfile=tasksfile, verbose=True)


# In[7]:


# Defime each trial type as a dictionary of conditions to be met
conditions_dict1 = {'trigger': 'hold_for_water', 'success': True, 'water by spout': True}

# conditions_dict2 = {'trigger': 'hold_for_water', 'success': False, 'hold_timer': True}

conditions_dict2 = {'trigger': 'hold_for_water', 'success': False, 'break_after_abort': True}

# conditions_dict2 = {'trigger': 'hold_for_water', 'spout':False, 'valid': True, 'busy_win_timer': False, 'button_press': False}

# Aggregate all condition dictionaries in a list
conditions_list = [conditions_dict1, conditions_dict2]
# Aliases for conditions
cond_aliases = ['Go - hold hit', 'Go - aborted']
# Groups as a list of lists
# groups = [[280, 281, 282, 289],[295, 282, 284, 285, 292, 297]]
groups = None


# In[8]:


# Find if there is a matching photometry file:
exp_cohort_mixed.match_sessions_to_files(photometry_dir, ext='ppd')

# rsync synchronization pulses matching between behaviour and photometry
exp_cohort_mixed.sync_photometry_files(2)
exp_cohort_mixed.save()


# In[9]:


trigs = [None, 'bar_off', 'bar_off', 'spout', 'US_end_timer']
last_befores = [None, None, 'spout', None, None]
trial_window = [-4000, 6000]
photo_dataset_mixed = dict()
for idx, trig in enumerate(trigs):
    photo_dataset_mixed[idx] = exp_cohort_mixed.get_photometry_groups(
            groups = None, # or use groups variable defined above
            conditions_list = conditions_list, 
            cond_aliases = cond_aliases,
            trial_window = trial_window,
            when = 'all', 
            task_names = ['reaching_go_spout_bar_nov22'], #'pavlovian_nobar_nodelay', #'reaching_go_nogo',
            trig_on_ev = trig, # align to the first event of a kind e.g. bar_off
            last_before = last_befores[idx],
            baseline_low_pass = 0.01, 
            low_pass = 45, 
            median_filt = 3,
            motion_corr = True, 
            df_over_f = True,
            z_score = True, 
            downsampling_factor = 10, 
            export_vars = ['analog_1_df_over_f', 'zscored_df_over_f'], 
            # remove_artifacts = False, # To Deprecate in favor of Exp level artifact clustering
            verbose = True) # will plot all the process of remove_artifacts if remove_artifacts == True

# 8m 46s


# Plot photometry trials triggered on different events based on above extraction
# - 1: Trial onset (CS-onset = hold period start)
# - 2: First bar_off
# - 3: Last bar_off before spout
# - 4: First spout
# - 5: Reward (US_end_timer)

# In[10]:


phase_labels = ('Cue_onset','First_mov','Mov_bef_spout','Spout','Reward')

for idx, trig in enumerate(trigs):

    photo_dataset_mixed[idx].filter_reset()
    photo_dataset_mixed[idx].filterout_subjects([0,1,58,63,313,314,315,318])
    photo_dataset_mixed[idx].filter_min_by_session(min_trials = 10)
    photo_dataset_mixed[idx].filter_lastNdays(n = 3)
    if idx == 4:
        figsize = (9.75, 5)
    else:
        figsize = (15, 5)

    fig, axs, out_df = photo_dataset_mixed[idx].lineplot(
        vars = ['zscored_df_over_f'],
        time_lim = [-500, 500],
        # time_unit = 'seconds',
        ylim = [[-1, 5]],# [[-0.004, 0.006]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
        error = True,
        colormap = 'jet',
        legend = True,
        plot_subjects = True,
        plot_groups = True,
        liney0 = False,
        linex0 = True,
        figsize = figsize,
        dpi = 100,
        verbose = False)

    # file_path = 'C:\\Users\\phar0732\\Documents\\GitHub\\trialexp\\outputs\\' + 'photo_ave_' + phase_labels[idx] + '.pdf'
    # fig.savefig(file_path)


# # Compare the results for analog_2_filt

# In[11]:


photo_dataset_mixed2 = dict()
for idx, trig in enumerate(trigs):
    photo_dataset_mixed2[idx] = exp_cohort_mixed.get_photometry_groups(
            groups = None, # or use groups variable defined above
            conditions_list = conditions_list, 
            cond_aliases = cond_aliases,
            trial_window = trial_window,
            when = 'all', 
            task_names = ['reaching_go_spout_bar_nov22'], #'pavlovian_nobar_nodelay', #'reaching_go_nogo',
            trig_on_ev = trig, # align to the first event of a kind e.g. bar_off
            last_before = last_befores[idx],
            baseline_low_pass = 0.01, 
            low_pass = 45, 
            median_filt = 3,
            motion_corr = True, 
            df_over_f = True,
            z_score = True, 
            downsampling_factor = 10, 
            export_vars = ['analog_2_filt'], 
            # remove_artifacts = False, # To Deprecate in favor of Exp level artifact clustering
            verbose = True) # will plot all the process of remove_artifacts if remove_artifacts == True

# 9m 40s


# In[13]:


phase_labels = ('Cue_onset','First_mov','Mov_bef_spout','Spout','Reward')

for idx, trig in enumerate(trigs):

    photo_dataset_mixed2[idx].filter_reset()
    photo_dataset_mixed2[idx].filterout_subjects([0,1,58,63,313,314,315,318])
    photo_dataset_mixed2[idx].filter_min_by_session(min_trials = 10)
    photo_dataset_mixed2[idx].filter_lastNdays(n = 3)
    if idx == 4:
        figsize = (9.75, 5)
    else:
        figsize = (15, 5)

    fig, axs, out_df = photo_dataset_mixed2[idx].lineplot(
        vars = ['analog_2_filt'],
        time_lim = [-500, 500],
        # time_unit = 'seconds',
        ylim = [[0, 1]],# [[-0.004, 0.006]],#[[-0.03, 0.1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
        error = True,
        colormap = 'jet',
        legend = True,
        plot_subjects = True,
        plot_groups = True,
        liney0 = False,
        linex0 = True,
        figsize = figsize,
        dpi = 100,
        verbose = False)

    # file_path = 'C:\\Users\\phar0732\\Documents\\GitHub\\trialexp\\outputs\\' + 'photo_ave_' + phase_labels[idx] + '.pdf'
    # fig.savefig(file_path)


# Don't understand why they are so flat? Looks very suspicious
