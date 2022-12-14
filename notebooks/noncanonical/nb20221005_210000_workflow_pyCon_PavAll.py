#!/usr/bin/env python
# coding: utf-8

# ## Workflow to analyze pyControl data
# 
# 

# ```batch script
# jupyter nbconvert "D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical\nb20221005_210000_workflow_pyCon_PavAll.ipynb" --to="python" --output-dir="D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical" --output="nb20221005_210000_workflow_pyCon_PavAll"
# ```

# ### Imports

# In[1]:


# Import Session and Experiment class with helper functions
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
tasksfile = os.path.join(basefolder,'params/tasks_params.csv')
# use this to put a local full path
#tasksfile = -r'C:/.../tasks_params.csv' 

photometry_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\test_folder\photometry'
video_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\videos'


# ### Tasks
# - A tasks definition file (.csv) contains all the information to perform the extractions of behaviorally relevant information from **PyControl** files, for each **task** file. It includes what are the **triggers** of different trial types, what **events** to extract (with time data), and what are events or printed lines that could be relevant to determine the **conditions** (e.g: free reward, optogenetic stimulation type, etc.)
# - To analyze a new task you need to append task characteristics like **task** filename, **triggers**, **events** and **conditions**

# In[3]:


tasks = pd.read_csv(tasksfile, usecols = [1,2,3,4], index_col = False)
tasks


# # Optional
# 
# 1m 7s

# In[4]:


photo_root_dir = 'T:\\Data\\head-fixed\\pyphotometry\\data'
pycontrol_root_dir = 'T:\\Data\\head-fixed\\pycontrol'

root_folders = [photo_root_dir, pycontrol_root_dir]
horizontal_folder_pycontrol = 'T:\\Data\\head-fixed\\test_folder\\pycontrol'
horizontal_folder_photometry = 'T:\\Data\\head-fixed\\test_folder\\photometry'

copy_files_to_horizontal_folders(
    root_folders, horizontal_folder_pycontrol, horizontal_folder_photometry)


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


# ## retain only pavlovian sessions

# In[6]:


exp_cohort.sessions = exp_cohort.get_sessions(task_names='pavlovian_nobar_nodelay')
# exp_cohort.save()
print(len(exp_cohort.sessions ))
exp_cohort.subject_IDs


# ### Perform extraction of behavioural information by trial
# 
# 3m 7.6s

# In[7]:


# Process the whole experimental folder by trials
exp_cohort.process_exp_by_trial(trial_window, timelim, tasksfile, 
  blank_spurious_event='spout', blank_timelim=[0, 65])

# Save the file as sessions.pkl in folder_path
# exp_cohort.save()


# In[8]:


len(exp_cohort.sessions)


# In[9]:


exp_cohort.sessions[1].df_events.head(50)


# # Define conditions and groups for extraction

# ## Pavlovian

# In[10]:


# List of uncued conditions as listed on the tasks .csv file for task pavlovian_nobar_nodelay:
# free_reward_timer; reward spout cued; reward bar cued; reward bar_off; reward spout uncued; reward bar uncued; reward free; reward free_uncued

# Many combinations possible
conditions_dict0 = {'success': True}
conditions_dict1 = {'success': False}


# Aggregate all condition dictionaries in a list
condition_list = [conditions_dict0, conditions_dict1]
# Aliases for conditions
cond_aliases = ['Hit', 'Miss']
# Groups as a list of lists
groups = None

# right_handed = [281]
# groups = [[280, 282, 299, 300, 301],\
#     [284, 285, 296, 297, 306, 307]]
# Window to exctract (in ms)
trial_window = [-2000, 6000]


# In[11]:


exp_cohort.sessions


# Behaviour: Create a dataset

# In[12]:


ev_dataset = exp_cohort.behav_events_to_dataset(
        groups = groups,
        conditions_list = condition_list, 
        cond_aliases = cond_aliases, 
        when = 'all', 
        task_names='pavlovian_nobar_nodelay',
        trig_on_ev = None)

ev_dataset.set_trial_window(trial_window=trial_window, unit='milliseconds')
ev_dataset.set_conditions(conditions=condition_list, aliases=cond_aliases)


# Behaviour: Compute distribution

# In[13]:


dist_as_continuous = ev_dataset.compute_distribution(
    trial_window=trial_window,
        bin_size = 100, # do not work as expected with cued-uncued
        normalize = True,
        per_session = True,
        out_as_continuous = True)
dist_as_continuous.set_conditions(conditions=condition_list, aliases=cond_aliases)
# Remove test files
dist_as_continuous.filterout_subjects([0,1])


# ### Optional methods
# 
# - Implementation of these optional filtering options is first understood as removing subjects, groups, conditions...
# - It is a non-permanent way of discarding elements for analysis
# - It is based on a "keep" column in the metadata that is True by default and set to False with the filtering function.
# - At anytime, <trial_dataset>.filter_reset() can be called to re-include all the elements in the analysis (set all "keep" to True)
# - Comment or uncomment lines and fill the lists according to your needs

# In[14]:


# # Get a list of the groups
# dist_as_continuous.get_groups()
# # Get a list of the variables
# dist_as_continuous.get_col_names()

# # reset previous filtering of the dataset
# dist_as_continuous.filter_reset()

# # exclude some conditions by IDs
# dist_as_continuous.filterout_conditions([])

# # exclude some groups by IDs
# dist_as_continuous.filterout_groups([])

# # exclude some subjects
# dist_as_continuous.filterout_subjects([0, 1])
# #     subjects_IDs_to_exclude = [289, 290, 293, 294, 295, 299, 301, 303, 304, 305, 306])

# # filter subjects/sessions with less than x trials (by condition)
# dist_as_continuous.filter_min(min_trials = 1)

# # To remove subjects who do not have trials
# # in all the conditions, if called after filter_min(),
# # will discard any subject who do not have the minimum number
# # of trials in all the conditions

# # dist_as_continuous.filter_if_not_in_all_cond()


# Indicative preview of the behavioural metadata

# In[15]:


dist_as_continuous.metadata_df.head(50)


# ## Behaviour: Plot distribution
# 
# #TODO what is T = 0?
# How to plot differently? Or not necessary?

# In[16]:


import trialexp.utils.pycontrol_utilities as pycutl

dist_as_continuous.set_trial_window([a/1000 for a in trial_window], 's')

figs, axs, df1 = dist_as_continuous.lineplot(
    vars = [ 'spout_dist'],
    time_lim = [-1,4],
    time_unit='s',
    error = True,
    ylim = None,#[[-0.1,1.6]], #[[-0.1, 0.7]], #[[-0.1, 1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    colormap = pycutl.cmap10(),
    legend = True,
    plot_subjects = False,
    plot_groups = True,
    figsize = (5*1.618, 5),
    dpi = 200,
    verbose = False)


axs[0, 0].set_xlabel('Relative to Cue onset (s)', fontsize=14) #TODO not sure
axs[0, 0].set_ylabel('Spout touches (counts/s)', fontsize=14) #TODO not sure
# Return a count of overall number of trials
dist_as_continuous.metadata_df['keep'].value_counts()


# ## Success Rate computation
# 
# 65 ms window for spout touch
# 
# df_events.spout_trial_time < 65
# 
# `Session.compute_success()` does this
# 
# and is inherited to `ev_dataset.meatadata_df` and `ev_dataset.data`
# 
# 
# 
# 

# ### By Days, Dates, or By Sessions?
# 

# ### sessions

# In[26]:


gr_df, out_list, _, _ = ev_dataset.analyse_successrate(
    [0], [1], bywhat='sessions')

print(gr_df)

print(out_list)


# ### sessions with gaps

# In[27]:


gr_df, out_list, _, _ = ev_dataset.analyse_successrate(
    [0], [1], bywhat='sessions_with_gaps')

print(gr_df)

print(out_list)


# ### days

# In[28]:


gr_df, out_list, _, _ = ev_dataset.analyse_successrate([0], [1], bywhat='days')

print(gr_df)

print(out_list)


# ### days_with_gaps

# In[29]:


gr_df, out_list, _, _ = ev_dataset.analyse_successrate(
    [0], [1], bywhat='days_with_gaps')

print(gr_df)

print(out_list)


# ### Dates

# In[30]:


out_list[0].index


# In[31]:


gr_df, out_list, ax, _ = ev_dataset.analyse_successrate([0], [1], bywhat='dates')

print(gr_df)

print(out_list)

ax.set_xticks(range(0, len(out_list[0].index), 5))
ax.set_xticklabels(out_list[0].index[range(0, len(out_list[0].index), 5)])

