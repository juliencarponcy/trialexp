#!/usr/bin/env python
# coding: utf-8

# ## Workflow to analyze pyControl data
# 
# ```bash
# jupyter nbconvert "D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical\nb20221002_214600_workflow_pyCon_PavAll.ipynb" --to="python" --output-dir="D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical" --output="nb20221002_214600_workflow_pyCon_PavAll"
# ```

# ### Imports

# In[357]:


# Import Session and Experiment class with helper functions
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Import Session and Experiment class with helper functions
from trialexp.process.data_import import *


# ### Variables

# In[358]:


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

# In[359]:


tasks = pd.read_csv(tasksfile, usecols = [1,2,3,4], index_col = False)
tasks


# # Optional
# 
# 1m 7s

# In[360]:


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

# In[361]:


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

# In[362]:


exp_cohort.sessions = exp_cohort.get_sessions(task_names='pavlovian_nobar_nodelay')
# exp_cohort.save()
print(len(exp_cohort.sessions ))
exp_cohort.subject_IDs


# ### Perform extraction of behavioural information by trial
# 
# 3m 7.6s

# In[363]:


# Process the whole experimental folder by trials
exp_cohort.process_exp_by_trial(trial_window, timelim, tasksfile, 
  blank_spurious_event='spout', blank_timelim=[0, 65])

# Save the file as sessions.pkl in folder_path
# exp_cohort.save()


# In[364]:


len(exp_cohort.sessions)


# In[365]:


exp_cohort.sessions[1].df_events.head(50)


# # Define conditions and groups for extraction

# ## Pavlovian

# In[366]:


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


# In[367]:


exp_cohort.sessions


# Behaviour: Create a dataset

# In[368]:


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

# In[369]:


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

# In[370]:


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

# In[371]:


dist_as_continuous.metadata_df.head(50)


# ## Behaviour: Plot distribution
# 
# #TODO what is T = 0?
# How to plot differently? Or not necessary?

# In[372]:


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

# In[375]:



metadata_df = ev_dataset.metadata_df.loc[ev_dataset.metadata_df['keep'] & ev_dataset.metadata_df['valid'], :]
metadata_df


# In[213]:



from numpy import NaN


group_IDs = list(set(metadata_df['group_ID']))

for g in group_IDs:
  subject_IDs = list(set(metadata_df.loc[metadata_df.group_ID == g, 'subject_ID']))

  ss_dfs = [0] * len(subject_IDs)
  for s_idx, s in enumerate(subject_IDs):
    session_nbs = list(set(metadata_df.loc[
      (metadata_df['group_ID'] == g)
      & (metadata_df['subject_ID'] == s) , 
      'session_nb']))

    ss_sc = [NaN] * len(session_nbs)
    ss_tn = [NaN] * len(session_nbs)
    ss_sr = [NaN] *  len(session_nbs)
   
    for ss_idx, ss in enumerate(session_nbs):
      ss_sc[ss_idx] = np.count_nonzero(metadata_df.loc[
        (metadata_df['group_ID'] == g)
        & (metadata_df['subject_ID'] == s)
        & (metadata_df['session_nb'] == ss),
        'success'])

      ss_tn[ss_idx] = len(metadata_df.loc[
        (metadata_df['group_ID'] == g)
        & (metadata_df['subject_ID'] == s)
        & (metadata_df['session_nb'] == ss),
        'success'])       

    ss_sr = (np.array(ss_sc)/np.array(ss_tn)).tolist()

    ss_df = pd.DataFrame(list(zip(session_nbs, ss_sc, ss_tn, ss_sr)))
    ss_df.columns = ['session_nb','success_n','trial_n', 'success_rate']

    ss_df.astype({'session_nb':'int', 'success_n': 'int', 'trial_n' : 'int'})

    ss_df['subject_ID'] = pd.Series([s] * len(session_nbs))

    ss_dfs[s_idx] = ss_df

ss_dfs = pd.concat(ss_dfs)
gr_df = pd.DataFrame(list(zip([g] * len(subject_IDs), subject_IDs)),
                     columns=['group_ID', 'subject_ID'])

gr_df = pd.merge(gr_df, ss_dfs, 'outer')


# In[380]:



gr_df


# ### By Days, Dates, or By Sessions?
# 

# In[390]:


bywhat = 'days'

def extract_successrates(gr_df : pd.DataFrame, bywhat):

  group_IDs = list(set(gr_df['group_ID']))
  out_list = [None] * len(group_IDs)
  for g_idx, g in enumerate(group_IDs):
    subject_IDs = list(
        set(gr_df.loc[gr_df['group_ID'] == g, 'subject_ID']))

    ss_dfs = [0] * len(subject_IDs)
    out_listlist = [None] * len(subject_IDs)
    for s_idx, s in enumerate(subject_IDs):
      thismouse_df = gr_df.loc[(gr_df['group_ID'] == g)
        & (gr_df['subject_ID'] == s), :]

      if bywhat == 'days':
        # gaps are ignored
        # success rate is computed daily
        dates = list(set(thismouse_df.date))
        dates.sort()
        sr = [None] * len(dates)
        for d_idx, d in enumerate(dates):
          X = thismouse_df.loc[thismouse_df.date == d,
              [ 'success_n', 'trial_n']].sum(axis=0)

          sr[d_idx] = X.success_n/X.trial_n

        out_listlist[s_idx] = pd.DataFrame(list(zip(range(1, len(dates)+1), sr)), columns=['dayN', str(s)])
        out_listlist[s_idx] = out_listlist[s_idx].set_index('dayN')

      elif bywhat == 'days_with_gaps':
        # gaps are considered
        # success rate is computed daily
        dates = list(set(thismouse_df.date))
        dates.sort()
        sr = [None] * len(dates)
        for d_idx, d in enumerate(dates):
          X = thismouse_df.loc[thismouse_df.date == d,
                              ['success_n', 'trial_n']].sum(axis=0)

          sr[d_idx] = X.success_n/X.trial_n

        out_listlist[s_idx] = pd.DataFrame(
            list(zip([d - dates[0] for d in dates], sr)), columns=['dayN', str(s)])
        out_listlist[s_idx] = out_listlist[s_idx].set_index('dayN')
      elif bywhat == 'dates':
        # gaps are considered
        # success rate is computed daily
        # use dates instead of days
        dates = list(set(thismouse_df.date))
        dates.sort()
        sr = [None] * len(dates)
        for d_idx, d in enumerate(dates):
          X = thismouse_df.loc[thismouse_df.date == d,
              [ 'success_n', 'trial_n']].sum(axis=0)

          sr[d_idx] = X.success_n/X.trial_n

        out_listlist[s_idx] = pd.DataFrame(list(zip(dates, sr)), columns=['dates', str(s)])
        out_listlist[s_idx] = out_listlist[s_idx].set_index('dates')
      elif bywhat == 'sessions':
        # gaps are ignored
        # success rate is computed by session
          
        out_listlist[s_idx] = thismouse_df.loc[:, ['session_nb', 'success_rate']] 
        out_listlist[s_idx] = out_listlist[s_idx].set_index('session_nb')
        out_listlist[s_idx].columns = [str(s)]
      else:
        raise 'bywhat must be ''days'', ''days_with_gap'', ''dates'', or ''sessions'''

    #out_list[g_idx] = 1 = pd.concat(out_listlist) # TODO this is wrong
    # session_nb as index, success_rate as value, subject_ID as columns
    out_list[g_idx] = pd.concat(out_listlist, axis=1)

      
  return out_list


# ### sessions

# In[399]:


import matplotlib

bywhat = 'sessions'  # 'days', 'dates', 'sessions'
out_list = extract_successrates(gr_df, bywhat)

fig, (ax1, ax2) = plt.subplots(2,1)

nml = matplotlib.colors.Normalize(0,1)

im1 = ax1.imshow(out_list[0].iloc[:,0:6].transpose(), norm=nml)
ax1.set_title('Cohort 1')
ax1.set_xlabel('Sessions')
ax1.set_ylabel('Mice')
ax1.set_yticks(range(0,6))
ax1.set_yticklabels(out_list[0].columns[0:6])
ax1.set_facecolor('k')
ax1.tick_params(axis='both', which='major', labelsize=8)


im2 = ax2.imshow(out_list[0].iloc[:,6:].transpose(), norm=nml)
ax2.set_title('Cohort 2')
ax2.set_xlabel('Sessions')
ax2.set_ylabel('Mice')
ax2.set_yticks(range(0, 7))
ax2.set_yticklabels(out_list[0].columns[6:])
ax2.set_facecolor('k')
ax2.tick_params(axis='both', which='major', labelsize=8)

cb1 = plt.colorbar(im1, location='right', ax=[ax1, ax2], label='Success rate')



# ### days

# In[400]:



bywhat = 'days'  # 'days', 'dates', 'sessions'
out_list = extract_successrates(gr_df, bywhat)

fig, (ax1, ax2) = plt.subplots(2, 1)

nml = matplotlib.colors.Normalize(0, 1)

im1 = ax1.imshow(out_list[0].iloc[:, 0:6].transpose(), norm=nml)
ax1.set_title('Cohort 1')
#ax1.set_xlabel('Days')
ax1.set_ylabel('Mice')
ax1.set_yticks(range(0, 6))
ax1.set_yticklabels(out_list[0].columns[0:6])
ax1.set_facecolor('k')
ax1.tick_params(axis='both', which='major', labelsize=8)


im2 = ax2.imshow(out_list[0].iloc[:, 6:].transpose(), norm=nml)
ax2.set_title('Cohort 2')
ax2.set_xlabel('Days')
ax2.set_ylabel('Mice')
ax2.set_yticks(range(0, 7))
ax2.set_yticklabels(out_list[0].columns[6:])
ax2.set_facecolor('k')
ax2.tick_params(axis='both', which='major', labelsize=8)

cb1 = plt.colorbar(im1, location='right', ax=[ax1, ax2], label='Success rate')


# ### days_with_gaps

# In[401]:



bywhat = 'days_with_gaps'  # 'days', 'dates', 'sessions'
out_list = extract_successrates(gr_df, bywhat)

fig, (ax1, ax2) = plt.subplots(2, 1)

nml = matplotlib.colors.Normalize(0, 1)

im1 = ax1.imshow(out_list[0].iloc[:, 0:6].transpose(), norm=nml)
ax1.set_title('Cohort 1')
#ax1.set_xlabel('Days')
ax1.set_ylabel('Mice')
ax1.set_yticks(range(0, 6))
ax1.set_yticklabels(out_list[0].columns[0:6])
ax1.set_facecolor('k')
ax1.tick_params(axis='both', which='major', labelsize=8)


im2 = ax2.imshow(out_list[0].iloc[:, 6:].transpose(), norm=nml)
ax2.set_title('Cohort 2')
ax2.set_xlabel('Days')
ax2.set_ylabel('Mice')
ax2.set_yticks(range(0, 7))
ax2.set_yticklabels(out_list[0].columns[6:])
ax2.set_facecolor('k')
ax2.tick_params(axis='both', which='major', labelsize=8)

cb1 = plt.colorbar(im1, location='right', ax=[ax1, ax2], label='Success rate')


# ### Dates

# In[421]:


out_list[0].index[range(0, 30, 5)]


# In[424]:




bywhat = 'dates'  # 'days', 'dates', 'sessions'
out_list = extract_successrates(gr_df, bywhat)

fig, (ax1, ax2) = plt.subplots(2, 1)

nml = matplotlib.colors.Normalize(0, 1)

im1 = ax1.imshow(out_list[0].iloc[:, 0:6].transpose(), norm=nml)
ax1.set_title('Cohort 1')
ax1.set_xlabel('Dates')
ax1.set_ylabel('Mice')
ax1.set_xticks(range(0, 30, 5)) # needed
ax1.set_xticklabels(
    out_list[0].index[range(0, 30, 5)], rotation=30, ha='right')
ax1.set_yticks(range(0, 6))
ax1.set_yticklabels(out_list[0].columns[0:6])
ax1.set_facecolor('k')
ax1.tick_params(axis='both', which='major', labelsize=8)


im2 = ax2.imshow(out_list[0].iloc[:, 6:].transpose(), norm=nml)
ax2.set_title('Cohort 2')
ax2.set_xlabel('Dates')
ax2.set_ylabel('Mice')
ax2.set_xticks(range(0, 30, 5))  # needed
ax2.set_xticklabels(out_list[0].index[range(0, 30, 5)], rotation=30, ha='right')
ax2.set_yticks(range(0, 7))
ax2.set_yticklabels(out_list[0].columns[6:])
ax2.set_facecolor('k')
ax2.tick_params(axis='both', which='major', labelsize=8)

cb1 = plt.colorbar(im1, location='right', ax=[ax1, ax2], label='Success rate')


# In[49]:



exp_cohort.sessions[1].df_events.head(50)

realsuccess_n = np.zeros((len(exp_cohort.sessions), 1))
trial_n = np.zeros((len(exp_cohort.sessions), 1))
successrate = np.zeros((len(exp_cohort.sessions), 1))

for s_idx, s in enumerate( exp_cohort.sessions):
    realsuccess_tf = np.full((s.df_events.shape[0], 1), False)
    for r in range(0, s.df_events.shape[0]):
      if s.df_events.loc[r+1,'spout_trial_time']:
        if any(x > 65 for x in s.df_events.loc[r+1, 'spout_trial_time']):
          realsuccess_tf[r] = True

    realsuccess_n[s_idx] = np.count_nonzero(realsuccess_tf)
    trial_n[s_idx] = s.df_events.shape[0]
    successrate[s_idx] = realsuccess_n[s_idx]/trial_n[s_idx]

df_success = pd.DataFrame(np.hstack([realsuccess_n, trial_n, successrate]))
df_success.columns = ['realsuccess_n', 'trial_n', 'success_rate']


del realsuccess_n, trial_n, successrate #, s_idx, s, r
#TODO otther columns, animal, date, task etc


# In[57]:


s.df_conditions


# ## Set DeepLabCut bodyparts to compute paws centroids

# In[158]:


# Name of the labelled body parts from both upper limbs
# The bodyparts from which we draw here are user-defined
# when creating a new DeepLabCut project (config.yaml)

L_paw_parts  = ['MCP II', 'MCP III', 'MCP IV', 'MCP V', 'IP II', 'IP III',     'IP IV', 'IP V', 'tip II', 'tip III', 'tip IV', 'tip V'] 

R_paw_parts = ['r MCP II', 'r MCP III', 'r MCP IV', 'r MCP V', 'r IP II',     'r IP III', 'r IP IV', 'r IP V', 'r tip II', 'r tip III', 'r tip IV', 'r tip V']

names_of_ave_regions = ['Left_paw','Right_paw']


# ### Extract DeepLabCut trials and create a Continuous_Dataset

# In[ ]:


cont_dataset = exp_cohort.get_deeplabcut_groups(
        groups = None,
        conditions_list = condition_list,
        cond_aliases = cond_aliases,
        when='all', 
        task_names = ['reaching_go_nogo'],
        bodyparts_to_ave = [L_paw_parts, R_paw_parts],
        names_of_ave_regions = ['Left_paw','Right_paw'], 
        bodyparts_to_store = ['spout', 'jaw', 'ear', 'tongue', 'tip III',  'IP III', 'MCP III'],
        normalize_between = ['Left_paw', 'spout'],
        bins_nb = 100,
        three_dims = False, 
        p_thresh = 0.9,
        camera_fps = 100, # not yet functional
        camera_keyword = 'Side', 
        trig_on_ev = None, 
        verbose = True)


# ### Save DLC Dataset

# In[338]:


folder_path = r'C:\Users\phar0732\Documents\GitHub\pycontrol_share\outputs'

cont_dataset.save(folder_path, 'DLC_dataset_gonogo')


# ### Reload a pre-existing dataset

# In[110]:


dataset_full_path = r'C:\Users\phar0732\Documents\GitHub\pycontrol_share\outputs\DLC_dataset_opto_continuous_full.pkl'
cont_dataset = load_dataset(dataset_full_path)


# In[111]:


cont_dataset.metadata_df


# ### Optional methods

# In[374]:


# Get a list of the groups
cont_dataset.get_groups()
# Get a list of the variables
cont_dataset.get_col_names()

# reset previous filtering of the dataset
cont_dataset.filter_reset()

# exclude some conditions by IDs
cont_dataset.filter_conditions([])

# exclude some groups by IDs
cont_dataset.filter_groups([])

# exclude some subjects
cont_dataset.filter_subjects([0, 1,289,299,305,306])
#     subjects_IDs_to_exclude = [289, 290, 293, 294, 295, 299, 301, 303, 304, 305, 306])

# filter subjects/sessions with less than x trials (by condition)
cont_dataset.filter_min(min_trials = 10)

# To remove subjects who do not have
# trials in all the conditions
# cont_dataset.filter_if_not_in_all_cond()

# method to build (not finished)
# cont_dataset.set_groups()


# In[375]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from trial_dataset_classes import *


# In[378]:


### Plot the photometry by condition
cont_dataset.set_trial_window([-2, 6], 's')
cont_dataset.set_conditions(condition_list, cond_aliases)
cont_dataset.lineplot(
    vars = ['Left_paw_x'],
    time_lim = [-2000, 6000],
    time_unit = 'milliseconds',
    error = True,
    ylim = [[-0.05, 0.8]], #[[-0.1, 1]],#,[-0.005, 0.007]],#[[-0.001, 0.0011],[-0.001, 0.0011]],
    colormap = 'jet',
    legend = True,
    plot_subjects = False,
    plot_groups = True,
    figsize = (5,5),
    dpi = 100,
    verbose = False)

# Return a count of overall number of trials
cont_dataset.metadata_df['keep'].value_counts()


# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
cont_dataset.set_groups(groups)


# In[276]:


cont_dataset.metadata_df.keep[cont_dataset.metadata_df.subject_ID == 307].value_counts()


# In[177]:


for row in cont_dataset.metadata_df.itertuples():
    print(row.group_ID)


# In[171]:


row


# In[172]:


row.subject_ID


# In[ ]:




