#!/usr/bin/env python
# coding: utf-8

# # nb20230212_194200_reaching_go_spout_incr_break2_nov22_JC313_315
# 
# ```bash
# jupyter nbconvert "D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical\nb20230212_194200_reaching_go_spout_incr_break2_nov22_JC313_315.ipynb" --to="python" --output-dir="D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical" --output="nb20230212_194200_reaching_go_spout_incr_break2_nov22_JC313_315"
# ```

# Quick analysis of instrumental reaching

# 

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


# In[3]:


tasks = pd.read_csv(tasksfile, usecols=[1, 2, 3, 4], index_col=False)
tasks


# In[4]:


photo_root_dir = 'T:\\Data\\head-fixed\\pyphotometry\\data'
pycontrol_root_dir = 'T:\\Data\\head-fixed\\pycontrol'


# ### Create an experiment object
# 

# In[5]:


# Folder of a full experimental batch, all animals included

# Enter absolute path like this
# pycontrol_files_path = r'T:\Data\head-fixed\test_folder\pycontrol'

# or this if you want to use data from the sample_data folder within the package
#pycontrol_files_path = os.path.join(basefolder, 'sample_data/pycontrol')
pycontrol_files_path = r'T:\Data\head-fixed\pycontrol\reaching_go_spout_incr_break2_nov22'

# Load all raw text sessions in the indicated folder or a sessions.pkl file
# if already existing in folder_path
exp_cohort = Experiment(pycontrol_files_path, update = True) #TODO

# Only use if the Experiment cohort as been processed by trials before
# TODO: assess whether this can be removed or not
exp_cohort.by_trial = True


smrx_folder_path = r'T:\Data\head-fixed\kms_pycontrol\smrx'


# ## Select sessions

# In[6]:


import datetime
ss = exp_cohort.sessions

ss_ = [this_ss for this_ss in ss
       # [313, 314, 315, 316, 317, 318]
       if (this_ss.subject_ID in [313, 314, 315])
       and (this_ss.experiment_name == 'reaching_go_spout_incr_break2_nov22')
       and (this_ss.datetime.date() >= datetime.date(2022, 12, 3))]
ss_


# In[7]:


exp_cohort.sessions = ss_


# In[8]:


ss_


# In[9]:


ss_[0].datetime.date()


# # SLOW 3m

# In[10]:


# # Process the whole experimental folder by trials

# exp_cohort.process_exp_by_trial(
#     trial_window, timelim, tasksfile, blank_spurious_event='spout', blank_timelim=[0, 65])
#     # not working

# # Find if there is a matching photometry file and if it can be used:
# # rsync synchronization pulses matching between behaviour and photometry

# # Find if there is a matching photometry file:
# exp_cohort.match_sessions_to_files(photometry_dir, ext='ppd')

# # rsync synchronization pulses matching between behaviour and photometry
# exp_cohort.sync_photometry_files(2)

# # Find matching videos
# exp_cohort.match_sessions_to_files(video_dir, ext='mp4')

# # FInd matching DeepLabCut outputs files
# exp_cohort.match_sessions_to_files(video_dir, ext='h5', verbose=True)


# # exp_cohort.save()


# In[11]:


exp_cohort.subject_IDs


# In[12]:


# Many combinations possible
conditions_dict0 = {'trigger': 'hold_for_water', 'valid': True}


# Aggregate all condition dictionaries in a list
condition_list = [conditions_dict0]
# Aliases for conditions
cond_aliases = [
    'any_trial',
]

# Groups as a list of lists
groups = None

# right_handed = [281]
# groups = [[280, 282, 299, 300, 301],\
#     [284, 285, 296, 297, 306, 307]]
# Window to exctract (in ms)


# In[13]:


exp_cohort.sessions[0].times.keys()


# # Session plot 
# 
# I realised that this plot can never tell if a water drop was triggered by bar_off or spout.
# 

# In[14]:


exp_cohort.sessions[0].print_lines[0:30]


# In[15]:


import re

re.match('abc ','abc de')

expr = '^\d+(?= ' + '.?Timestamp' + ')'
a = [re.match(expr, L) for L in exp_cohort.sessions[0].print_lines if re.match(expr , L) is not None]
int(a[0].group(0))


# In[ ]:





# In[16]:


for ss in exp_cohort.sessions:
    smrxname = re.sub('\.txt', f'_{ss.task_name}.smrx', ss.file_name)
    print(smrxname)


# 
# # export

# In[22]:


keys = [
        'button_press', 'bar', 'bar_off', 'spout', 'US_delay_timer', 'CS_offset_timer']

state_def = [
    {'name': 'busy_win',    'onset': 'busy_win',    'offset': 'short_break'},
    {'name': 'short_break', 'onset': 'short_break', 'offset': 'busy_win'}]

summary_df = pd.DataFrame()

for ss in exp_cohort.sessions:

    file_name = os.path.split(ss.file_name)
    file_name_ = re.sub('\.txt',  f'_{ss.task_name}.smrx', file_name[1])
    smrxname = os.path.join(smrx_folder_path, file_name_)
    print(smrxname)


    bw = ss.times['busy_win']
    sp = ss.times['spout']

    x_spout = [this_bw for this_bw in bw for spouts in sp if (
        spouts < this_bw) and (this_bw - spouts < 100)]

    x_bar = [this_bw for this_bw in bw if not any(
        [(spouts < this_bw) and (this_bw - spouts < 100) for spouts in sp])]
        
    event_ms = [{
        'name': 'triggered by spout',
        'time_ms': x_spout
    },
    {
        'name': 'triggered by bar_off',
        'time_ms': x_bar
    }
    ]

    try:
        ss.plot_session(
            keys, state_def, export_smrx=True, event_ms=event_ms, smrx_filename= smrxname)

        summary_df = pd.concat([summary_df, 
            pd.DataFrame({
                'file':ss.file_name,
                'task':ss.task_name,
                'triggered_by_spout': len(x_spout),
                'triggered_by_bar_off': len(x_bar),
                'reaching_trials': len(bw),
                'trials': len(ss.times['busy_win'])},
                index=[0])
                ],
                ignore_index=True)
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}, for {file_name_}")



# ## trouble shooting for 314
# 
# error is in event channel
# 
# which channel?
# ReadEvents seems working for the first 10 divides at least

# In[ ]:


keys = [
        'button_press', 'bar', 'bar_off', 'spout', 'US_delay_timer', 'CS_offset_timer']
state_def = [{'name': 'hold_for_water', 'onset': 'hold_for_water', 'offset': 'waiting_for_spout'},
                    {'name': 'waiting_for_spout', 'onset': 'waiting_for_spout',
                    'offset': 'busy_win'},
                    {'name': 'busy_win', 'onset': 'busy_win',
                        'offset': 'break_after_water'},
                    {'name': 'break_after_water', 'onset': 'break_after_water',    'offset': 'waiting_for_bar'},
                    {'name': 'break_after_no_water',       'onset': 'break_after_no_water', 'offset': 'waiting_for_bar'}]
summary_df = pd.DataFrame()

for ss in [exp_cohort.sessions[0]]:
    smrxname = re.sub('\.txt', f'_{ss.task_name}.smrx', ss.file_name)
    print(smrxname)


    bw = ss.times['busy_win']
    sp = ss.times['spout']

    x_spout = [this_bw for this_bw in bw for spouts in sp if (
        spouts < this_bw) and (this_bw - spouts < 100)]

    x_bar = [this_bw for this_bw in bw if not any(
        [(spouts < this_bw) and (this_bw - spouts < 100) for spouts in sp])]
        
    event_ms = [{
        'name': 'triggered by spout',
        'time_ms': x_spout
    },
        {
            'name': 'triggered by bar_off',
            'time_ms': x_bar
    }
    ]

    if re.search('11\-23',ss.file_name): #adapt to a bug 
        state_def[-1]['offset'] = 'wating_for_bar'
    else:
        state_def[-1]['offset'] = 'waiting_for_bar'


    ss.plot_session(
        keys, state_def, export_smrx=True, event_ms=event_ms, srmx_filename= smrxname, verbose=True)

    summary_df = summary_df.append({
        'file':ss.file_name,
        'task':ss.task_name,
        'triggered_by_spout': len(x_spout),
        'triggered_by_bar_off': len(x_bar),
        'reaching_trials': len(bw),
        'trials': len(ss.times['hold_for_water'])},
        ignore_index=True)


# In[ ]:




