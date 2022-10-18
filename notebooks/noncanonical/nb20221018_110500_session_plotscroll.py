#!/usr/bin/env python
# coding: utf-8

# # visalise Session as Scroll using Ploty `Session.plotscroll()`
# 
# ```bash
# jupyter nbconvert "D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical\nb20221018_110500_session_plotscroll.ipynb" --to="python" --output-dir="D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical" --output="nb20221018_110500_session_plotscroll"
# ```
# 
# ## Requirements
# 
# Plotly `conda install plotly`
# 
# 
#     fftw-3.3.9                 |       h2bbff1b_1         672 KB
#     icc_rt-2022.1.0            |       h6049295_2         6.5 MB
#     numpy-1.23.1               |   py38h7a0a035_0          10 KB
#     numpy-base-1.23.1          |   py38hca35cd5_0         5.0 MB
#     plotly-5.9.0               |   py38haa95532_0         4.1 MB
#     scikit-learn-1.1.2         |   py38hd77b12b_0         5.5 MB
#     scipy-1.9.1                |   py38he11b74f_0        15.7 MB
#     setuptools-63.4.1          |   py38haa95532_0         1.0 MB
#     tenacity-8.0.1             |   py38haa95532_1          34 KB

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


# ## Visualise a session using Plotly

# In[8]:


import plotly.graph_objects as go


# %TODO
# - drowdown to change time units
# 

# In[9]:


exp_cohort.sessions[0].plot_session()


# In[10]:


session = exp_cohort.sessions[50]


# In[14]:


exp_cohort.sessions[0].plot_session(state_def=dict(
    name='trial', onset='CS_Go', offset='refrac_period'))


# In[19]:


exp_cohort.sessions[0].plot_session(state_def=[
    dict(name='trial', onset='CS_Go', offset='refrac_period'),
    dict(name='bar-touch', onset='bar', offset='bar_off'),
    dict(name='reach', onset='bar_off', offset='spout'),
    ])


# In[ ]:




