#!/usr/bin/env python
# coding: utf-8

# # visalise Session as Scroll using Ploty `Session.plotscroll()`
# 
# ```bash
# jupyter nbconvert "D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical\nb20221017_104300_session_plotscroll.ipynb" --to="python" --output-dir="D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical" --output="nb20221017_104300_session_plotscroll"
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

# In[24]:


# allow for automatic reloading of classes and function when updating the code
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Import Session and Experiment class with helper functions
from trialexp.process.data_import import *


# ### Variables

# In[25]:


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

# In[26]:


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

# In[27]:


photo_root_dir = 'T:\\Data\\head-fixed\\pyphotometry\\data'
pycontrol_root_dir = 'T:\\Data\\head-fixed\\pycontrol'

root_folders = [photo_root_dir, pycontrol_root_dir]
horizontal_folder_pycontrol = 'T:\\Data\\head-fixed\\test_folder\\pycontrol'
horizontal_folder_photometry = 'T:\\Data\\head-fixed\\test_folder\\photometry'

copy_files_to_horizontal_folders(root_folders, horizontal_folder_pycontrol, horizontal_folder_photometry)


# ### Create an experiment object
# 
# This will include all the pycontrol files present in the folder_path directory (do not include subdirectories)

# In[28]:


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

# In[29]:


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

# In[30]:


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

# In[31]:


import plotly.graph_objects as go


# %TODO
# - drowdown to change time units
# 

# In[32]:


exp_cohort.sessions[0].plot_session()


# In[33]:


session = exp_cohort.sessions[50]


# In[34]:


exp_cohort.sessions[0].plot_session(state_def=dict(
    name='trial', onset='CS_Go', offset='refrac_period'))


# In[35]:


# Line with a gap

fig = go.Figure()

line1 = go.Line(x=[1, 2, nan, 3, 5, nan, 7, 10], y=['hoge']*8,
                name='hoge', mode='lines', line=dict(width=10))
fig.add_trace(line1)


# In[36]:


from plotly.graph_objects.scatter import Marker


# In[37]:


# ev_array = np.array(session.df_events.loc[(trial), event_col])
event_cols = [event_col for event_col in session.df_events.columns if '_trial_time' in event_col]
event_names = [event_col.split('_trial_time')[0] for event_col in event_cols]




print(ev_trial_nb.shape, ev_times.shape)


# In[ ]:


session.df_events[event_cols[0]].apply(lambda x: np.array(x)).values


# In[ ]:


plot_names =  [trig + ' ' + event for event in session.events_to_process for trig in session.triggers]
plot_names


# In[38]:


from plotly.validators.scatter.marker import SymbolValidator
from plotly.subplots import make_subplots

def plot_trials(session):

    raw_symbols  = SymbolValidator().values
    symbols = [raw_symbols[i+2] for i in range(0, len(raw_symbols), 12)]

    event_cols = [event_col for event_col in session.df_events.columns if '_trial_time' in event_col]
    event_names = [event_col.split('_trial_time')[0] for event_col in event_cols]

    plot_names =  [trig + ' ' + event for event in session.events_to_process for trig in session.triggers]

    fig = make_subplots(
        rows= len(event_cols), 
        cols= len(session.triggers), 
        shared_xaxes= True,
        subplot_titles= plot_names
    )

    for trig_idx, trigger in enumerate(session.triggers):
        
        # sub-selection of df_events based on trigger, should be condition for event_dataset class
        df_subset = session.df_events[session.df_events.trigger == trigger]

        for ev_idx, event_col in enumerate(event_cols):

            ev_times = df_subset[event_cols[0]].apply(lambda x: np.array(x)).values
            ev_trial_nb = [np.ones(len(array)) * df_subset.index[idx] for idx, array in enumerate(ev_times)]

            ev_trial_nb = np.concatenate(ev_trial_nb)
            ev_times =  np.concatenate(ev_times)

            fig.add_trace(
                go.Scatter(
                    x=ev_times/1000,
                    y=ev_trial_nb,
                    name=event_names[ev_idx],
                    mode='markers',
                    marker_symbol=symbols[ev_idx % 40]

                ), row= ev_idx+1, col = trig_idx+1)

            fig.update_xaxes(
                ticks="inside",
                ticklen=6,
                tickwidth=2,
                tickfont_size=12,
                range=[session.trial_window[0]/1000, session.trial_window[1]/1000],
                showline=True,
                linecolor='black'
                )
            
            fig.update_yaxes(    
                ticks="inside",
                ticklen=6,
                tickwidth=2,   
                tickfont_size=12,
                showline=True,
                linecolor='black'
                )


    fig.update_layout(

        height=800,
        width=600,
    )

    fig.show()
            


# In[ ]:





# In[39]:


session.plot_trials()


# In[40]:


exp_cohort.sessions[50].df_events


# In[ ]:




