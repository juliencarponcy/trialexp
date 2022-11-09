#!/usr/bin/env python
# coding: utf-8

# # Visualise a trial PETH using matplotlib
# 
# ```bash
# jupyter nbconvert "D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical\nb20221101_115800_event_plotraster.ipynb" --to="python" --output-dir="D:\OneDrive - Nexus365\Private_Dropbox\Projects\trialexp\notebooks\noncanonical" --output="nb20221101_115800_event_plotraster"
# ```
# 

# ### Imports

# In[45]:


# allow for automatic reloading of classes and function when updating the code
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Import Session and Experiment class with helper functions
from trialexp.process.data_import import *


# ### Variables

# In[ ]:


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

# In[ ]:


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

# In[ ]:


photo_root_dir = 'T:\\Data\\head-fixed\\pyphotometry\\data'
pycontrol_root_dir = 'T:\\Data\\head-fixed\\pycontrol'

root_folders = [photo_root_dir, pycontrol_root_dir]
horizontal_folder_pycontrol = 'T:\\Data\\head-fixed\\test_folder\\pycontrol'
horizontal_folder_photometry = 'T:\\Data\\head-fixed\\test_folder\\photometry'

copy_files_to_horizontal_folders(root_folders, horizontal_folder_pycontrol, horizontal_folder_photometry)


# ### Create an experiment object
# 
# This will include all the pycontrol files present in the folder_path directory (do not include subdirectories)

# In[ ]:


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


# ### Perform extraction of behavioural information by trial (SLOW)
# 
# 5m55.4s

# In[ ]:


# Process the whole experimental folder by trials
exp_cohort.process_exp_by_trial(trial_window, timelim, tasksfile, blank_spurious_event='spout', blank_timelim=[0, 65])

# Save the file as sessions.pkl in folder_path
# exp_cohort.save() # Do I need to save this???


# ### Match with photometry, videos, and DeepLabCut files (SLOW)
# 
# The following Warning : 
# 
# KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads...
# 
# is due to rsync function for photometry-pycontrol alignment
# 
# 2m10.9s
# 

# In[ ]:


# Find if there is a matching photometry file and if it can be used:
# rsync synchronization pulses matching between behaviour and photometry
from copy import deepcopy

# Find if there is a matching photometry file:
exp_cohort.match_sessions_to_files(photometry_dir, ext='ppd')

# rsync synchronization pulses matching between behaviour and photometry
exp_cohort.sync_photometry_files(2)

# Find matching videos
exp_cohort.match_sessions_to_files(video_dir, ext='mp4')

# FInd matching DeepLabCut outputs files
exp_cohort.match_sessions_to_files(video_dir, ext='h5', verbose=True)


exp_cohort.save()




# In[ ]:


# Many combinations possible
conditions_dict0 = {'trigger': 'cued', 'valid': True, 'reward spout cued': True, 'free_reward_timer': False, 'success': True}
conditions_dict1 = {'trigger': 'cued', 'valid': True, 'reward bar cued': True, 'free_reward_timer': False, 'success': True}
conditions_dict2 = {'trigger': 'cued', 'valid': True, 'reward free': True, 'success': True}
conditions_dict3 = {'trigger': 'cued', 'valid': True, 'success': False}
conditions_dict4 = {'trigger': 'uncued', 'valid': True, 'reward spout uncued': True, 'free_reward_timer': False, 'success': True}
conditions_dict5 = {'trigger': 'uncued', 'valid': True, 'reward bar uncued': True, 'free_reward_timer': False, 'success': True}
conditions_dict6 = {'trigger': 'uncued', 'valid': True, 'reward free_uncued': True} # reward after [20, 30] s

# Aggregate all condition dictionaries in a list
condition_list = [conditions_dict0, conditions_dict1, conditions_dict2, conditions_dict3, \
                  conditions_dict4, conditions_dict5, conditions_dict6]
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


# # Load saved Experiment
# 
# 15 s

# In[ ]:


exp_cohort = Experiment(pycontrol_files_path)


# In[ ]:


exp_cohort.sessions = [session for session in exp_cohort.sessions
                       if (session.subject_ID in [47, 48, 49, 51, 53]) and (session.number > 2) 
                       and (session.task_name == 'reaching_go_spout_cued_uncued')]

ev_dataset = exp_cohort.behav_events_to_dataset(
    groups=groups,
    conditions_list=condition_list,
    cond_aliases=cond_aliases,
    when='all',
    task_names='reaching_go_spout_cued_uncued',
    trig_on_ev=None)

ev_dataset.set_trial_window(trial_window=trial_window, unit='milliseconds')
ev_dataset.set_conditions(conditions=condition_list, aliases=cond_aliases)


# ## Visualise a trial PETH using matplotlib

# %TODO
# - drowdown to change time units
# 

# In[ ]:


print(len(ev_dataset.metadata_df['keep']))

print(np.count_nonzero(ev_dataset.metadata_df['keep'] == True))


# In[ ]:


ev_dataset.metadata_df.columns


# In[ ]:


dates = ev_dataset.metadata_df['datetime'].apply( lambda x: x.date()  )
set(dates)


# In[ ]:


tf = (ev_dataset.get_tfkeep_subjects(47)) & (ev_dataset.get_tfkeep_dates(date(2022,9,26)))

np.count_nonzero(tf)


# In[ ]:


set(ev_dataset.metadata_df.session_nb[tf])


# In[ ]:


ev_dataset.set_keep(tf)


# In[ ]:


ev_dataset.triggers


# In[ ]:


ev_dataset.data.head()


# In[ ]:


from  matplotlib import pyplot as plt

plt.rcParams['font.family'] = ['Arial']

event_cols = [
    event_col for event_col in ev_dataset.data.columns if '_trial_time' in event_col]
    
event_name_stems = [event_col.split('_trial_time')[0] for event_col in event_cols]

triggers = ev_dataset.triggers

cm = 1/2.54  # centimeters in inches
fig, ax = plt.subplots(len(event_cols), len(triggers), sharex=True, sharey=True, figsize=(21.0*cm, 29.7*cm))

for trig_idx, trigger in enumerate(triggers):

    df_subset = ev_dataset.data.loc[(ev_dataset.data['trigger'] == trigger) & (ev_dataset.metadata_df['keep']) , :] #only include keep

    df_subset = df_subset.reset_index()

    for ev_idx, event_col in enumerate(event_cols):
        
        for r in range(0,df_subset.shape[0]):
           
            ev_times = df_subset.at[r, event_col]
            
            X = np.array(ev_times)
            X.shape = (1, len(X))
            X = np.tile(X,(2,1))/1000 # ms

            Y = np.array([r, r+1])
            Y.shape = (2,1)
            Y = np.tile(Y, (1, X.shape[1]))

            ax[ev_idx][trig_idx].plot(X, Y, 'k-', linewidth = 0.5)

            plot_names = trigger + ' ' + event_col

            event_name_stem = event_col.split('_trial_time')[0] 

            ax[ev_idx][trig_idx].set_ylabel('Trials: ' + event_name_stem)

            ax[ev_idx][trig_idx].spines['top'].set_visible(False)
            ax[ev_idx][trig_idx].spines['right'].set_visible(False)

    ax[0][trig_idx].set_title(trigger)

    ax[ev_idx][trig_idx].set_xlabel('Time (s)')




# # Event_Dataset.plot_raster()

# In[ ]:


ev_dataset.plot_raster()



# In[55]:


ev_dataset.plot_raster(separate=False)


# In[ ]:


# ev_array = np.array(session.df_events.loc[(trial), event_col])
event_cols = [event_col for event_col in session.df_events.columns if '_trial_time' in event_col]
event_names = [event_col.split('_trial_time')[0] for event_col in event_cols]




print(ev_trial_nb.shape, ev_times.shape)


# In[ ]:


session.df_events[event_cols[0]].apply(lambda x: np.array(x)).values


# In[ ]:


plot_names =  [trig + ' ' + event for event in session.events_to_process for trig in session.triggers]
plot_names


# # Plotly

# In[ ]:


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





# In[ ]:


session.plot_trials()


# In[ ]:


exp_cohort.sessions[50].df_events


# In[ ]:


pyc_dataset = Event_Dataset(exp_cohort.sessions[50].df_events, exp_cohort.sessions[50].df_conditions)
pyc_dataset.plot_trials()


# In[ ]:


pyc_dataset.metadata_df


# In[ ]:




