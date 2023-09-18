#!/usr/bin/env python
# coding: utf-8

# In[63]:


import os

nb_name = "nb20230815_155800_ACh_cueonset_response_change.ipynb" #TODO change this

basename, ext = os.path.splitext(nb_name)
input_path = os.path.join(os.getcwd(), nb_name)

get_ipython().system('jupyter nbconvert "{input_path}" --to="python" --output="{basename}"')


# based on notebooks\noncanonical\nb20230626_190400_ACh_cueonset_response_change.ipynb
# 
# based on notebooks\noncanonical\nb20230622_215600_ACh_cueonset_2_outcomes.ipynb
# 
# - animals?
# - troughs for movement and reward?
# 
# 
# - Calculate the CC and linregress slope for ACh dip, rebound, and DA peak against trial_nb
# - We should be able to find sessions in which ACh rebound stays while ACh dip goes away.
# - By detecting slow trough in data, we could also find sessions with recovery
# - Should we also analyse data around spount? Movement encoding may be more stable.
# - We can use `lme4` for ACh dip etc.
# 
# 
# - scatter plots for `trial_nb` against dip size etc 
# - dip size
#     - selected_data = xr_photometry['hold_for_water_zscored_df_over_f'].sel(trial_nb=k, event_time=slice(150, 300))
# - rebound size
#     - selected_data = xr_photometry['hold_for_water_zscored_df_over_f'].sel(trial_nb=k, event_time=slice(350, 700))
# 

# \\ettina\Magill_Lab\Julien\Data\head-fixed\by_sessions\reaching_go_spout_bar_nov22\TT002-2023-06-22-111549\processed
# 
# - lots of abortions
# 

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import re
from matplotlib import pyplot as plt
import itertools
import seaborn as sns
import patchworklib as pw


from trialexp.process.pyphotometry.utils import *
from trialexp.process.pycontrol import event_filters
from trialexp.process.pycontrol.event_filters import extract_event_time
from trialexp.process.pyphotometry.utils import measure_ACh_dip_rebound, measure_DA_peak



# by_sessions_dir = r'\\ettina\Magill_Lab\Julien\Data\head-fixed\by_sessions'
# task_dir = os.path.join(by_sessions_dir,  'reaching_go_spout_bar_nov22')
# task_dir = os.path.join(by_sessions_dir,  'reaching_go_spout_bar_nov22')
# data_dir = os.path.join(task_dir, 'TT002-2023-06-05-154932', 'processed')

# xr_photometry = xr.open_dataset(os.path.join(data_dir, 'xr_photometry.nc'))
# xr_session = xr.open_dataset(os.path.join(data_dir, 'xr_session.nc'))
# df_pycontrol = pd.read_pickle(os.path.join(data_dir, 'df_pycontrol.pkl'))
# df_events = pd.read_pickle(os.path.join(data_dir, 'df_events_cond.pkl'))


# In[ ]:


by_sessions_dir = r'\\ettina\Magill_Lab\Julien\Data\head-fixed\by_sessions'
task_dir = os.path.join(by_sessions_dir,  'reaching_go_spout_bar_nov22')

items = os.listdir(task_dir)
data_dirs = [os.path.join(task_dir, item, 'processed') for item in items if os.path.isdir(os.path.join(task_dir, item))]
session_ids = [item for item in items if os.path.isdir(os.path.join(task_dir, item))]
subject_ids = [re.match(r"(\w+)-", ssid).group(1) for ssid in session_ids]


# In[ ]:


## Test data

# data_dirs = [os.path.join(
#     task_dir, 'TT002-2023-06-05-154932', 'processed')]
# session_ids = [item for item in items if os.path.isdir(os.path.join(task_dir, item))]
# subject_ids = ['TT002']


# # Compute ACh
# 3 m 47 s for the folder 'reaching_go_spout_bar_nov22' and the 5 mice

# In[ ]:


subject_ids_ACh = ['TT001','TT002','TT005','RE606', 'RE607']

ind_ACh = [ind for ind, sbj in enumerate(subject_ids) if sbj in subject_ids_ACh]


# In[ ]:


data = []

for dd, ss, sj in zip([data_dirs[i] for i in ind_ACh], [session_ids[i] for i in ind_ACh], [subject_ids[i] for i in ind_ACh]):

    df_trials, lin_regress_dip, lin_regress_rebound, lin_regress_dip_rebound, \
        is_success, msg = measure_ACh_dip_rebound(dd)
    n_trials = np.nan
    if isinstance(df_trials, pd.DataFrame):
        n_trials = df_trials.shape[0]
    row_data_list = [ss] + [sj] + [df_trials] + [n_trials] + list(lin_regress_dip.values()) + list(
        lin_regress_rebound.values()) + list(lin_regress_dip_rebound.values()) + [is_success] + [msg] + [dd]
    data.append(row_data_list)

df_ACh_cue_onset = pd.DataFrame(data)

df_ACh_cue_onset.columns = ['session_id', 'subject_id', 'df_trials', 'n_trials',
              'trial_nb_dip_slope', 'trial_nb_dip_intercept', 'trial_nb_dip_r_value', 'trial_nb_dip_p_value', 'trial_nb_dip_std_er',
              'trial_nb_rebound_slope', 'trial_nb_rebound_intercept', 'trial_nb_rebound_r_value', 'trial_nb_rebound_p_value', 'trial_nb_rebound_std_er',
              'dip_rebound_slope', 'dip_rebound_intercept', 'dip_rebound_r_value', 'dip_rebound_p_value', 'dip_rebound_std_er',
              'is_success', 'msg', 'data_dir']


# In[ ]:


mask = (df_ACh_cue_onset['n_trials'].notnull()) & (df_ACh_cue_onset['n_trials'] > 100) & df_ACh_cue_onset['is_success']
df_ACh_cue_onset_100 = df_ACh_cue_onset.loc[mask]

df_ACh_cue_onset_100['n_trials']


# # Compute DA
# 

# In[ ]:


subject_ids_DA = ['kms058','kms062','kms063','kms064', 'JC317L']

ind_DA = [ind for ind, sbj in enumerate(subject_ids) if sbj in subject_ids_DA]


# In[ ]:


data = []

for dd, ss, sj in zip([data_dirs[i] for i in ind_DA], [session_ids[i] for i in ind_DA], [subject_ids[i] for i in ind_DA]):

    df_trials, lin_regress_pk, \
        is_success, msg = measure_DA_peak(dd)
    n_trials = np.nan
    if isinstance(df_trials, pd.DataFrame):
        n_trials = df_trials.shape[0]
    row_data_list = [ss] + [sj] + [df_trials] + [n_trials] + list(lin_regress_pk.values()) + [is_success] + [msg] + [dd]
    data.append(row_data_list)

df_DA_cue_onset = pd.DataFrame(data)

df_DA_cue_onset.columns = ['session_id', 'subject_id', 'df_trials', 'n_trials',
                            'trial_nb_pk_slope', 'trial_nb_pk_intercept', 'trial_nb_pk_r_value', 'trial_nb_pk_p_value', 'trial_nb_pk_std_er',
                            'is_success', 'msg', 'data_dir']


# #'TT002-2023-06-05-154932',
# 
# print(df_ACh_cue_onset.trial_nb_dip_r_value)
# print(df_ACh_cue_onset.trial_nb_rebound_r_value)
# print(df_ACh_cue_onset.dip_rebound_r_value)
# 
# 0    0.525112
# Name: trial_nb_dip_r_value, dtype: float64
# 0   -0.29187
# Name: trial_nb_rebound_r_value, dtype: float64
# 0    0.001605
# Name: dip_rebound_r_value, dtype: float64
# 

# In[ ]:


mask = (df_DA_cue_onset['n_trials'].notnull()) & (
    df_DA_cue_onset['n_trials'] > 100) & df_DA_cue_onset['is_success']
df_DA_cue_onset_100 = df_DA_cue_onset.loc[mask]

df_DA_cue_onset_100['n_trials']


# ## Plotting style

# In[ ]:


# Define your list of markers
markers = itertools.cycle(
    ('o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'))

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams["legend.frameon"] = False
plt.rcParams['xtick.bottom']=True
plt.rcParams['ytick.left']=True
plt.rcParams['font.family']= 'Arial'

plt.rcParams['axes.labelsize'] = 12


# # ACh

# In[ ]:


fig, ax = plt.subplots()

subject_ids_ = sorted(list(set(df_ACh_cue_onset_100['subject_id'])))

for sbj in subject_ids_:
    x = - 1 * df_ACh_cue_onset_100['trial_nb_dip_r_value'][df_ACh_cue_onset_100['subject_id'] == sbj]
    y = df_ACh_cue_onset_100['trial_nb_rebound_r_value'][df_ACh_cue_onset_100['subject_id'] == sbj]

    ax.plot(x, y, marker=next(markers), linestyle='None', fillstyle='none', label = sbj)

plt.axis('equal')

# ax.set_xlim()
# ax.set_ylim()

ax.plot([-0.8, 0.4], [0, 0], '--k')
ax.plot([0, 0], [-0.5, 0.5],  '--k')

ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('CC of ACh dip size and trial numbers')
plt.ylabel('CC of ACh rebound size and trial numbers')

# Negative CCs mean the absolute size of dip and rebound is reducing

ax.text(0.4, 0.5, 'Increasing in size', ha = 'right')
ax.text(-0.8, -0.5, 'Decreasing in size', ha = 'left')


# In[ ]:


np.count_nonzero((df_ACh_cue_onset_100['trial_nb_dip_r_value'] * -1 < 0.1) &
                 (df_ACh_cue_onset_100['trial_nb_rebound_r_value'] > 0.1))


# In[ ]:


# find sessions with CC for dip > 0.1, CC for rebound > 0.1

ss_dp = df_ACh_cue_onset_100.loc[(df_ACh_cue_onset_100['trial_nb_dip_r_value'] * -1 < -0.2) & 
                          (df_ACh_cue_onset_100['trial_nb_rebound_r_value'] > 0.2) , 'session_id']

ss_dp



# In[ ]:


ss_rdp = df_ACh_cue_onset_100.loc[(df_ACh_cue_onset_100['trial_nb_rebound_r_value'] > 0.2), 'session_id']

ss_rdp


# In[ ]:


ss_dd = df_ACh_cue_onset_100.loc[(df_ACh_cue_onset_100['trial_nb_dip_r_value'] * -1 < -0.2) &
                                 (df_ACh_cue_onset_100['trial_nb_rebound_r_value'] < -0.2), 'session_id']
ss_dd


# In[ ]:


fig, ax = plt.subplots()

subject_ids_ = sorted(list(set(df_ACh_cue_onset_100['subject_id'])))

for sbj in subject_ids_:
    x = -1 * np.mean(df_ACh_cue_onset_100['trial_nb_dip_r_value'][df_ACh_cue_onset_100['subject_id'] == sbj])
    y = np.mean(df_ACh_cue_onset_100['trial_nb_rebound_r_value'][df_ACh_cue_onset_100['subject_id'] == sbj])

    ax.plot(x, y, marker=next(markers), linestyle='None',
            fillstyle='none', label=sbj)


XLIM = ax.get_xlim()
ax.set_xlim([XLIM[0], 0.1])
XLIM = ax.get_xlim()

YLIM = ax.get_ylim()

ax.plot(XLIM, [0, 0], '--k')
ax.plot([0, 0], YLIM,  '--k')

ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('CC of ACh dip size and trial_nb')
plt.ylabel('CC of ACh rebound size and trial_nb')
plt.title('Average per animal')


# In[ ]:


fig, ax = plt.subplots()

subject_ids_ = sorted(list(set(df_ACh_cue_onset_100['subject_id'])))

for i, sbj in enumerate(subject_ids_):
    # minus means moving the other ways
    y = (df_ACh_cue_onset_100['trial_nb_dip_r_value']
         [df_ACh_cue_onset_100['subject_id'] == sbj]) * -1
    # y = np.mean(df_ACh_cue_onset_100['trial_nb_rebound_r_value']
    #             [df_ACh_cue_onset_100['subject_id'] == sbj])

    sns.swarmplot(x=i, y=y)

    # ax.plot(x, marker=next(markers), linestyle='None',
    #         fillstyle='none', label=sbj)


ax.plot(ax.get_xlim(), [0, 0], '--k')

plt.xticks(range(0,5), subject_ids_)
plt.ylabel('CC of ACh dip size and trial number')


# In[ ]:


fig, ax = plt.subplots()

subject_ids_ = sorted(list(set(df_ACh_cue_onset_100['subject_id'])))

for i, sbj in enumerate(subject_ids_):
    # minus means moving the other ways
    y = (df_ACh_cue_onset_100['trial_nb_rebound_r_value']
         [df_ACh_cue_onset_100['subject_id'] == sbj])
    # y = np.mean(df_ACh_cue_onset_100['trial_nb_rebound_r_value']
    #             [df_ACh_cue_onset_100['subject_id'] == sbj])

    sns.swarmplot(x=i, y=y)

    # ax.plot(x, marker=next(markers), linestyle='None',
    #         fillstyle='none', label=sbj)


ax.plot(ax.get_xlim(), [0, 0], '--k')

plt.xticks(range(0,5), subject_ids_)
plt.ylabel('CC of ACh rebound size and trial number')


# In[ ]:


fig, ax = plt.subplots()

subject_ids_ = sorted(list(set(df_ACh_cue_onset_100['subject_id'])))

for i, sbj in enumerate(subject_ids_):
    y = (df_ACh_cue_onset_100['dip_rebound_r_value'][df_ACh_cue_onset_100['subject_id'] == sbj]) * -1 # minus means moving the other ways
    # y = np.mean(df_ACh_cue_onset_100['trial_nb_rebound_r_value']
    #             [df_ACh_cue_onset_100['subject_id'] == sbj])

    sns.swarmplot(x=i, y=y)

    # ax.plot(x, marker=next(markers), linestyle='None',
    #         fillstyle='none', label=sbj)


ax.plot(ax.get_xlim(), [0, 0], '--k')

plt.xticks(range(0,5), subject_ids_)
plt.ylabel('CC of DA peak size and trial number')


# In[ ]:


sbj = 'kms058'

df_ACh_cue_onset_100.loc[df_ACh_cue_onset_100['subject_id'] == sbj].index
 


# In[ ]:


subject_ids_ = sorted(list(set(df_ACh_cue_onset_100['subject_id'])))

for sbj in subject_ids_:

    fig, ax = plt.subplots()

    df_ACh_cue_onset_100['subject_id'] == sbj

    ax.set_title(sbj)

    ind = df_ACh_cue_onset_100.loc[df_ACh_cue_onset_100['subject_id'] == sbj].index

    for i in ind:
        
        ax.plot(df_ACh_cue_onset_100['df_trials'][i]['trial_nb'], 
                df_ACh_cue_onset_100['df_trials'][i]['dip'], color = '#1f77b4', alpha = 0.2)
        ax.plot(df_ACh_cue_onset_100['df_trials'][i]['trial_nb'],
                df_ACh_cue_onset_100['df_trials'][i]['rebound'], color='#ff7f0e', alpha = 0.2)
        
        ax.set_xlabel('Trial number')
        ax.set_ylabel('Dip/rebound size in z-scored delta F/F')


# # DA

# In[ ]:


import seaborn as sns

fig, ax = plt.subplots()

subject_ids_ = sorted(list(set(df_DA_cue_onset_100['subject_id'])))

for i, sbj in enumerate(subject_ids_):
    y = (df_DA_cue_onset_100['trial_nb_pk_r_value']
         [df_DA_cue_onset_100['subject_id'] == sbj])
    # y = np.mean(df_DA_cue_onset_100['trial_nb_rebound_r_value']
    #             [df_DA_cue_onset_100['subject_id'] == sbj])

    sns.swarmplot(x=i, y=y)

    # ax.plot(x, marker=next(markers), linestyle='None',
    #         fillstyle='none', label=sbj)


ax.plot(ax.get_xlim(), [0, 0], '--k')

plt.xticks(range(0,5), subject_ids_)


# In[ ]:


subject_ids_ = sorted(list(set(df_DA_cue_onset_100['subject_id'])))

for sbj in subject_ids_:

    fig, ax = plt.subplots()

    df_DA_cue_onset_100['subject_id'] == sbj

    ax.set_title(sbj)

    ind = df_DA_cue_onset_100.loc[df_DA_cue_onset_100['subject_id'] == sbj].index

    for i in ind:
        
        ax.plot(df_DA_cue_onset_100['df_trials'][i]['trial_nb'], 
                df_DA_cue_onset_100['df_trials'][i]['peak'], color = '#1f77b4', alpha = 0.2)
        
        ax.set_xlabel('Trial number')
        ax.set_ylabel('Peak size in z-scored delta F/F')


# 
# # Moving average
# 
# Use moving average and detection of slow peak or trough in response size
# 
# ```python
# data['value_smooth'] = data['value'].rolling(window=5).mean()
# 
# ```
# 
# I hoped to detect large scale change in response sizes, eg. a trough followed by rebound, implying they stopped performing and then engaged in the task again.
# 
# It's certainly less noisy, but 

# In[ ]:


subject_ids_ = sorted(list(set(df_ACh_cue_onset_100['subject_id'])))

for sbj in subject_ids_:

    ind = df_ACh_cue_onset_100.loc[df_ACh_cue_onset_100['subject_id'] == sbj].index

    for i in ind:
        df_ACh_cue_onset_100.loc[i, 'df_trials']['dip_smooth20'] = \
            df_ACh_cue_onset_100.loc[i, 'df_trials']['dip'].rolling(window=20).mean()
        df_ACh_cue_onset_100.loc[i, 'df_trials']['rebound_smooth20'] = \
            df_ACh_cue_onset_100.loc[i, 'df_trials']['rebound'].rolling(window=20).mean()


# In[ ]:


plt.rcParams['font.size']= 16
plt.rcParams['axes.labelsize'] = 20

subject_ids_ = sorted(list(set(df_ACh_cue_onset_100['subject_id'])))

ax = [None]*6
for j, sbj in enumerate(subject_ids_):

    ax[j] = pw.Brick(figsize=(6, 6))


    ax[j].set_title(sbj + ": after smoothing")

    ind = df_ACh_cue_onset_100.loc[df_ACh_cue_onset_100['subject_id'] == sbj].index

    for i in ind:

        ax[j].plot(df_ACh_cue_onset_100['df_trials'][i]['trial_nb'],
                df_ACh_cue_onset_100['df_trials'][i]['dip_smooth20'], color='#1f77b4', alpha=0.2)
        ax[j].plot(df_ACh_cue_onset_100['df_trials'][i]['trial_nb'],
                df_ACh_cue_onset_100['df_trials'][i]['rebound_smooth20'], color='#ff7f0e', alpha=0.2)

        ax[j].set_xlabel('Trial number')
        ax[j].set_ylabel('Dip/rebound size in z-scored delta F/F')

ax[5] = pw.Brick(figsize=(3, 3))

ax01 = ax[0] | ax[1] 
ax23 = ax[2] | ax[3] 
ax45 = ax[4] | ax[5]
ax0123 = ax01 / ax23
ax012345 = ax0123 / ax45

ax012345.savefig()


# - trial_outcome
# - cueonset
# - spout
# 
# 

# In[ ]:


ss_d = df_ACh_cue_onset_100.loc[(df_ACh_cue_onset_100['trial_nb_dip_r_value'] * -1 < -0.3), 'session_id']
ss_d


# In[ ]:


ss = 'TT001-2023-06-23-154127'
datadir = r'\\ettina\Magill_Lab\Julien\Data\head-fixed\by_sessions\reaching_go_spout_bar_nov22' + '\\' + ss + r'\processed'

xr_photometry = xr.open_dataset(os.path.join(datadir, 'xr_photometry.nc'))
xr_session = xr.open_dataset(os.path.join(datadir, 'xr_session.nc'))


# In[ ]:


from scipy.stats import pearsonr
ind_success = np.where(xr_session['trial_outcome'].values == 'success')[1] + 1

dip_suc = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
    event_time=slice(75, 250), trial_nb=ind_success).min(dim='event_time').values
dip_suc_r, dip_suc_p = pearsonr(dip_suc, np.arange(1, len(dip_suc) +1))


# In[ ]:


dip_suc_r


# In[ ]:


os.path.isfile(os.path.join(datadir, 'xr_photometry.nc'))


# In[ ]:


dip_suc = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
    event_time=slice(75, 250), trial_nb=ind_success).min(dim='event_time').values


# In[ ]:


#  Calculate CC or slope

from scipy.stats import pearsonr

if 'list_size3' in globals():
    del list_size3

list_size3 = []
for ss in ss_d:

    datadir = r'\\ettina\Magill_Lab\Julien\Data\head-fixed\by_sessions\reaching_go_spout_bar_nov22' + '\\' + ss + r'\processed'

    try:
        xr_photometry = xr.open_dataset(os.path.join(datadir, 'xr_photometry.nc'))
        xr_session = xr.open_dataset(os.path.join(datadir, 'xr_session.nc'))
        
        # need to select trials for success and 

        ind_success = np.where(xr_session['trial_outcome'].values == 'success')[1] + 1

        trial_nbs = xr_session['trial_nb'].values

        xr_photometry['hold_for_water_zscored_df_over_f'].sel(
            event_time=slice(75, 250), trial_nb=ind_success).min(dim='event_time').values


        ind_success = np.where(xr_session['trial_outcome'].values == 'success')[1] + 1

        dip_suc = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
            event_time=slice(75, 250), trial_nb=ind_success).min(dim='event_time').values
        dip_suc = dip_suc[~np.isnan(dip_suc)]
        dip_suc_r, dip_suc_p = pearsonr(-1 * dip_suc, range(1, len(dip_suc) +1))
        # Y could be trial_nb, range(1, len(x) +1), or timestamps of cue

        reb_suc = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
            event_time=slice(200, 600), trial_nb=ind_success).max(dim='event_time').values
        reb_suc = reb_suc[~np.isnan(reb_suc)]
        reb_suc_r, reb_suc_p = pearsonr(reb_suc, range(1, len(reb_suc) + 1))

        lbo_suc = xr_photometry['last_bar_off_zscored_df_over_f'].sel(
            event_time=slice(0, 150), trial_nb=ind_success).max(dim='event_time').values
        lbo_suc = lbo_suc[~np.isnan(lbo_suc)]
        lbo_suc_r, lbo_suc_p = pearsonr(lbo_suc, range(1, len(lbo_suc) + 1))

        rew_suc = xr_photometry['first_spout_zscored_df_over_f'].sel(
            event_time=slice(500, 750), trial_nb=ind_success).max(dim='event_time').values
        rew_suc = rew_suc[~np.isnan(rew_suc)]
        rew_suc_r, rew_suc_p = pearsonr(rew_suc, range(1, len(rew_suc) + 1))

        items = [ss, dip_suc, reb_suc, lbo_suc, rew_suc, 
                dip_suc_r, reb_suc_r, lbo_suc_r, rew_suc_r]
    except Exception as e: 
        print(f'{ss}, {e}')
        items = [ss, np.nan, np.nan, np.nan, np.nan, 
                np.nan, np.nan, np.nan, np.nan]

    df_size3 = pd.DataFrame([items])
    df_size3.columns = ['session_id', 'dip_suc', 'reb_suc', 'lbo_suc', 'rew_suc', 
                  'dip_suc_r', 'reb_suc_r', 'lbo_suc_r', 'rew__suc_r']
    list_size3.append(df_size3)




# In[ ]:


df_size3_ = pd.concat(list_size3, axis=0)

df_size3_ = df_size3_.dropna(subset=['dip_suc_r'])

df_size3_['subject_id'] = [re.search('\w+', sid).group(0) for sid in df_size3_['session_id']]
df_size3_


# In[46]:


df_size3_


# In[48]:


df_size3__ = df_size3_.loc[:,  ['dip_suc_r', 'reb_suc_r', 'lbo_suc_r', 'rew__suc_r']]

df_melted = df_size3__.melt(var_name='group', value_name='value')


# In[56]:


ax.get_xlim()


# In[60]:


ax.get_xlim()


# In[62]:


plt.figure(figsize=(10, 6))
palette = sns.color_palette('deep', 4)
sns.swarmplot(x='group', y='value', data=df_melted, palette=palette,
              zorder=1)  # zorder=1 to be underneath the boxplot
# zorder=2 to be over the swarmplot
sns.boxplot(x='group', y='value', data=df_melted, zorder=2,
            boxprops=dict(alpha=.3), palette=palette, width=0.4)
plt.axhline(0, ls='--', color='gray')
plt.ylabel('Correlation coefficient of \nresponse size and successful trial ordinal\nDecreasing — Increasing')
plt.xticks(range(0, 4), ['Cue onset dip', 'Cue onset rebound',
           'Last bar_off peak', 'Reward peak'], rotation=60)


# # Early, middle, late .... difficult to get statistical significance because of large variance

# In[ ]:


list_size3 = []
for ss in ss_d:

    datadir = r'\\ettina\Magill_Lab\Julien\Data\head-fixed\by_sessions\reaching_go_spout_bar_nov22' + '\\' + ss + r'\processed'

    xr_photometry = xr.open_dataset(os.path.join(datadir, 'xr_photometry.nc'))
    xr_session = xr.open_dataset(os.path.join(datadir, 'xr_session.nc'))
    
    # need to select trials for success and 

    trial_nbs = xr_session['trial_nb'].values

    first_30_index = int(len(trial_nbs) * 0.20)
    last_30_index = int(len(trial_nbs) * 0.80)

    first_30 = trial_nbs[:first_30_index]
    last_30 = trial_nbs[last_30_index:]

    start_middle_45_index = int(len(trial_nbs) * 0.45)
    end_middle_65_index = int(len(trial_nbs) * 0.65)
    middle_30 = trial_nbs[start_middle_45_index:end_middle_65_index]

    ind_success = np.where(xr_session['trial_outcome'].values == 'success')[1] + 1

    first_30_index  = list(set(first_30) & set(ind_success))
    first_30_index.sort()

    middle_30_index  = list(set(middle_30) & set(ind_success))
    middle_30_index.sort()

    last_30_index  = list(set(last_30) & set(ind_success))
    last_30_index.sort()

    if first_30_index: 
        dip_1 = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
            event_time=slice(75, 250), trial_nb=first_30_index).min(dim='event_time').values
    else:
        dip_1 = []

    if middle_30_index: 
        dip_2 = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
            event_time=slice(75, 250), trial_nb=middle_30_index).min(dim='event_time').values
    else:
        dip_2 = []

    if last_30_index: 
        dip_3 = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
            event_time=slice(75, 250), trial_nb=last_30_index).min(dim='event_time').values
    else:
        dip_3 = []

    if first_30_index : 
        reb_1 = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
            event_time=slice(200, 600), trial_nb=first_30_index).max(dim='event_time').values
    else:
        reb_1 = []
    
    if middle_30_index: 
        reb_2 = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
            event_time=slice(200, 600), trial_nb=middle_30_index).max(dim='event_time').values
    else:
        reb_2 = []

    if last_30_index:
        reb_3 = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
            event_time=slice(200, 600), trial_nb=last_30_index).max(dim='event_time').values
    else:
        dip_3 = []

    if first_30_index:
        lbo_1 = xr_photometry['last_bar_off_zscored_df_over_f'].sel(
            event_time=slice(0, 150), trial_nb=first_30_index).max(dim='event_time').values
    else:
        lbo_1 = []

    if middle_30_index:
        lbo_2 = xr_photometry['last_bar_off_zscored_df_over_f'].sel(
            event_time=slice(0, 150), trial_nb=middle_30_index).max(dim='event_time').values
    else:
        lbo_2 = []

    if last_30_index:
        lbo_3 = xr_photometry['last_bar_off_zscored_df_over_f'].sel(
            event_time=slice(0, 150), trial_nb=last_30_index).max(dim='event_time').values
    else:
        lbo_3 = []

    if first_30_index:
        rew_1 = xr_photometry['first_spout_zscored_df_over_f'].sel(
            event_time=slice(500, 750), trial_nb=first_30_index).max(dim='event_time').values
    else:
        rew_1 = []

    if middle_30_index:
        rew_2 = xr_photometry['first_spout_zscored_df_over_f'].sel(
            event_time=slice(500, 750), trial_nb=middle_30_index).max(dim='event_time').values
    else:
        rew_2 = []

    if last_30_index:
        rew_3 = xr_photometry['first_spout_zscored_df_over_f'].sel(
            event_time=slice(500, 750), trial_nb=last_30_index).max(dim='event_time').values
    else:
        rew_3 = []
    
    items = [ss, dip_1, dip_2, dip_3, reb_1, reb_2, reb_3, lbo_1, lbo_2, lbo_3, rew_1, rew_2, rew_3]
    df_size3 = pd.DataFrame([items])
    df_size3.columns = ['session_nb','dip_1' ,'dip_2', 'dip_3', 'reb_1', 'reb_2', 'reb_3',
                  'lbo_1', 'lbo_2', 'lbo_3', 'rew_1', 'rew_2', 'rew_3']
    list_size3.append(df_size3)


# In[ ]:


df_size3 = pd.concat(list_size3, axis=0)


# In[ ]:


y = df_size3['dip_1']
y = [item for sublist in df_size3['dip_1'] for item in sublist]


# In[ ]:


df_size3.columns


# In[ ]:


def plot_one(key, k):
    y = [item for sublist in df_size3[key] for item in sublist]
    y = [x for x in y if not np.isnan(x)]
    # sns.swarmplot(x=1, y=y)
    bp = ax.boxplot(y, positions=[k])


    # Change the box size
    for box in bp['boxes']:
        box.set(linewidth=2)

    # Change the median line color
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    # Change outlier size
    for flier in bp['fliers']:
        flier.set(marker='o', color='red', alpha=0.5, markersize=5)


fig, ax = plt.subplots()

keys = ['dip_1', 'dip_2', 'dip_3', 'reb_1', 'reb_2', 'reb_3',
       'lbo_1', 'lbo_2', 'lbo_3', 'rew_1', 'rew_2', 'rew_3']

for i, key in enumerate(keys):
    plot_one(key, i);

keys2 = ['dip early', 'dip middle', 'dip late', 
         'rebound early', 'rebound middle', 'rebound late',
       'last bar_off early', 'last bar_off middle', 'ast bar_off late',
        'reward early', 'reward middle', 'reward late']
ax.set_xticklabels(keys2)
plt.xticks(rotation=90)  # Adjust font size here
plt.ylabel('Response size in z-scored delta F/F', fontsize=14)


# In[ ]:


keys[0:3]


# In[ ]:


from scipy import stats
from scipy.stats import mannwhitneyu


def stats2(keys):
    y1= [item for sublist in df_size3[keys[0]] for item in sublist]
    y1 = [x for x in y1 if not np.isnan(x)]

    # y2= [item for sublist in df_size3[keys[1]] for item in sublist]
    # y2 = [x for x in y1 if not np.isnan(x)]

    y3= [item for sublist in df_size3[keys[2]] for item in sublist]
    y3 = [x for x in y1 if not np.isnan(x)]



    # sns.swarmplot(x=1, y=y)
    stat, p = mannwhitneyu(y1, y3)

    print(stat)
    print(p)

stats2(keys[0:3])
stats2(keys[3:6])
stats2(keys[6:9])
stats2(keys[9:12])


# I guess this is because there are so much variance due to the noisy nature of the data.
