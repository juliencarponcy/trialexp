#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

nb_name = "nb20230626_190400_ACh_cueonset_response_change.ipynb" #TODO change this

basename, ext = os.path.splitext(nb_name)
input_path = os.path.join(os.getcwd(), nb_name)

get_ipython().system('jupyter nbconvert "{input_path}" --to="python" --output="{basename}"')


# based on notebooks\noncanonical\nb20230622_215600_ACh_cueonset_2_outcomes.ipynb
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

# In[1]:


import os
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from trialexp.process.pyphotometry.utils import *
from trialexp.process.pycontrol import event_filters
from trialexp.process.pycontrol.event_filters import extract_event_time
from trialexp.process.pyphotometry.utils import measure_ach_dip_rebound


# by_sessions_dir = r'\\ettina\Magill_Lab\Julien\Data\head-fixed\by_sessions'
# task_dir = os.path.join(by_sessions_dir,  'reaching_go_spout_bar_nov22')
# task_dir = os.path.join(by_sessions_dir,  'reaching_go_spout_bar_nov22')
# data_dir = os.path.join(task_dir, 'TT002-2023-06-05-154932', 'processed')

# xr_photometry = xr.open_dataset(os.path.join(data_dir, 'xr_photometry.nc'))
# xr_session = xr.open_dataset(os.path.join(data_dir, 'xr_session.nc'))
# df_pycontrol = pd.read_pickle(os.path.join(data_dir, 'df_pycontrol.pkl'))
# df_events = pd.read_pickle(os.path.join(data_dir, 'df_events_cond.pkl'))


# In[2]:


by_sessions_dir = r'\\ettina\Magill_Lab\Julien\Data\head-fixed\by_sessions'
task_dir = os.path.join(by_sessions_dir,  'reaching_go_spout_bar_nov22')

items = os.listdir(task_dir)
data_dirs = [os.path.join(task_dir, item, 'processed') for item in items if os.path.isdir(os.path.join(task_dir, item))]
session_ids = [item for item in items if os.path.isdir(os.path.join(task_dir, item))]


# In[3]:


## Test data

data_dirs = [os.path.join(
    task_dir, 'TT002-2023-06-05-154932', 'processed')]
session_ids = [item for item in items if os.path.isdir(os.path.join(task_dir, item))]

print(data_dirs)


# In[10]:


data = []

for dd, ss in zip(data_dirs, session_ids):

    df_trials, lin_regress_dip, lin_regress_rebound, lin_regress_dip_rebound, \
        is_success, msg = measure_ach_dip_rebound(dd)
    row_data_list = [ss] + [df_trials] + list(lin_regress_dip.values()) + list(
        lin_regress_rebound.values()) + list(lin_regress_dip_rebound.values()) + [is_success] + [msg] + [dd]
    data.append(row_data_list)

df_ACh_cue_onset = pd.DataFrame(data)

df_ACh_cue_onset.columns = ['session_id', 'df_trials', 
              'trial_nb_dip_slope', 'trial_nb_dip_intercept', 'trial_nb_dip_r_value', 'trial_nb_dip_p_value', 'trial_nb_dip_std_er',
              'trial_nb_rebound_slope', 'trial_nb_rebound_intercept', 'trial_nb_rebound_r_value', 'trial_nb_rebound_p_value', 'trial_nb_rebound_std_er',
              'dip_rebound_slope', 'dip_rebound_intercept', 'dip_rebound_r_value', 'dip_rebound_p_value', 'dip_rebound_std_er',
              'is_success', 'msg', 'data_dir']


# In[11]:


print(df_ACh_cue_onset.trial_nb_dip_r_value)
print(df_ACh_cue_onset.trial_nb_rebound_r_value)
print(df_ACh_cue_onset.dip_rebound_r_value)


# In[ ]:


task_dir = os.path.join(by_sessions_dir,  'reaching_go_spout_bar_nov22')
data_dir = os.path.join(task_dir, 'TT002-2023-06-05-154932', 'processed')


# xr_photometry = xr.open_dataset(os.path.join(data_dir, 'xr_photometry.nc'))
# xr_session = xr.open_dataset(os.path.join(data_dir, 'xr_session.nc'))
# df_pycontrol = pd.read_pickle(os.path.join(data_dir, 'df_pycontrol.pkl'))
# df_events = pd.read_pickle(os.path.join(data_dir, 'df_events_cond.pkl'))


# In[ ]:


from trialexp.process.pyphotometry.utils import measure_ach_dip_rebound

df_trials, lin_regress_dip, lin_regress_rebound = measure_ach_dip_rebound(data_dir)

print(lin_regress_dip)
print(lin_regress_rebound)


# In[ ]:


selected_data = xr_photometry['hold_for_water_zscored_df_over_f'].sel(event_time=slice(0, 1000))
selected_data.dims

average_curve = selected_data.mean(dim='trial_nb')


fig, ax = plt.subplots()

plt.plot(average_curve)
ax.grid(True)
plt.xticks(np.arange(0,1000,100))


# In[ ]:


trial_nb_all = int(max(xr_session.trial_nb))


# In[ ]:


# find_peaks
dip_values = []
reb_values = []

# Loop over trial numbers from 1 to trial_nb_all
for k in range(1, trial_nb_all+1):
    # Calculate the mean over the specified event_time interval for dip
    dip = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
        trial_nb=k, event_time=slice(75, 250))
    
    # Append the value to the list
    dip_values.append(dip.values.min())

    # Calculate the mean over the specified event_time interval for reb
    reb = xr_photometry['hold_for_water_zscored_df_over_f'].sel(
        trial_nb=k, event_time=slice(200, 600)).mean(dim='event_time')
    # Append the value to the list
    reb_values.append(reb.values.max())

# Convert lists to pandas Series
dip_series = pd.Series(dip_values)
reb_series = pd.Series(reb_values)


# In[ ]:


new_inex = range(1, trial_nb_all+1)
df_trials = pd.DataFrame({
    'trial_nb': list(range(1,trial_nb_all+1)),
    'dip': dip_series.reindex(new_inex),
    'rebound': reb_series.reindex(new_inex),
    'outcome': xr_session['trial_outcome'].values.T.flatten(),  # flatten is used to convert (175, 1) to (175,)
    })
df_trials

df_trials = df_trials.dropna(subset=['dip'])
df_trials = df_trials.dropna(subset=['rebound'])


# In[ ]:


from matplotlib import pyplot as plt

fig, ax1 = plt.subplots()

ax1.plot(df_trials['trial_nb'], df_trials['dip'],'o')
# plt.plot(df_trials['trial_nb'], df_trials['rebound'],'o-')

plt.ylabel('Dip size')
plt.xlabel('Trial number')
plt.show()


# In[ ]:


df_trials['trial_nb']


# In[ ]:


from scipy.stats import pearsonr
from scipy.stats import linregress

corr, p_val = pearsonr(df_trials['trial_nb'], df_trials['dip'])

print(f'Pearsons correlation: {corr:.3f}, p = {p_val}')

slope, intercept, r_value, p_value, std_err = linregress(df_trials['trial_nb'], df_trials['dip'])


fig, ax1 = plt.subplots()

ax1.plot(df_trials['trial_nb'], df_trials['dip'],'o')
# plt.plot(df_trials['trial_nb'], df_trials['rebound'],'o-')

y_regress = df_trials['trial_nb'].values * slope + intercept

ax1.plot(df_trials['trial_nb'], y_regress)

plt.ylabel('Dip size')
plt.xlabel('Trial number')
plt.show()


# In[ ]:


from matplotlib import pyplot as plt

corr, p_val = pearsonr(df_trials['trial_nb'], df_trials['rebound'])

print(f'Pearsons correlation: {corr:.3f}, p = {p_val}')

slope, intercept, r_value, p_value, std_err = linregress(
    df_trials['trial_nb'], df_trials['rebound'])


fig, ax1 = plt.subplots()

ax1.plot(df_trials['trial_nb'], df_trials['rebound'], 'o')

y_regress = df_trials['trial_nb'].values * slope + intercept

ax1.plot(df_trials['trial_nb'], y_regress)

plt.ylabel('Rebound size')
plt.xlabel('Trial number')
plt.show()


# In[ ]:


from matplotlib import pyplot as plt

plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams["legend.frameon"] = False
plt.rcParams['font.family'] = ['Arial']

fig, ax1 = plt.subplots()

ax1.plot( df_trials['outcome'], df_trials['rebound'], 'o', fillstyle='none')

plt.ylabel('rebound size')

plt.show()


# In[ ]:


df_trials['dip'][df_trials['outcome'] == 'success']


# In[ ]:


from matplotlib import pyplot as plt

fig, ax1 = plt.subplots()

ax1.plot(np.ones([np.count_nonzero(df_trials['outcome'] == 'success'),1]), df_trials['dip'][df_trials['outcome'] == 'success'],  'o', fillstyle='none')
ax1.boxplot(df_trials['dip'][df_trials['outcome'] == 'success'],  positions=[
            1.2], vert=True, patch_artist=False)

ax1.plot(np.ones([np.count_nonzero(df_trials['outcome'] == 'aborted'),1])*2, df_trials['dip'][df_trials['outcome'] == 'aborted'],  'o', fillstyle='none')
ax1.boxplot(df_trials['dip'][df_trials['outcome'] == 'aborted'],  positions=[
            2.2], vert=True, patch_artist=False)

ax1.plot(np.ones([np.count_nonzero(df_trials['outcome'] == 'no_reach'),1])*3, df_trials['dip'][df_trials['outcome'] == 'no_reach'],  'o', fillstyle='none')
ax1.boxplot(df_trials['dip'][df_trials['outcome'] == 'no_reach'],  positions=[
            3.2], vert=True, patch_artist=False)


ax1.set_xticklabels(['success','aborted','no_reach'])
plt.ylabel('Dip size')
plt.show()


# In[ ]:


reach_dip


# In[ ]:


fig, ax1 = plt.subplots()
# Get 'dip' values for 'reach' and 'no_reach' groups
reach_dip = df_trials['dip'][df_trials['outcome'].isin(
    ['success', 'aborted'])].dropna()
no_reach_dip = df_trials['dip'][df_trials['outcome'] == 'no_reach'].dropna()


ax1.plot(np.ones([reach_dip.shape[0], 1]),
         reach_dip,  'o', fillstyle='none')
ax1.boxplot(reach_dip,  positions=[
            1.2], vert=True, patch_artist=False)

ax1.plot(np.ones([no_reach_dip.shape[0], 1])*2,
         no_reach_dip,  'o', fillstyle='none')
ax1.boxplot(no_reach_dip,  positions=[
            2.2], vert=True, patch_artist=False)


ax1.set_xticklabels(['Reach',  'No reach'])
plt.ylabel('Dip size')
plt.show()


# In[ ]:


from scipy.stats import mannwhitneyu

# Get 'dip' values for 'reach' and 'no_reach' groups
reach_dip = df_trials['dip'][df_trials['outcome'].isin(
    ['success', 'aborted'])].dropna()
no_reach_dip = df_trials['dip'][df_trials['outcome'] == 'no_reach'].dropna()

# Perform Mann-Whitney U test
u_val, p_val = mannwhitneyu(reach_dip, no_reach_dip, alternative='two-sided')

print('U-value:', u_val)
print('p-value:', p_val)


# In[ ]:


import scipy.stats as stats

# Get dip values for each outcome category
success_dip = df_trials['dip'][df_trials['outcome'] == 'success'].dropna()
aborted_dip = df_trials['dip'][df_trials['outcome'] == 'aborted'].dropna()
no_reach_dip = df_trials['dip'][df_trials['outcome'] == 'no_reach'].dropna()

# Perform ANOVA
f_val, p_val = stats.f_oneway(success_dip, aborted_dip, no_reach_dip)

print('F-value:', f_val)
print('p-value:', p_val)


# In[ ]:


from matplotlib import pyplot as plt

fig, ax1 = plt.subplots()

ax1.plot(np.ones([np.count_nonzero(df_trials['outcome'] == 'success'),1]), df_trials['rebound'][df_trials['outcome'] == 'success'],  'o', fillstyle='none')
ax1.boxplot(df_trials['rebound'][df_trials['outcome'] == 'success'],  positions=[
            1.2], vert=True, patch_artist=False)

ax1.plot(np.ones([np.count_nonzero(df_trials['outcome'] == 'aborted'),1])*2, df_trials['rebound'][df_trials['outcome'] == 'aborted'],  'o', fillstyle='none')
ax1.boxplot(df_trials['rebound'][df_trials['outcome'] == 'aborted'],  positions=[
            2.2], vert=True, patch_artist=False)

ax1.plot(np.ones([np.count_nonzero(df_trials['outcome'] == 'no_reach'),1])*3, df_trials['rebound'][df_trials['outcome'] == 'no_reach'],  'o', fillstyle='none')
ax1.boxplot(df_trials['rebound'][df_trials['outcome'] == 'no_reach'],  positions=[
            3.2], vert=True, patch_artist=False)


ax1.set_xticklabels(['success','aborted','no_reach'])
plt.ylabel('Rebound size')
plt.show()


# In[ ]:


df_trials100 = df_trials.loc[0:99, :]


# In[ ]:


from matplotlib import pyplot as plt

fig, ax1 = plt.subplots()

ax1.plot(np.ones([np.count_nonzero(df_trials100['outcome'] == 'success'), 1]),
         df_trials100['dip'][df_trials100['outcome'] == 'success'],  'o', fillstyle='none')
ax1.boxplot(df_trials100['dip'][df_trials100['outcome'] == 'success'],  positions=[
            1.2], vert=True, patch_artist=False)

ax1.plot(np.ones([np.count_nonzero(df_trials100['outcome'] == 'aborted'), 1])*2,
         df_trials100['dip'][df_trials100['outcome'] == 'aborted'],  'o', fillstyle='none')
ax1.boxplot(df_trials100['dip'][df_trials100['outcome'] == 'aborted'],  positions=[
            2.2], vert=True, patch_artist=False)

ax1.plot(np.ones([np.count_nonzero(df_trials100['outcome'] == 'no_reach'), 1])*3,
         df_trials100['dip'][df_trials100['outcome'] == 'no_reach'],  'o', fillstyle='none')
ax1.boxplot(df_trials100['dip'][df_trials100['outcome'] == 'no_reach'],  positions=[
            3.2], vert=True, patch_artist=False)


ax1.set_xticklabels(['success', 'aborted', 'no_reach'])
plt.ylabel('Dip size')
plt.show()


# In[ ]:


fig, ax1 = plt.subplots()
# Get 'dip' values for 'reach' and 'no_reach' groups
reach_dip = df_trials100['dip'][df_trials100['outcome'].isin(
    ['success', 'aborted'])].dropna()
no_reach_dip = df_trials100['dip'][df_trials100['outcome'] == 'no_reach'].dropna()


ax1.plot(np.ones([reach_dip.shape[0], 1]),
         reach_dip,  'o', fillstyle='none')
ax1.boxplot(reach_dip,  positions=[
            1.2], vert=True, patch_artist=False)

ax1.plot(np.ones([no_reach_dip.shape[0], 1])*2,
         no_reach_dip,  'o', fillstyle='none')
ax1.boxplot(no_reach_dip,  positions=[
            2.2], vert=True, patch_artist=False)


ax1.set_xticklabels(['Reach',  'No reach'])
plt.ylabel('Dip size')
plt.show()

