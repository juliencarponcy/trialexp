#!/usr/bin/env python
# coding: utf-8

# 
# # Export a session as Spike2 data with Photometry data
# 
# 

# In[1]:


# import ipynbname
import os

nb_name = "nb20230425_100000_Spike2Ppotometry.ipynb"

basename, ext = os.path.splitext(nb_name)
input_path = os.path.join(os.getcwd(), nb_name)

get_ipython().system('jupyter nbconvert "{input_path}" --to="python" --output="{basename}"')


# # Imports

# In[2]:


# allow for automatic reloading of classes and function when updating the code
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# Import Session and Experiment class with helper functions
from trialexp.process.data_import import *
from trialexp.process.pyphotometry.photometry_functional import *
import pandas as pd
import datetime
import re


# # Variables

# In[3]:


trial_window = [-2000, 6000]  # in ms

# time limit around trigger to perform an event
# determine successful trials
timelim = [0, 2000]  # in ms

# Digital channel nb of the pyphotometry device
# on which rsync signal is sent (from pycontrol device)
rsync_chan = 2

basefolder, _ = os.path.split(os.path.split(os.getcwd())[0])

# These must be absolute paths
# use this to use within package tasks files (in params)
tasksfile = os.path.join(basefolder, 'params\\tasks_params.csv')
# use this to put a local full path
#tasksfile = -r'C:/.../tasks_params.csv'

# photometry_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\test_folder\photometry'
photometry_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\pyphotometry\data\reaching_go_spout_bar_nov22'
video_dir = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\videos'


# In[4]:


tasks = pd.read_csv(tasksfile, usecols=[1, 2, 3, 4], index_col=False)
tasks


# ### Create an experiment object
# 

# In[5]:


# Folder of a full experimental batch, all animals included

# Enter absolute path like this
# pycontrol_files_path = r'T:\Data\head-fixed\test_folder\pycontrol'

# or this if you want to use data from the sample_data folder within the package
#pycontrol_files_path = os.path.join(basefolder, 'sample_data/pycontrol')
pycontrol_files_path = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\pycontrol\reaching_go_spout_bar_nov22'

# Load all raw text sessions in the indicated folder or a sessions.pkl file
# if already existing in folder_path
exp_cohort = Experiment(pycontrol_files_path, update=True)  # TODO

# Only use if the Experiment cohort as been processed by trials before
# TODO: assess whether this can be removed or not
exp_cohort.by_trial = True


smrx_folder_path = r'\\ettin\Magill_Lab\Julien\Data\head-fixed\pycontrol\reaching_go_spout_bar_nov22\processed'


# In[6]:


exp_cohort.match_sessions_to_files(photometry_dir, ext='ppd')
exp_cohort.sync_photometry_files(2)
exp_cohort.save()


# In[7]:


update_all_smrx = False

ss = exp_cohort.sessions

ss_ = [this_ss for this_ss in ss
       if (this_ss.subject_ID in [58])
       and (this_ss.task_name == 'reaching_go_spout_bar_mar23')
       and (this_ss.datetime.date() >= datetime.date(2023, 3, 25))]
ss_


# In[8]:


exp_cohort.sessions = ss_


# In[9]:


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


# In[10]:


exp_cohort.sessions[0].print_lines[0:30]


# In[11]:


for ss in exp_cohort.sessions:
    smrxname = re.sub('\.txt', f'_{ss.task_name}.smrx', ss.file_name)
    print(smrxname)


# In[12]:


exp_cohort.sessions[0].print_lines[0]

a = re.sub('\n', '', exp_cohort.sessions[0].print_lines[0])

print(a)


# In[13]:


vars(exp_cohort.sessions[0].photometry_rsync)


# In[14]:


i = 0

photometry_aligner = Rsync_aligner(exp_cohort.sessions[i].photometry_rsync.pulse_times_A, 
    exp_cohort.sessions[i].photometry_rsync.pulse_times_B,
    chunk_size=5, plot=False, raise_exception=True)
photometry_dict = import_ppd(exp_cohort.sessions[i].files['ppd'][0])
photometry_times_pyc = photometry_aligner.B_to_A(photometry_dict['time'])


# In[15]:


lst = list(exp_cohort.sessions[i].__dict__.keys())
print(lst)


# In[16]:


condition_list
cond_aliases


# ## modified from `get_photometry_trials`
# 
# taken from `trialexp\process\pyphotometry\photometry_functional.py`
# 
# confusingly the same name of fuctions are found in 
# ```
# \process\data_import.py
# \process\pycontrol\data_import.py
# \process\pycontrol\session_analysis.py
# \process\pyphotometry\photometry_functional.py
# ```
# 
# 
# parameters are taken from `notebooks\noncanonical\nb20230222_092800_poster_delayed_tasks_photometry.ipynb`
# calls `get_photometry_groups` in `trialexp\process\data_import.py`
# calls `get_photometry_trials` in `trialexp\process\pyphotometry\photometry_functional.py`

# In[17]:


session = exp_cohort.sessions[i]

photometry_dict = import_ppd(session.files['ppd'][0])

trig_on_ev = None
last_before = None
baseline_low_pass = 0.001, # var name changed from former high-pass,
# was misleading on baseline computation
# see https://github.com/juliencarponcy/trialexp/issues/8
# first fix 
low_pass = 45
median_filt = 3
motion_corr = True
df_over_f = True
z_score = True
downsampling_factor = 10
return_full_session = True
export_vars = ['analog_1_df_over_f', 'zscored_df_over_f']

if low_pass:
    # Filter signals with specified high and low pass frequencies (Hz).
    b, a = get_filt_coefs(low_pass=low_pass, high_pass=None, sampling_rate=photometry_dict['sampling_rate'])
    
    if median_filt:
        analog_1_medfilt = median_filtering(photometry_dict['analog_1'], medfilt_size = median_filt)
        analog_2_medfilt = median_filtering(photometry_dict['analog_2'], medfilt_size = median_filt)
        photometry_dict['analog_1_filt'] = filtfilt(b, a, analog_1_medfilt)
        photometry_dict['analog_2_filt'] = filtfilt(b, a, analog_2_medfilt)

    else:
        photometry_dict['analog_1_filt'] = filtfilt(b, a, photometry_dict['analog_1'])
        photometry_dict['analog_2_filt'] = filtfilt(b, a, photometry_dict['analog_2'])
else:
    if median_filt:
        photometry_dict['analog_1_filt'] = median_filtering(photometry_dict['analog_1'], medfilt_size = median_filt)
        photometry_dict['analog_2_filt'] = median_filtering(photometry_dict['analog_2'], medfilt_size = median_filt)  
    else:
        photometry_dict['analog_1_filt'] = photometry_dict['analog_2_filt'] = None
# TODO: verify/improve/complement the implementation of the following:


if motion_corr == True:

    slope, intercept, r_value, p_value, std_err = linregress(x=photometry_dict['analog_2_filt'], y=photometry_dict['analog_1_filt'])
    photometry_dict['analog_1_est_motion'] = intercept + slope * photometry_dict['analog_2_filt']
    photometry_dict['analog_1_corrected'] = photometry_dict['analog_1_filt'] - photometry_dict['analog_1_est_motion']
    
    if df_over_f == False:
        export_vars.append('analog_1_corrected')
        # signal = photometry_dict['analog_1_corrected']
    elif df_over_f == True:
        # fror 
        b,a = butter(2, baseline_low_pass, btype='low', fs=photometry_dict['sampling_rate'])
        photometry_dict['analog_1_baseline_fluo'] = filtfilt(b,a, photometry_dict['analog_1_filt'], padtype='even')

        # Now calculate the dF/F by dividing the motion corrected signal by the time varying baseline fluorescence.
        photometry_dict['analog_1_df_over_f'] = photometry_dict['analog_1_corrected'] / photometry_dict['analog_1_baseline_fluo'] 
        export_vars.append('analog_1_df_over_f')
        # signal = photometry_dict['analog_1_df_over_f']
if z_score:
    # z-score the signal
    photometry_dict['zscored_df_over_f'] = zscore(photometry_dict['analog_1_df_over_f'])
    export_vars.append('zscored_df_over_f')
elif baseline_low_pass or low_pass:
    # signal = photometry_dict['analog_1_filt']']
    export_vars.append('analog_1_filt')

    # control = photometry_dict['analog_2_filt']']
else:
    export_vars.append('analog_1')
# signal = photometry_dict['analog_1']']

# only keep unique items (keys for the photometry_dict)
export_vars = list(set(export_vars))

if downsampling_factor:
    # downsample
    for k in export_vars:
        photometry_dict[k] = decimate(photometry_dict[k], downsampling_factor) 
    # adjust sampling rate accordingly (maybe unnecessary)
    photometry_dict['sampling_rate'] = photometry_dict['sampling_rate'] / downsampling_factor

# fs = photometry_dict['sampling_rate']

df_meta_photo = pd.DataFrame(columns=['subject_ID', 'datetime', 'task_name', 'condition_ID', 'trial_nb'])

# Prepare dictionary output with keys are variable names and values are columns index
col_names_numpy = {var: var_idx for var_idx, var in enumerate(export_vars)}






# In[18]:


trials_idx = np.zeros((1, 1))
timestamps_pycontrol = np.zeros((1, 1))

print(trials_idx.shape)


# In[19]:


# assumes that sync between pycontrol and photometry has been performed in previous step
timestamps_photometry = session.photometry_rsync.A_to_B(timestamps_pycontrol)

photometry_idx = (timestamps_photometry / (1000/photometry_dict['sampling_rate'])).round().astype(int)

# retain only trials with enough values left and right
complete_mask = (photometry_idx + trial_window[0]/(1000/photometry_dict['sampling_rate']) >= 0) & (
    photometry_idx + trial_window[1] < len(photometry_dict[export_vars[0]])) 

# complete_idx = np.where(complete_mask)
# trials_idx = np.array(trials_idx)
# photometry_idx = np.array(photometry_idx)

trials_idx = trials_idx[complete_mask]           
photometry_idx = photometry_idx[complete_mask]

# if verbose:
#     print(f'condition {condition_ID} trials: {len(trials_idx)}')

if len(trials_idx) == 0:
    print('nothing')

# Construct ranges of idx to get chunks (trials) of photometry data with np.take method 
photometry_idx = [range(idx + int(trial_window[0]/(1000/photometry_dict['sampling_rate'])) ,
    idx + int(trial_window[1]/(1000/photometry_dict['sampling_rate']))) for idx in photometry_idx]


# In[20]:


photo_array = np.ndarray((len(trials_idx), len(photometry_idx[0]),len(export_vars)))

for var_idx, photo_var in enumerate(export_vars):
    # print(f'condition {condition_ID} var: {var_idx} shape {np.take(photometry_dict[photo_var], photometry_idx).shape}')
    photo_array[:,:,var_idx] = np.take(photometry_dict[photo_var], photometry_idx)


df_meta_photo['trial_nb'] = trials_idx
df_meta_photo['subject_ID'] = session.subject_ID
df_meta_photo['datetime'] = session.datetime
df_meta_photo['task_name'] = session.task_name
# df_meta_photo['condition_ID'] = condition_ID


# In[21]:


if 'photo_array' in locals():
            photo_array = photo_array.swapaxes(2,1)
else:
    # This occurs when no photometry data is recored at all for the session
    # would occur anyway without the previous check, 
    # avoid it happening spontaneously on return.
    # useless but could be use to convey extra information to calling method
    
    if verbose:
        print(f'No photometry data to collect for subject ID:{session.subject_ID}\
            \nsession: {session.datetime}')

    raise UnboundLocalError()

    # Trying to implement empty arrays and dataframe when nothing to return
    # df_meta_photo = pd.DataFrame(columns=['subject_ID', 'datetime', 'task_name', 'condition_ID', 'trial_nb'])
    # ra
    # photo_array = np.ndarray((len(trials_idx), len(photometry_idx),len(export_vars)))

# df_meta_photo, col_names_numpy, photo_array, photometry_dict


# # Photometry data processing?
# 
# 

# In[22]:


[print(k) for k in photometry_dict.keys()];


# In[23]:


nan_indices = np.argwhere(np.isnan(photometry_times_pyc))
T_nonan = np.delete(photometry_times_pyc, nan_indices)
max_time_ms = T_nonan[-1]

def get_newTandY_orig(T, photometry_dict, name, max_time_ms):
    T = photometry_times_pyc # not down-sampled yet

    nan_indices = np.argwhere(np.isnan(T))
    T_nonan = np.delete(T, nan_indices)

    Y = photometry_dict[name]
    Y_nonan = np.delete(Y, nan_indices)  # []
    max_time_ms = T_nonan[-1]



    new_T = np.arange(0, max_time_ms, 1/1000*1000) #NOTE sampling_rate was originally 1000
    new_Y = np.interp(new_T, T_nonan, Y_nonan)
    return new_T, new_Y


def get_newTandY_down(T, photometry_dict, name, max_time_ms):
    Tdown = [T[i] for i in range(0, len(T), 10)] #down sampled time vector

    nan_indices = np.argwhere(np.isnan(Tdown))
    Tdown_nonan = np.delete(Tdown, nan_indices)

    Y = photometry_dict[name]
    Y_nonan = np.delete(Y, nan_indices) # []

    # Need to use interp to accomodate data into Spike2 bins
    new_T = np.arange(0, max_time_ms, 1/photometry_dict['sampling_rate']*1000) #NOTE sampling_rate is already downsampled by 10
    new_Y = np.interp(new_T, Tdown_nonan, Y_nonan)
    return new_T, new_Y


# # Motion correction ideas
# 
# `process/pyphotometry/utils.py/motion_corretion()`
# 
# 1. Linear regression for the subset of data when dynamic
# 2. Partial least square (PLS) regression
# 3. Motion correction with linear regression on average waveforms (question of when to apply motion correction)
# 4. Create waveform averages of `analog_2_filt`
# 
# 
# 
# 

# ## PLS regression

# In[24]:


# PLS

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Prepare data
T = photometry_times_pyc # not down-sampled yet
nan_indices = np.argwhere(np.isnan(T))

X = photometry_dict['analog_2_filt'].reshape(-1, 1)
X = np.delete(X, nan_indices)
X = X.reshape(-1, 1)

y = photometry_dict['analog_1_filt'].reshape(-1, 1)
y = np.delete(y, nan_indices)
y = y.reshape(-1, 1)


# split the data to training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# PLS regression model creation and learning
pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)

# model prediction and evaluation
y_pred = pls.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean squared error: {mse:.2f}')
print(f'R2 score: {r2:.2f}')




# In[25]:


len(X_test)


# PLS did not solve the problem out of the box
# 
# Model is still twice as large as the real data

# In[26]:


from matplotlib import pyplot as plt

T = photometry_times_pyc # not down-sampled yet
nan_indices = np.argwhere(np.isnan(T))
T_nonan = np.delete(T, nan_indices)
T_nonan_s = [t/1000 for t in T_nonan]

lr = np.delete(photometry_dict['analog_1_est_motion'], nan_indices)

# plt.plot(T_nonan_s, X)
plt.plot(T_nonan_s, y, linewidth=0.5)
plt.plot(T_nonan_s, pls.predict(X), linewidth=0.5)
plt.plot(T_nonan_s, lr, linewidth=0.5)

plt.xlim(2100,2200)
plt.show()



# In[47]:


#  Applying bandpass (including highpass 0.001 Hz) filter to remove ultraslow components

from scipy.signal import butter
from trialexp.process.pyphotometry.utils import *



# In[60]:


list(lowpass_freq)


# In[61]:


# explore low-pass cut off freq
start_time_ms = 2000 * 1000  # 2000 seconds in milliseconds
end_time_ms = 2100 * 1000

pairs = [(t, d) for t, d in zip(T, photometry_dict['analog_1']) if start_time_ms <= t < end_time_ms]
T_subset, analog_1_subset = zip(*pairs)
T_subset = list(T_subset)
analog_1_subset = list(analog_1_subset)

sampling_rate = 1000
highpass_freq = 0.001
lowpass_freq = range(10, 50, 5)
for L in lowpass_freq:
    b, a = get_filt_coefs(low_pass=L, 
                        high_pass=highpass_freq, 
                        sampling_rate=sampling_rate)
    #TODO may be filter is not stable or something? I don't see an effect of low-pass 
    analog_1_subset_filt = filtfilt(b, a, analog_1_subset, padtype='even')

    plt.plot([t/1000 for t in T_subset], 
             analog_1_subset_filt, label=f'{L:d}',
             linewidth=0.3)

plt.legend(loc='upper right')
plt.xlim(2000, 2010)
plt.show()


# In[62]:


lowpass_freq = 33.3
highpass_freq = 0.001 # 10 s cycle
sampling_rate = 1000

# trialexp\process\pyphotometry\utils.py
# see https://vscode.dev/github/juliencarponcy/trialexp/blob/fd1e0dcc857275cafa7f809a104fd60e73ce1458/trialexp/process/pyphotometry/utils.py#L51
b, a = get_filt_coefs(low_pass=lowpass_freq, 
                      high_pass=highpass_freq, 
                      sampling_rate=sampling_rate)
analog_1_filt = filtfilt(b, a, photometry_dict['analog_1'], padtype='even')
analog_2_filt = filtfilt(b, a, photometry_dict['analog_2'], padtype='even')


# In[63]:


plt.plot(T/1000, analog_1_filt, linewidth=0.5, label='1: dLight', color='g')
plt.plot(T/1000, analog_2_filt, linewidth=0.5, alpha=0.3, label='2: tdTomato', color = 'r')
plt.legend(loc='upper right')
plt.show()

plt.plot(T/1000, analog_1_filt, linewidth=0.5, label='1: dLight', color='g')
plt.plot(T/1000, analog_2_filt, linewidth=0.5, alpha=0.3, label='2: tdTomato', color = 'r')
plt.legend(loc='upper right')
plt.xlim(2000, 2100)
plt.show()


# # Linear regression after band-pass filtering

# In[42]:


# https://vscode.dev/github/juliencarponcy/trialexp/blob/fd1e0dcc857275cafa7f809a104fd60e73ce1458/trialexp/process/pyphotometry/utils.py#L63
# def motion_correction()
slope, intercept, r_value, p_value, std_err = linregress(x=analog_2_filt, y=analog_1_filt)
analog_1_est_motion = intercept + slope * analog_2_filt
analog_1_corrected = analog_1_filt - analog_1_est_motion


# In[68]:


fig, ax = plt.subplots(4,1, sharey=True, sharex= True)

ax[0].plot(T/1000, analog_1_filt, linewidth=0.5, label='bandpass filtered')
ax[1].plot(T/1000, analog_1_est_motion, linewidth=0.5,  label='estimate')
ax[2].plot(T/1000, analog_1_corrected, linewidth=0.5, label='corrected (new)')
ax[3].plot(T/1000, photometry_dict['analog_1_corrected'], linewidth=0.5, label='corrected (old)')

plt.xlim(2000, 2010)
plt.ylim(-0.04, 0.04)

for i in range(0,3):
    leg = ax[i].legend(loc='upper right')
    leg.get_frame().set_edgecolor('none')

plt.show()


# In[ ]:


#TODO window approach




# I thought that using average waveforms, it would be easier to fit the red and green channels, but this comes with an assumption that the vast majority of data points in average waveform does not have dopamine signals.
# 
# For instance, if there is a constant dopamine release, we cannot see it.
# 
# How about trial-based motion correction? This can avoid the issue of baseline domination.
# 
# Or work of averages of groups of 10 trials instead of all the trials? We can get many smoothed curves for modelling.
# 
# How does pcolor look like for individual trials for analog_2_filt? Are they highly reproducible?
# 
# 

# In[ ]:


# correction on average approach




# In[24]:


from matplotlib import pyplot as plt

photometry_dict['analog_1_filt']

photometry_dict['analog_2_filt']


# In[ ]:


slope, intercept, r_value, p_value, std_err = linregress(
    x=photometry_dict['analog_2_filt'], y=photometry_dict['analog_1_filt'])

print(f"slope: {slope:.3f}\nintercept: {intercept:.3f}\nr_value: {r_value:.3f}\np_value: {p_value}")




fig, ax = plt.subplots()

plt.hist2d(y2, y1, bins=(100, 100), cmap='viridis')

# Add a colorbar
plt.colorbar()

x0 = np.arange(np.min(y2), np.max(y2), (np.max(y2) - np.min(y2))/1000)
y0 = slope*x0 + intercept

plt.plot(x0, y0, '-', color='yellow', linewidth=2)

# Customize and show the plot
plt.xlabel('analog_2_filt, red')
plt.ylabel('analog_1_filt, green')
plt.title('Heatmap Representation')
# plt.grid(True)
plt.show()

fig, ax = plt.subplots()

plt.hist2d(y2, y1, bins=(100, 100), cmap='viridis', vmax=2000)
plt.plot(x0, y0, '-', color='yellow', linewidth=2)
plt.colorbar()

plt.xlabel('analog_2_filt, red')
plt.ylabel('analog_1_filt, green')
plt.title('Heatmap Representation')
plt.show()


# This method uses motion artifacts and real signals together.  
# 
# My concern is that because there are so many data points at baseline, eg X = 0.55, y= 0.165, the regression cofficients are not working well to fit values with higher fluorescence.
# 
# I suspect that the intercept is working well but the slope is far from being accurate.
# 
# Perhaps we need to apply some kind of adjusted weight to carry out regression better.
# 
# It appears that at higher X and Y values, data points go further up from the regression line. Because they have more than baseline signals, this deflection may be critical. 
# 
# In other words, the regressdion line is __underestimating__ an effect from motion captured in red channels.
# 
# Then, model predicted motion component is expected to be lower than the signals in analog_1. In reality, it is too big. So the direction of error is not 

# In[ ]:


fig, ax = plt.subplots()

t, y3 = get_newTandY_orig(T, photometry_dict, 'analog_1_est_motion', max_time_ms)
t, y4 = get_newTandY_orig(T, photometry_dict, 'analog_1_corrected', max_time_ms)

plt.plot(t/1000, y1, '-', label='analog_1_filt', linewidth=0.75)
# plt.plot(t/1000, y2, '-', label='analog_2_filt', linewidth=0.75)
plt.plot(t/1000, y3, '-', label='analog_1_est_motion', linewidth=0.75)
plt.plot(t/1000, y4, '-', label='analog_1_corrected', linewidth=0.75)

# Customize and show the plot
plt.legend()

plt.xlabel('Time (s)')
plt.ylabel('Photometry')
plt.title('Motion correction')
# plt.grid(True)
plt.xlim(2015,2040)
plt.show()


# 
