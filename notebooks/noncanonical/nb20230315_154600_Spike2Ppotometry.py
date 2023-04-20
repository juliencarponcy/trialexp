#!/usr/bin/env python
# coding: utf-8

# 
# # Export a session as Spike2 data with Photometry data
# 
# 

# In[76]:


import os

nb_name = "nb20230315_154600_Spike2Ppotometry.ipynb" #TODO change this

basename, ext = os.path.splitext(nb_name)
input_path = os.path.join(os.getcwd(), nb_name)

get_ipython().system('jupyter nbconvert "{input_path}" --to="python" --output-dir="{nb_dir}" --output="{basename}"')


# In[74]:


notebook_name


# # Imports

# In[1]:


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

# In[2]:


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


# In[3]:


tasks = pd.read_csv(tasksfile, usecols=[1, 2, 3, 4], index_col=False)
tasks


# ### Create an experiment object
# 

# In[4]:


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


# In[5]:


exp_cohort.match_sessions_to_files(photometry_dir, ext='ppd')
exp_cohort.sync_photometry_files(2)
exp_cohort.save()


# In[6]:


update_all_smrx = False

ss = exp_cohort.sessions

ss_ = [this_ss for this_ss in ss
       if (this_ss.subject_ID in [58])
       and (this_ss.task_name == 'reaching_go_spout_bar_mar23')
       and (this_ss.datetime.date() >= datetime.date(2023, 3, 25))]
ss_


# In[7]:


exp_cohort.sessions = ss_


# In[8]:


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


# In[9]:


exp_cohort.sessions[0].print_lines[0:30]


# In[10]:


for ss in exp_cohort.sessions:
    smrxname = re.sub('\.txt', f'_{ss.task_name}.smrx', ss.file_name)
    print(smrxname)


# In[11]:


exp_cohort.sessions[0].print_lines[0]

a = re.sub('\n', '', exp_cohort.sessions[0].print_lines[0])

print(a)


# In[12]:


vars(exp_cohort.sessions[0])


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


# Copy and modify `get_photometry_trials`
# 

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

fs = photometry_dict['sampling_rate']

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


# In[22]:


df_meta_photo


# In[23]:


col_names_numpy


# In[24]:


photometry_dict


# In[27]:


photometry_dict


# In[34]:


Y = photometry_dict['analog_1']
T = photometry_times_pyc 

nan_indices = np.argwhere(np.isnan(T))

T_nonan = np.delete(T, nan_indices)
Y_nonan = np.delete(Y, nan_indices)


# Use Waveform
# Need to use interp to accomodate data into Spike2 bins
new_T = np.arange(0, 100, 0.01) #TODO
new_Y = np.interp(new_T, T_nonan, Y_nonan)


# In[38]:


int16_info = np.iinfo(np.int16)
data = new_Y
scale = ((np.max(data) - np.min(data))*6553.6) / float(int16_info.max - int16_info.min)
offset = np.max(data) - float(int16_info.max) * scale/6553.6
print(scale)
print(offset)


# In[40]:


session.plot_session(keys=[], export_smrx = True, smrx_filename = 'temp.smrx', photometry_dict = photometry_dict)

# plot_session(self, keys: list = None, state_def: list = None, print_expr: list = None, 
#     event_ms: list = None, export_smrx: bool = False, smrx_filename: str = None, verbose :bool = False,
#     print_to_text: bool = True, vchange_to_text: bool = True, photometry_dict = photometry_dict)


# In[ ]:


# see  \Users\phar0528\Anaconda3\envs\trialexp\Lib\site-packages\sonpy\MakeFile.py
#NOTE cannot put file path in the pydoc block

raw_symbols  = SymbolValidator().values
symbols = [raw_symbols[i+2] for i in range(0, len(raw_symbols), 12)]
# 40 symbols

fig = go.Figure()
if keys is None:
    keys = session.times.keys()
else:
    for k in keys: 
        assert k in session.times.keys(), f"{k} is not found in session.time.keys()"

if export_smrx:
    import time

    from sonpy import lib as sp
    if smrx_filename is None:
        raise Exception('smrx_filename is required')
    #TODO assert .smlx

    mtc = re.search('\.smrx$', smrx_filename)
    if mtc is None:
        raise Exception('smrx_filename has to end with .smrx')

    MyFile = sp.SonFile(smrx_filename, nChans = int(400)) #NOTE int() is required
    #NOTE nChans = ctypes.c_uint16(400) # TypeError
    #NOTE nChans = 400 # MyFile.MaxChannels() = -1
    #NOTE sonpy 1.9.5 works with nChans (1.8.5. doesn't)
    CurChan = 0
    UsedChans = 0
    Scale = 65535/20
    Offset = 0
    ChanLow = 0
    ChanHigh = 5
    tFrom = 0
    tUpto = sp.MaxTime64()         # The maximum allowed time in a 64-bit SON file
    dTimeBase = 1e-6               # s = microseconds
    x86BufSec = 2.
    EventRate = 1/(dTimeBase*1e3)  # Hz, period is 1000 greater than the timebase
    SubDvd = 1                     # How many ticks between attached items in WaveMarks

    times_ = [np.max(session.times[k]) for k in keys if any(session.times[k])]
    if times_ == []:
        raise Exception('No time stamp found: Cannot determine MaxTime()')

    else:
        max_time_ms1 = np.max(times_) #TODO ValueError when np.max([]) 

        list_of_match = [re.match('^\d+', L) for L in session.print_lines if re.match('^\d+', L) is not None]
        max_time_ms2 = np.max([int(m.group(0)) for m in list_of_match])

        max_time_ms = np.max([max_time_ms1, max_time_ms2])
        time_vec_ms = np.arange(0, max_time_ms, 1000/EventRate)
        # time_vec_micros = np.arange(0, max_time_ms*1000, 10**6 * 1/EventRate)
        
        samples_per_s = EventRate
        interval = 1/samples_per_s

        samples_per_ms = 1/1000 * EventRate
        interval = 1/samples_per_s

        MyFile.SetTimeBase(dTimeBase)  # Set timebase


def write_event(MyFile, X_ms, title, y_index, EventRate, time_vec_ms):
    (hist, ___) = np.histogram(X_ms, bins=time_vec_ms) # time is 1000 too small

    eventfalldata = np.where(hist)

    MyFile.SetEventChannel(y_index, EventRate)
    MyFile.SetChannelTitle(y_index, title)
    if eventfalldata[0] is not []:
        MyFile.WriteEvents(int(y_index), eventfalldata[0]*1000) #dirty fix but works
        time.sleep(0.05)# might help?

    if verbose:
        print(f'{y_index}, {title}:')
        nMax = 10
        # nMax = int(MyFile.ChannelMaxTime(int(y_index))/MyFile.ChannelDivide(int(y_index))) 
        print(MyFile.ReadEvents(int(y_index), nMax, tFrom, tUpto)) #TODO incompatible function arguments.
        # [-1] when failed

        # ReadEvents(session: sonpy.amd64.sonpy.SonFile, 
        #     chan: int, 
        #     nMax: int, # probably the end of the range to read in the unit of number of channel divide
        #     tFrom: int, 
        #     tUpto: int = 8070450532247928832, 
        #     Filter: sonpy.amd64.sonpy.MarkerFilter = <sonpy.MarkerFilter> in mode 'First', with trace column -1 and items
        #     Layer 1 [
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        #     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

def write_marker_for_state(MyFile,X_ms, title, y_index, EventRate, time_vec_ms):

    # remove NaN
    X_notnan_ms = [x for x in X_ms if not np.isnan(x)]

    (hist, ___) = np.histogram(X_notnan_ms, bins=time_vec_ms) # time is 1000 too small

    eventfalldata = np.where(hist)

    nEvents = len(eventfalldata[0])

    MarkData = np.empty(nEvents, dtype=sp.DigMark)
    for i in range(nEvents):
        if (i+1) % 2 == 0:
            MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 0) #offset
        elif (i+1) % 2 == 1:
            MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 1) #onset
        else:
            raise Exception('oh no')
    MyFile.SetMarkerChannel(y_index, EventRate)
    MyFile.SetChannelTitle(y_index, title)
    if eventfalldata[0] is not []:
        MyFile.WriteMarkers(int(y_index), MarkData)
        time.sleep(0.05)# might help?

    if verbose:             
        print(f'{y_index}, {title}:')
        print(MyFile.ReadMarkers(int(y_index), nEvents, tFrom, tUpto)) #TODO failed Tick = -1

def write_textmark(MyFile, X_ms, title, y_index, txt, EventRate, time_vec_ms):

    (hist, ___) = np.histogram(X_ms, bins=time_vec_ms) # time is 1000 too small

    eventfalldata = np.where(hist)

    nEvents = len(eventfalldata[0])

    MarkData = np.empty(nEvents, dtype=sp.DigMark)

    TMrkData = np.empty(nEvents, dtype=sp.TextMarker)

    for i in range(nEvents):
        if (i+1) % 2 == 0:
            MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 0) #offset
        elif (i+1) % 2 == 1:
            MarkData[i] = sp.DigMark(eventfalldata[0][i]*1000, 1) #onset
        else:
            raise Exception('oh no')
        
        #NOTE Spike2 truncates text longer than 79 characters???
        TMrkData[i] = sp.TextMarker(re.sub('\n', '', txt[i]), MarkData[i])

    if len(txt) == 0:
        MyFile.SetTextMarkChannel(y_index, EventRate, 32)
    else:
        MyFile.SetTextMarkChannel(y_index, EventRate, max(len(s) for s in txt)+1)
    MyFile.SetChannelTitle(y_index, title)
    if eventfalldata[0] is not []:
        MyFile.WriteTextMarks(y_index, TMrkData)
        time.sleep(0.05)# might help?

    if verbose:
        print(f'{y_index}, {title}:')
        try:
            print(MyFile.ReadTextMarks(int(y_index), nEvents, tFrom, tUpto))
        except:
            print('error in print')

def find_states(state_def_dict: dict):
    """
    state_def: dict, list, or None = None
    must be None (default)
    or dictionary of 
        'name' : str 
            Channel name
        'onset' : str | list of str 
            key for onset 
        'offset' : str | list of str 
            key for offset
    or list of such dictionaries

    eg. dict(name='trial', onset='CS_Go', offset='refrac_period')
    eg. {'name':'trial', 'onset':'CS_Go', 'offset':'refrac_period'}
    eg. {'name':'trial', 'onset':'CS_Go', 'offset': ['refrac_period', 'break_after_abortion']}

    For each onset, find the first offset event before the next onset 
    You can use multiple definitions with OR operation, eg. 'offset' determined by 'abort' or 'success', whichever comes first            
    """
    if state_def_dict is None:
        return None

    if isinstance(state_def_dict['onset'], str):
        all_on_ms = session.times[state_def_dict['onset']]
    elif isinstance(state_def_dict['onset'], list):
        # OR operation
        all_on_ms = []
        for li in state_def_dict['onset']:
            assert isinstance(li, str), 'onset must be str or list of str'
            all_on_ms.extend(session.times[li])
        all_on_ms = sorted(all_on_ms)
        
    else:
        raise Exception("onset is in a wrong type") 

    if isinstance(state_def_dict['offset'], str):
        all_off_ms = session.times[state_def_dict['offset']]
    elif isinstance(state_def_dict['offset'], list):
        # OR operation
        all_off_ms = []
        for li in state_def_dict['offset']:
            assert isinstance(li, str), 'offset must be str or list of str'                    
            all_off_ms.extend(session.times[li])
        all_off_ms = sorted(all_off_ms)
    else:
        raise Exception("offset is in a wrong type") 

    onsets_ms = [np.NaN] * len(all_on_ms)
    offsets_ms = [np.NaN] * len(all_on_ms)

    for i, this_onset in enumerate(all_on_ms):  # slow
        good_offset_list_ms = []
        for j, _ in enumerate(all_off_ms):
            if i < len(all_on_ms)-1:
                if all_on_ms[i] < all_off_ms[j] and all_off_ms[j] < all_on_ms[i+1]:
                    good_offset_list_ms.append(all_off_ms[j])
            else:
                if all_on_ms[i] < all_off_ms[j]:
                    good_offset_list_ms.append(all_off_ms[j])

        if len(good_offset_list_ms) > 0:
            onsets_ms[i] = this_onset
            offsets_ms[i] = good_offset_list_ms[0]
        else:
            ...  # keep them as nan

    onsets_ms = [x for x in onsets_ms if not np.isnan(x)]  # remove nan
    offsets_ms = [x for x in offsets_ms if not np.isnan(x)]

    state_ms = map(list, zip(onsets_ms, offsets_ms,
                    [np.NaN] * len(onsets_ms)))
    # [onset1, offset1, NaN, onset2, offset2, NaN, ....]
    state_ms = [item for sublist in state_ms for item in sublist]
    return state_ms

y_index = 0
for kind, k in enumerate(keys):
    y_index += 1
    line1 = go.Scatter(x=session.times[k]/1000, y=[k]
                * len(session.times[k]), name=k, mode='markers', marker_symbol=symbols[y_index % 40])
    fig.add_trace(line1)

    if export_smrx:
        write_event(MyFile, session.times[k], k, y_index, EventRate, time_vec_ms)



if print_expr is not None: #TODO
    if isinstance(print_expr, dict):
        print_expr = [print_expr]

    for dct in print_expr:
        y_index += 1
        expr = '^\d+(?= ' + dct['expr'] + ')'
        list_of_match = [re.match(expr, L) for L in session.print_lines if re.match(expr, L) is not None]
        ts_ms = [int(m.group(0)) for m in list_of_match]
        line2 = go.Scatter(
            x=[TS_ms/1000 for TS_ms in ts_ms], y=[dct['name']] * len(ts_ms), 
            name=dct['name'], mode='markers', marker_symbol=symbols[y_index % 40])
        fig.add_trace(line2)

        if export_smrx:
            write_event(
                MyFile, ts_ms, dct['name'], y_index, EventRate, time_vec_ms)

if event_ms is not None:
    if isinstance(event_ms, dict):
        event_ms = [event_ms]
    
    for dct in event_ms:
        y_index += 1
        line3 = go.Scatter(
            x=[t/1000 for t in dct['time_ms']],
            y=[dct['name']] * len(dct['time_ms']),
            name=dct['name'], mode='markers', marker_symbol=symbols[y_index % 40])
        fig.add_trace(line3)

        if export_smrx:
            write_event(
                MyFile, dct['time_ms'], dct['name'], y_index, EventRate, time_vec_ms)

if print_to_text:

    EXPR = '^(\d+)\s(.+)' #NOTE . doesn't capture \n and re.DOTALL is required below
    list_of_match = [re.match(EXPR, L, re.DOTALL) for L in session.print_lines if re.match(EXPR, L) is not None]
    ts_ms = [int(m.group(1)) for m in list_of_match]
    txt = [m.group(2) for m in list_of_match]

    # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

    y_index += 1
    txtsc = go.Scatter(x=[TS_ms/1000 for TS_ms in ts_ms], y=['print_lines']*len(ts_ms), 
        text=txt, textposition="top center", 
        mode="markers", marker_symbol=symbols[y_index % 40])
    fig.add_trace(txtsc)

    if export_smrx:
        write_textmark( MyFile, ts_ms, 'print lines', y_index, txt, EventRate, time_vec_ms)

if vchange_to_text:
    EXPR = '^([1-9]\d*)\s(.+)' #NOTE Need to ignore the defaults (V 0 ****)
    list_of_match = [re.match(EXPR, L) for L in session.v_lines if re.match(EXPR, L) is not None]
    ts_ms = [int(m.group(1)) for m in list_of_match]
    txt = [m.group(2) for m in list_of_match]

    # df_print = pd.DataFrame(list(zip(ts_ms, txt)), columns=['ms', 'text'])

    y_index += 1
    txtsc = go.Scatter(x=[TS_ms/1000 for TS_ms in ts_ms], y=['V changes']*len(ts_ms), 
        text=txt, textposition="top center", 
        mode="markers", marker_symbol=symbols[y_index % 40])
    fig.add_trace(txtsc)

    if export_smrx:
        write_textmark( MyFile, ts_ms, 'V changes', y_index, txt, EventRate, time_vec_ms)


if state_def is not None:
    # Draw states as gapped lines
    # Assuming a list of lists of two names

    if isinstance(state_def, dict):# single entry
        state_def = [state_def]
        # state_ms = find_states(state_def)

        # line1 = go.Scatter(x=[x/1000 for x in state_ms], y=[state_def['name']] * len(state_ms), 
        #     name=state_def['name'], mode='lines', line=dict(width=5))
        # fig.add_trace(line1)

    if isinstance(state_def, list):# multiple entry
        state_ms = None
        for i in state_def:
            assert isinstance(i, dict)
            
            y_index +=1
            state_ms = find_states(i)

            line1 = go.Scatter(x=[x/1000 for x in state_ms], y=[i['name']] * len(state_ms), 
                name=i['name'], mode='lines', line=dict(width=5))
            fig.add_trace(line1)

            if export_smrx:
                write_marker_for_state(MyFile, state_ms, i['name'], y_index, EventRate, time_vec_ms)
    else:
        state_ms = None
else:
    state_ms = None
        

fig.update_xaxes(title='Time (s)')
fig.update_yaxes(fixedrange=True) # Fix the Y axis

fig.update_layout(
    
    title =dict(
        text = f"{session.task_name}, {session.subject_ID} #{session.number}, on {session.datetime_string} via {session.setup_ID}"
    )
)

fig.show()

if export_smrx:
    del MyFile
    #NOTE when failed to close the file, restart the kernel to delete the corrupted file(s)
    print(f'saved {smrx_filename}')

# Implemented in Event_dataset(), in trial_dataset_classes but left here for convenience as well
def plot_trials_events(session, events_to_plot:list = 'all',  sort:bool = False):

# I dont get that K, review symbol selection? 
raw_symbols  = SymbolValidator().values
symbols = [raw_symbols[i+2] for i in range(0, len(raw_symbols), 12)]

event_cols = [event_col for event_col in session.events_to_process]
event_names = [event_col.split('_trial_time')[0] for event_col in event_cols]

if events_to_plot == 'all':
    events_to_plot = session.events_to_process

elif isinstance(events_to_plot, list):

    # check if events requested exist
    check = all(ev in event_names for ev in events_to_plot)

    if not check:
        raise Exception('Check your list of requested events, event not found')
    
    event_cols = [ev + '_trial_time' for ev in events_to_plot]
    event_names = events_to_plot

elif isinstance(events_to_plot, str):
    if events_to_plot not in event_names:
        raise Exception('Check the name of your requested event, event not found')
    
    event_cols = [events_to_plot + '_trial_time']
    event_names = [events_to_plot]

else:
    raise Exception('bad format for requesting plot_trials events')

# Implement this as abstract method to check requested arguments (events) match the session obj.

plot_names =  [trig + ' ' + event for event in event_cols for trig in session.triggers]

# https://plotly.com/python/subplots/
# https://plotly.com/python/line-charts/
fig = make_subplots(
    rows= len(event_cols), 
    cols= len(session.triggers), 
    shared_xaxes= True,
    shared_yaxes= True,
    subplot_titles= plot_names
)

for trig_idx, trigger in enumerate(session.df_events.trigger.unique()):
    
    # sub-selection of df_events based on trigger, should be condition for event_dataset class
    df_subset = session.df_events[session.df_events.trigger == trigger]


    for ev_idx, event_col in enumerate(event_cols):
        # if sort:
        #     min_times = df_subset[event_cols[ev_idx]].apply(lambda x: find_min_time_list(x))
        #     min_times = np.sort(min_times)

        ev_times = df_subset[event_cols[ev_idx]].apply(lambda x: np.array(x)).values
        ev_trial_nb = [np.ones(len(array)) * df_subset.index[idx] for idx, array in enumerate(ev_times)]

        ev_trial_nb = np.concatenate(ev_trial_nb)
        ev_times =  np.concatenate(ev_times)

        fig.add_shape(type="line",
            x0=0, y0=1, x1=0, y1= ev_trial_nb.max(),
            line=dict(
            color="Grey",
            width=2,
            dash="dot"
            ),
            row= ev_idx+1,
            col = trig_idx+1)

        fig.add_trace(
            go.Scatter(
                x= ev_times/1000,
                y= ev_trial_nb,
                name= event_names[ev_idx],
                mode='markers',
                marker_symbol=symbols[ev_idx % 40]
                ),
                row= ev_idx+1,
                col = trig_idx+1)

            

        fig.update_xaxes(
            title_text = 'time (s)',
            ticks = 'outside',
            ticklen = 6,
            tickwidth = 2,
            tickfont_size = 12,
            showline = True,
            linecolor = 'black',
            # range=[session.trial_window[0]/1000, session.trial_window[1]/1000]
            autorange = True,
            row = ev_idx+1,
            col = trig_idx+1
            )
        
        fig.update_yaxes( 
            title_text = 'trial nb', 
            ticks = 'outside',
            ticklen = 6,
            tickwidth = 2,   
            tickfont_size = 12,
            showline = True,
            linecolor = 'black',
            range = [1, ev_trial_nb.max()],
            showgrid=True,
            row = ev_idx+1,
            col = trig_idx+1
            )

fig.update_layout(
    title_text= f'Events Raster plot, ID:{session.subject_ID} / {session.task_name} / {session.datetime_string}',
    height=800,
    width=800
                
)

fig.show()




# # get_photometry_trials?
# 
# `get_photometry_trials` > `get_trials_times_from_conditions`
# 
# This won't work for non-trial based analysis because it's dependent on `conditions_list`
# 
# 

# In[ ]:


df_meta_photo, col_names_numpy, photometry_array, photometry_dict = get_photometry_trials(
    exp_cohort.sessions[i],
    conditions_list=None,
    cond_aliases = None,
    trial_window = trial_window,
    trig_on_ev = None,
    last_before = None,
    baseline_low_pass = 0.001, # var name changed from former high-pass,
    # was misleading on baseline computation
    # see https://github.com/juliencarponcy/trialexp/issues/8
    # first fix 
    low_pass = 45, 
    median_filt = 3,
    motion_corr = True, 
    df_over_f = True, 
    z_score = True,
    downsampling_factor = 10, 
    return_full_session = True,
    export_vars = ['analog_1_df_over_f', 'zscored_df_over_f'],
    # remove_artifacts = remove_artifacts,
    verbose = True)



# #TODO
# 
# - How the pyphotometry data are stored?
# - How can I align them with behaviour data?

# In[ ]:


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

    if update_all_smrx or not os.path.isfile(smrxname):

        try:
            ss.plot_session(
                keys, state_def, export_smrx=True, event_ms=event_ms, smrx_filename= smrxname) #TODO

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


