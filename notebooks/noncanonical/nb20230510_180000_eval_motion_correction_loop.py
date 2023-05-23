#!/usr/bin/env python
# coding: utf-8

# I weant to test a few ideas about motion correciton on different animals and somehow evaluate the effects and determine which one to adopt.
# 
# - Methods
#     1. Old method: lowpass filtering > linear regression > subtraction > dF/F > zscore
#     2. bandpass filtering to flatten data >  linear regression > subtraction > dF/F > zscore
#     3. bandpass filtering to flatten data >  PLS  regression > subtraction > dF/F > zscore
#     4 ????
# 
# - Evalutation methods
#     1. Eyeballing the overlaid two curves (signal and estimated motion artefacts)
#     2. Scatter plot
#     3. Correlation coefficient
#     4. cross-correlation and peak at T = 0
#     5. Sum of absolute difference per unit time???
#     6. VAR
# 
# 

# In[1]:


import os

nb_name = "nb20230510_180000_eval_motion_correction_loop.ipynb" #TODO change this

basename, ext = os.path.splitext(nb_name)
input_path = os.path.join(os.getcwd(), nb_name)

get_ipython().system('jupyter nbconvert "{input_path}" --to="python" --output="{basename}"')


# # 1. Specify target sessions
# 
# debug_folders: list of folder paths

# In[2]:


dir_by_sessions = r"\\ettin\Magill_lab\Julien\Data\head-fixed\by_sessions"

def join_task_session(taskname, sessionnames: list):
    return [os.path.join(dir_by_sessions, taskname, ssn) for ssn in sessionnames]

task1 = join_task_session('reaching_go_spout_bar_nov22', [
    'kms058-2023-03-24-151254',
    'kms058-2023-03-25-184034',
    'kms062-2023-02-21-103400',
    'kms062-2023-02-22-150828',
    'kms063-2023-04-09-183115',
    'kms063-2023-04-10-194331',
    'kms064-2023-02-13-104949',
    'kms064-2023-02-15-104438',
    'kms064-2023-02-16-103424',
    'RE602-2023-03-22-121414'])

task2 = join_task_session('reaching_go_spout_bar_dual_dec22', [
    'JC316L-2022-12-07-163252',
    'JC316L-2022-12-08-143046'])


debug_folders = task1 + task2


# In[3]:


[print(d) for d in debug_folders]


# # Prep data

# In[4]:


from trialexp.process.pycontrol.utils import export_session
from snakehelper.SnakeIOHelper import getSnake

# from workflow.scripts import settings
from re import match
from pathlib import Path
from trialexp.process.pyphotometry.utils import *

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg
import img2pdf

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class sinput_class():
    def __init__(self):
        # mock class to mimic the output of getSnake()
        self.photometry_folder = None
        self.pycontrol_dataframe = None
        self.pycontrol_folder = None

def load_and_prep_photom(debug_folder):
    # (sinput, soutput) = getSnake(locals(), 'workflow/pycontrol.smk',
    #                             os.path.join(debug_folder, '/processed/spike2.smrx'),
    #                             'export_spike2')
    sinput = sinput_class()
    sinput.photometry_folder = os.path.join(debug_folder,'pyphotometry')
    sinput.pycontrol_dataframe = os.path.join(debug_folder,'processed','df_pycontrol.pkl')
    sinput.pycontrol_folder = os.path.join(debug_folder,'pycontrol')

    # %% Photometry dict

    # fn = glob(sinput.photometry_folder+'\*.ppd')[0]
    fn = list(Path(sinput.photometry_folder).glob('*.ppd'))
    if fn == []:
        data_photometry = None
    else:
        fn = fn[0]
        data_photometry = import_ppd(fn)

        data_photometry = denoise_filter(data_photometry)
        data_photometry = motion_correction(data_photometry)
        data_photometry = compute_df_over_f(data_photometry, low_pass_cutoff=0.001)


    # no down-sampling here

    # %% Load data
    df_pycontrol = pd.read_pickle(sinput.pycontrol_dataframe)

    pycontrol_time = df_pycontrol[df_pycontrol.name == 'rsync'].time

    # assuming just one txt file
    pycontrol_txt = list(Path(sinput.pycontrol_folder).glob('*.txt'))

    with open(pycontrol_txt[0], 'r') as f:
        all_lines = [line.strip() for line in f.readlines() if line.strip()]

    count = 0
    print_lines = []
    while count < len(all_lines):
        # all_lines[count][0] == 'P'
        if bool(match('P\s\d+\s', all_lines[count])):
            print_lines.append(all_lines[count][2:])
            count += 1
            while (count < len(all_lines)) and not (bool(match('[PVD]\s\d+\s', all_lines[count]))):
                print_lines[-1] = print_lines[-1] + \
                    "\n" + all_lines[count]
                count += 1
        else:
            count += 1

    v_lines = [line[2:] for line in all_lines if line[0] == 'V']


    # %%
    if fn == []:
        photometry_times_pyc = None
    else:
        photometry_aligner = Rsync_aligner(
            pycontrol_time, data_photometry['pulse_times_2'])
        photometry_times_pyc = photometry_aligner.B_to_A(data_photometry['time'])

    # remove all state change event
    df_pycontrol = df_pycontrol.dropna(subset='name')
    df2plot = df_pycontrol[df_pycontrol.type == 'event']
    # state is handled separately with export_state, whereas parameters are handled vchange_to_text

    keys = df2plot.name.unique()

    photometry_keys = ['analog_1', 'analog_2',  'analog_1_filt', 'analog_2_filt',
                    'analog_1_est_motion', 'analog_1_corrected', 'analog_1_baseline_fluo',
                    'analog_1_df_over_f']

    return df_pycontrol, pycontrol_time, data_photometry, photometry_times_pyc, photometry_keys

# export_session(df_pycontrol, keys, 
#     data_photometry = data_photometry,
#     photometry_times_pyc = photometry_times_pyc,
#     photometry_keys = photometry_keys,
#     print_lines = print_lines,
#     v_lines = v_lines,
#     smrx_filename=soutput.spike2_file)


# In[5]:


data_list = []

for d in debug_folders:
    data_dict = dict(df_pycontrol=None, pycontrol_time=None, data_photometry=None, photometry_times_pyc=None, photometry_keys = None, debug_folder=None)
    data_dict['df_pycontrol'], data_dict['pycontrol_time'], data_dict['data_photometry'], \
        data_dict['photometry_times_pyc'], data_dict['photometry_keys'] = load_and_prep_photom(d)
    data_dict['debug_folder'] = d
    data_dict['session_ID'] = os.path.basename(d)

    data_list.append(data_dict)
df_data = pd.DataFrame(data_list)

df_data['subject_ID'] = [r['subject_ID'] if r is not None else None for r in df_data['data_photometry']]
df_data['date_time'] = [r['date_time']
                        if r is not None else None for r in df_data['data_photometry']]


# In[6]:


df_data['session_ID']


# # Processing and evaluation
# 
# cf. 
# 
# ```
# process/pyphotometry/utils.py/motion_corretion()
# ```
# 
# - Correlation Coefficients and scatter plots
# - Cross-correlation and measuring the peak at 0

# In[7]:


df_data.columns


# In[8]:


df_data['subject_ID']


# In[9]:


df_data.iloc[0, 0].columns


# In[10]:


# nan_indices = np.argwhere(np.isnan(photometry_times_pyc))
# T_nonan = np.delete(photometry_times_pyc, nan_indices)
# max_time_ms = T_nonan[-1]


def get_newTandY_orig(T, photometry_dict, name, max_time_ms, photometry_times_pyc):
    nan_indices = np.argwhere(np.isnan(photometry_times_pyc))
    T_nonan = np.delete(photometry_times_pyc, nan_indices)
    max_time_ms = T_nonan[-1]


    T = photometry_times_pyc  # not down-sampled yet

    nan_indices = np.argwhere(np.isnan(T))
    T_nonan = np.delete(T, nan_indices)

    Y = photometry_dict[name]
    Y_nonan = np.delete(Y, nan_indices)  # []
    max_time_ms = T_nonan[-1]

    # NOTE sampling_rate was originally 1000
    new_T = np.arange(0, max_time_ms, 1/1000*1000)
    new_Y = np.interp(new_T, T_nonan, Y_nonan)
    return new_T, new_Y


def get_newTandY_down(T, photometry_dict, name, max_time_ms, photometry_times_pyc):
    nan_indices = np.argwhere(np.isnan(photometry_times_pyc))
    T_nonan = np.delete(photometry_times_pyc, nan_indices)
    max_time_ms = T_nonan[-1]

    Tdown = [T[i] for i in range(0, len(T), 10)]  # down sampled time vector

    nan_indices = np.argwhere(np.isnan(Tdown))
    Tdown_nonan = np.delete(Tdown, nan_indices)

    Y = photometry_dict[name]
    Y_nonan = np.delete(Y, nan_indices)  # []

    # Need to use interp to accomodate data into Spike2 bins
    # NOTE sampling_rate is already downsampled by 10
    new_T = np.arange(0, max_time_ms, 1/photometry_dict['sampling_rate']*1000)
    new_Y = np.interp(new_T, Tdown_nonan, Y_nonan)
    return new_T, new_Y


# In[11]:


[print(k) for k in df_data.loc[0,'data_photometry'].keys()]


# In[12]:


lowpass_freq = 20
highpass_freq = 0.001  # 1000 s cycle
sampling_rate = 1000 #TODO assuming 

# trialexp\process\pyphotometry\utils.py
# see https://vscode.dev/github/juliencarponcy/trialexp/blob/fd1e0dcc857275cafa7f809a104fd60e73ce1458/trialexp/process/pyphotometry/utils.py#L51
b, a = get_filt_coefs(low_pass=lowpass_freq,
                      high_pass=highpass_freq,
                      sampling_rate=sampling_rate)


# # Functions

# 
# GTP-4
# 
# > In the context of digital filters, the roots of the polynomial represented by the filter coefficients are very important, because they determine the behavior of the filter. Specifically, the roots of the 'a' coefficients (which form the denominator of the filter's transfer function) are called the "**poles**" of the filter. The locations of these poles in the complex plane determine whether the filter is stable or not.
# 
# > If all poles are inside the unit circle (meaning their absolute value is less than 1), then the filter is stable. If any pole is outside the unit circle, then the filter is unstable. This is why we use `numpy.roots(a)` and `numpy.abs()` to check the stability of the filter.

# In[13]:


def check_stability(b, a):
    """Check the stability of a digital filter."""
    # Get the poles of the filter
    poles = np.roots(a)

    # Check if all poles are inside the unit circle
    return np.all(np.abs(poles) < 1)




# In[14]:


def plot_scatter_lowpass(i,j, ax ):
    # i = 0
    # j = 0
    plt.sca(ax[i, j])

    plt.plot(analog_2_filt, analog_1_filt, '+', color=(0.5,0.5,0.5), markersize=2, zorder=1)

    # Calculate the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(analog_2_filt, analog_1_filt, bins=30)

    # Calculate the bin centers from the bin edges
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Create a meshgrid of bin centers
    X, Y = np.meshgrid(x_centers, y_centers)

    # Create the custom colormap

    log_norm = mcolors.LogNorm(vmin=1e-10, vmax=None)

    # Plot the contour plot
    #plt.contourf(X, Y, hist.T, cmap=custom_cmap, norm=norm)
    cnt = plt.contourf(X, Y, hist.T, cmap='viridis', norm=log_norm, zorder=2)


    plt.xlabel('analog_2_filt, red')
    plt.ylabel('analog_1_filt, green')

    plt.colorbar(cnt, location='top')

    plt.title('Lowpass filtering only', y = 1.25)

    # linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        x=analog_2_filt, y=analog_1_filt)

    x0 = np.arange(np.min(analog_2_filt), np.max(analog_2_filt), 
                (np.max(analog_2_filt) - np.min(analog_2_filt))/1000)
    y0 = slope*x0 + intercept

    plt.plot(x0, y0, '-', color='red', linewidth=2, zorder=3)

    plt.text(0.5,0.9,f'$R^2=${r_value**2:.3f}', transform=plt.gca().transAxes, ha='left')


# PLS result is virtually identical to linear regression.
# 
# Not worth!
# 
# GTP-4
# 
# > If you ran `linregress()` and `PLSRegression(n_components=1)` on the same dataset and obtained virtually identical results, it suggests that a linear relationship between the predictor variables and the response variable is sufficient to explain the variation in the data, and that the relationship is not highly nonlinear or complex.
# 
# > `linregress()` performs a simple linear regression analysis that fits a straight line to the data, while `PLSRegression(n_components=1)` performs a partial least squares regression analysis that projects the data onto a lower-dimensional space to capture the linear relationship between the variables.
# 
# > When `n_components=1` is used in `PLSRegression()`, the model will use only one latent variable to model the relationship between the predictor variables and the response variable. This means that the model will have a low level of complexity and may underfit the data if the relationship between the variables is more complex. However, if the data exhibits a linear relationship, then using only one latent variable can be a good starting point to explore the relationship between the variables and identify the most important features.
# 
# >L In summary, if `linregress()` and `PLSRegression(n_components=1)` yield similar results, it suggests that a linear relationship between the predictor variables and the response variable is sufficient to explain the variation in the data. However, it is always a good idea to check the assumptions of the regression models and to evaluate the performance of the models using appropriate metrics such as R-squared, mean squared error, etc., before drawing conclusions about the relationship between the variables. Additionally, you may want to experiment with different values of `n_components` in `PLSRegression()` to find the optimal level of complexity for your data.

# In[15]:


def plot_scatter_bandpass(i, j, ax):

    plt.sca(ax[i, j])

    plt.plot(analog_2_bp, analog_1_bp, '+', color=(0.5,0.5,0.5),markersize=2, zorder=1)

    # Calculate the 2D histogram
    hist, x_edges, y_edges = np.histogram2d(analog_2_bp, analog_1_bp, bins=30)

    # Calculate the bin centers from the bin edges
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2

    # Create a meshgrid of bin centers
    X, Y = np.meshgrid(x_centers, y_centers)

    # Create the custom colormap

    log_norm = mcolors.LogNorm(vmin=1e-10, vmax=None)

    # Plot the contour plot
    #plt.contourf(X, Y, hist.T, cmap=custom_cmap, norm=norm)
    cnt = plt.contourf(X, Y, hist.T, cmap='viridis', norm=log_norm, zorder=2)


    df_data.loc[0,'data_photometry']['analog_1']
    plt.xlabel('analog_2_bp, red')
    plt.ylabel('analog_1_bp, green')

    cb1= plt.colorbar(cnt, location='top')

    ax[i,j].set_title('Bandpass filtering', y = 1.25)

    # linregress
    slope, intercept, r_value, p_value, std_err = linregress(
        x=analog_2_bp, y=analog_1_bp)

    x0 = np.arange(np.min(analog_2_bp), np.max(analog_2_bp), 
                (np.max(analog_2_bp) - np.min(analog_2_bp))/1000)
    y0 = slope*x0 + intercept

    plt.plot(x0, y0, '-', color='red', linewidth=2, zorder=3)

    analog_1_est_motion_bp = slope * analog_2_bp + intercept

    plt.text(0.5, 0.9, f'$R^2$={r_value**2:.2f}',
            transform=plt.gca().transAxes, ha='left')
    

    # PLS reqression

    X_train, X_test, y_train, y_test = train_test_split(
        analog_2_bp.reshape(-1, 1), analog_1_bp.reshape(-1, 1), test_size=0.2, random_state=42)

    # PLS regression model creation and learning
    pls = PLSRegression(n_components=1)
    pls.fit(X_train, y_train)

    # model prediction and evaluation
    y_pred = pls.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    plt.text(0.5,0.84,f'PLS $R^2=${r2:.2f}', transform=plt.gca().transAxes, ha='left')
    plt.text(0.5,0.78,f'PLS $MSE$={mse:.2f}', transform=plt.gca().transAxes, ha='left')

    plt.plot(X_test, y_pred, ':', color='yellow', linewidth=2)


    return analog_1_est_motion_bp


# In[16]:


def get_waveform_average(signal, window_before_ms, window_after_ms, trig_ind):

    # Example time series data and events
    # signal = analog_1_bp

    # Define the window around events to compute the average waveform
    # window_before_ms = 1000 * 3
    # window_after_ms = 1000 * 3

    # Initialize an empty array to store the waveform segments
    waveform_segments = []

    # Extract waveform segments around each event index
    for event_index in trig_ind:
        start_index = event_index - window_before_ms
        end_index = event_index + window_after_ms + 1
        segment = signal[start_index:end_index]
        waveform_segments.append(segment)

    # Stack the waveform segments and compute the average along the first axis
    waveform_segments = np.stack(waveform_segments)
    waveform_average = np.mean(waveform_segments, axis=0)
    waveform_std = np.std(waveform_segments, axis=0)
    waveform_sem = waveform_std / np.sqrt(waveform_segments.shape[0])
    sample_size = waveform_segments.shape[0]
    # print("Waveform average:", waveform_average)

    return waveform_segments, waveform_average, waveform_std, waveform_sem, sample_size


# In[17]:


def plot_waveform_average(i, j, ax, ylim: list = None):
    # ylim = [-0.015, 0.015]

    plt.sca(ax[i, j])

    plt.cla()

    plt.plot(T_vec, wa_analog_1_bp['waveform_average'], label='analog_1_bp', color='#2ca02c', ls = '-')
    plt.plot(T_vec, wa_analog_1_est_motion['waveform_average'], label='analog_1_est_motion', color='#2ca02c', ls='--')
    plt.plot(T_vec, wa_analog_1_corrected['waveform_average'], label='analog_1_corrected', color='#2ca02c', ls='-.')

    plt.plot(T_vec, wa_analog_2_bp['waveform_average'], label='analog_2_bp', color='#d62728', ls = '-')

    plt.plot(T_vec, wa_analog_1_est_motion_bp['waveform_average'],
            label='analog_1_est_motion_bp', color='#bcbd22', ls='--')
    plt.plot(T_vec, wa_analog_1_corrected_bp['waveform_average'],
            label='analog_1_corrected_bp', color='#bcbd22', ls='-.')


    plt.xlabel('Time relative to `analog_2_bp`\n exceeding $\pm$ 2SD after >3 s interval (s)')
    plt.ylabel('Fluorescence (V)')

    plt.gca().legend(loc='upper left', frameon=False)

    plt.text(0.75, 0.03, f"n = {wa_analog_1_bp['sample_size']:d} events", transform=plt.gca().transAxes)

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])

    plt.gca().legend(loc='upper left', frameon=False, fontsize=7)


# In[18]:


def plot_overview(i, j, ax):

    plt.sca(ax[i,j])
    plt.cla()

    plt.plot([t/1000 for t in trig_ms], [0 for _ in range(len(trig_ms))],'.')
    plt.plot([t/1000 for t in photometry_times_pyc], [v - mn for v in analog_1_filt],'-', label='analog_1_filt', color='#2ca02c', linewidth=0.75)
    plt.plot([t/1000 for t in photometry_times_pyc], [v - mn for v in analog_2_filt],'-', label='analog_2_filt', color='#d62728',linewidth=0.75)
    plt.plot([t/1000 for t in photometry_times_pyc], [v - mn for v in analog_1_bp],'--', label='analog_1_bp', color='#2ca02c',linewidth=0.75, alpha=0.5)
    plt.plot([t/1000 for t in photometry_times_pyc], [v - mn for v in analog_2_bp],'--', label='analog_2_bp', color='#d62728',linewidth=0.75, alpha=0.5)

    plt.xlim(np.nanmin(photometry_times_pyc)/1000, np.nanmax(photometry_times_pyc)/1000)
    plt.legend(loc='upper right', frameon=False, fontsize=7)
    plt.ylabel('Fluorescence (V)')
    plt.xlabel('Time (s)')
    plt.title('The whole session view')


# In[19]:


def plot_boxes(i, j,  ax):
    data = [analog_1_filt, analog_1_est_motion, analog_1_corrected, analog_2_filt,
            analog_1_bp, analog_1_est_motion_bp, analog_1_corrected_bp, analog_2_bp]

    plt.sca(ax[i, j])

    bx = plt.boxplot(data, showfliers=False)  # too many outliers

    for i in range(0, 8):
        plt.plot(i+1, np.nanmax(data[i]), 'o',
                markerfacecolor='none', color=[0.5, 0.5, 0.5])
        plt.plot(i+1, np.nanmin(data[i]), 'o',
                markerfacecolor='none', color=[0.5, 0.5, 0.5])

    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8], ['analog_1_filt', 'analog_1_est_motion', 'analog_1_corrected',  'analog_2_filt',
                                        'analog_1_bp', 'analog_1_est_motion_bp', 'analog_1_corrected_bp', 'analog_2_bp'])
    plt.xticks(rotation=60, ha='right')

    # [L.set_alpha(0.2) for L in bx['fliers']]
    plt.ylabel('Fluorescence (V)')
    plt.title('Data distributions')



# In[20]:


def find_top_n_events(top_n: int, mn: float, sd: float, interval_ms: float):

    for thre in np.linspace(2, 0.5, 16):
        ind = np.where(np.abs(analog_2_bp - mn) > thre * sd)[0]
        diffs = np.diff(ind)
        non_consecutive_positions = np.where(diffs != 1)[0]

        preceded_by_interval = positions = np.where(diffs > interval_ms)[0]
        # preceded by 3 s intervals was too much


        ind_ = np.array(sorted(list(set(non_consecutive_positions) & set(preceded_by_interval))))

        if len(ind_) == 0:# failed to find an event
            continue

        trig_ind = ind[ind_ + 1]

        trig_ms = photometry_times_pyc[trig_ind]

        if len(trig_ms) >= top_n:
            break

    
    if len(trig_ms) < top_n:
        # failed to find top_n events satisfying the condition
        print(f'failed to find {top_n:d} events satisfying the condition; only {len(trig_ms):d} found')

    return trig_ind, trig_ms
    


# # For Loop

# In[21]:


plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['font.family'] = ['Arial']


plt.close()
plt.interactive(True) #TODO not sure

cm = 1/2.54  # centimeters in inches
A4_portrait = (21*cm, 29.7*cm)


for r in tqdm(range(0, df_data.shape[0])):
    fig, ax = plt.subplots(3,2, figsize=A4_portrait)

    if df_data.loc[r,'subject_ID'] is  None:
        fig.suptitle(df_data.loc[r, 'subject_ID'])

        fig.savefig(f"temp{r:02d}.png", dpi=400) # save temp bitmap figure
        
        # Close the figure
        plt.close(fig)
        continue

    analog_1_filt = df_data.loc[r,'data_photometry']['analog_1']
    analog_2_filt = df_data.loc[r,'data_photometry']['analog_2']

    analog_1_est_motion = df_data.loc[r,'data_photometry']['analog_1_est_motion']
    analog_1_corrected = df_data.loc[r,'data_photometry']['analog_1_corrected']
    analog_1_df_over_f = df_data.loc[r, 'data_photometry']['analog_1_est_motion']

    #do_analysis(df_data, r)

    analog_1_bp = filtfilt(b, a, df_data.loc[r,'data_photometry']['analog_1'], padtype='even')
    analog_2_bp = filtfilt(b, a, df_data.loc[r,'data_photometry']['analog_2'], padtype='even')

    is_stable = check_stability(b, a)
    print(f"The filter is {'stable' if is_stable else 'unstable'}.")



    
    fig.subplots_adjust(hspace=0.4, wspace=0.5)

    plot_scatter_lowpass(0, 0, ax )

    analog_1_est_motion_bp = plot_scatter_bandpass(0, 1, ax)


    # triggered waveform average analyses
    photometry_times_pyc = df_data.loc[r, 'photometry_times_pyc']

    mn = np.mean(analog_2_bp)
    sd = np.std(analog_2_bp)

    trig_ind, trig_ms = find_top_n_events(30, mn, sd, 1000)
    # ind = np.where(np.abs(analog_2_bp - mn) > 1 * sd)[0]
    # diffs = np.diff(ind)
    # non_consecutive_positions = np.where(diffs != 1)[0]

    # preceded_by_interval = positions = np.where(diffs > 300)[0]
    # # preceded by 3 s intervals was too much


    # ind_ = np.array(sorted(list(set(non_consecutive_positions) & set(preceded_by_interval))))

    # trig_ind = ind[ind_ + 1]

    # trig_ms = photometry_times_pyc[trig_ind]


    window_before_ms = 1000 * 3
    window_after_ms = 1000 * 4

    keys = ['waveform_segments', 'waveform_average', 'waveform_std', 'waveform_sem', 'sample_size']

    wa_analog_1_bp = dict(zip(keys, get_waveform_average(analog_1_bp, window_before_ms, window_after_ms, trig_ind)))
    wa_analog_2_bp = dict(zip(keys, get_waveform_average(analog_2_bp, window_before_ms, window_after_ms, trig_ind)))

    wa_analog_1_est_motion = dict(zip(keys, get_waveform_average(analog_1_est_motion, window_before_ms, window_after_ms, trig_ind)))
    wa_analog_1_corrected = dict(zip(keys, get_waveform_average(analog_1_corrected, window_before_ms, window_after_ms, trig_ind)))

    wa_analog_1_est_motion_bp = dict(zip(keys, get_waveform_average(analog_1_est_motion_bp, window_before_ms, window_after_ms, trig_ind)))

    analog_1_corrected_bp = analog_1_bp - analog_1_est_motion_bp #TODO is this correct

    wa_analog_1_corrected_bp = dict(zip(keys, get_waveform_average(analog_1_corrected_bp, window_before_ms, window_after_ms, trig_ind)))

    T_vec = np.linspace(-1 * window_before_ms , window_after_ms, len(wa_analog_1_bp['waveform_average']))/1000

    plot_waveform_average(1, 0, ax)

    plot_waveform_average(1, 1, ax, [-0.015, 0.015])

    plot_overview(2, 0, ax)

    plot_boxes(2, 1,  ax)

    fig.suptitle(df_data.loc[r, 'session_ID'])

    fig.savefig(f"temp{r:02d}.png", dpi=400) # save temp bitmap figure
        
    plt.close(fig)


# In[22]:


import re

png_files = [f for f in os.listdir('.') if re.search(r'^temp\d{2}\.png$', f)]
png_files.sort()


# In[23]:


import img2pdf
with open("nb20230510_180000_eval_motion_correction_loop.pdf", "wb") as f:
    f.write(img2pdf.convert(png_files))

for r in range(0, df_data.shape[0]):
    os.remove(f"temp{r:02d}.png")


# Event-triggered waveform average is so far the most informative.
# 
# - **kms058** It's very clear that the original motion correction is adding more problem than correcting. The new method is much better.
# - **kms062-2023-02-21** R^2 is 0.94 and we don't see motion artefact in the data. Maybe some problem in recording.
# - **kms062-2023-02-22** Again, no artefact
# - **kms063-2023-04-09** Relatively big motion artefact. The new method is slightly better. But it doesn't make a lot of sense
# - **kms064-2023-02-13** Motion artefact only exist in red channel. 
# - **kms064-2023-02-15** New method is slightly better?
# - **RE602-2023-03-22** Both are equally bad.
# 
# Overall, the new method only outperformed for kms058 and in other cases the effect was not clear.
# It is worth changing the pipeline?
# 
# What about $\frac{\Delta F}{F}$?

# 
# `PdfPages` is able to save figures as vector graphics in multi-page PDF, but the file size can be huge when there are too many data points.
# 
# ```python
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt
# 
# with PdfPages('output.pdf') as pdf:
#     for i in range(5):
#         fig, ax = plt.subplots()
#         ax.plot(range(10), [j ** (i+1) for j in range(10)])  # example plot
#         pdf.savefig(fig)
#         plt.close(fig)
# ```
# 
# 
# `img2pdf` can convert multiple PNGs into multi-page PDF
# 
# ```python
# import img2pdf
# import os
# 
# # Get all png files in the current directory
# png_files = [f for f in os.listdir('.') if f.endswith('.png')]
# 
# # Sort the images by name or modify as required to get them in the order you want
# png_files.sort()
# 
# with open("output.pdf", "wb") as f:
#     f.write(img2pdf.convert(png_files))
# ```
