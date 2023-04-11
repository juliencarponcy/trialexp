'''
Script to create the session folder structure
'''
#%%
import os
# import re
import shutil
import warnings
# from datetime import datetime, timedelta
# from glob import glob
from pathlib import Path

import numpy as np
# import pandas as pd
from tqdm.auto import tqdm
from tqdm import tqdm

from snakehelper.SnakeIOHelper import getSnake
from pathlib import Path
from workflows.scripts import settings

# Extractors from SpikeInterface
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
# from spikeinterface import concatenate_recordings
from spikeinterface.sorters.kilosort3 import Kilosort3Sorter
# Neo class and methods for openephys data extraction/exploration
from neo.rawio.openephysbinaryrawio import explore_folder

#%% Define hard-coded variables

Kilosort3Sorter.set_kilosort3_path('/home/MRC.OX.AC.UK/phar0732/Documents/GitHub/Kilosort')

ks3_outputs = [
'spike_templates.npy',
 'templates_ind.npy',
 'kilosort3_master.m',
 'channel_positions.npy',
 'params.py',
 'run_kilosort3.sh',
 'kilosort3.log',
 'cluster_Amplitude.tsv',
 'temp_wh.dat',
 'writeNPY.m',
 'whitening_mat.npy',
 'cluster_ContamPct.tsv',
 'whitening_mat_inv.npy',
 'amplitudes.npy',
 'spike_clusters.npy',
 'spike_times.npy',
 'constructNPYheader.m',
 'cluster_KSLabel.tsv',
 'templates.npy',
 'chanMap.mat',
 'similar_templates.npy',
 'cluster_group.tsv',
 'channel_map.npy',
 'ops.mat']
#%% Load inputs

(sinput, soutput) = getSnake(locals(), 'workflows/spikesort.smk',
  [settings.debug_folder + 'spike_templates.npy'],
  'spikesort')

#%% 
ephys_base_path = Path('/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/_Other/test_folder_ephys')
sorted_base_path = Path('/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/_Other/test_folder_ephys/sorted')

ephys_folders = os.listdir(ephys_base_path)


#%% 

# Loop for all open_ephys base folders
for ephys_folder in ephys_folders:
    
    # Explore folder with neo utilities for openephys
    folder_structure, all_streams, nb_block, nb_segment_per_block,\
    possible_experiment_names = explore_folder(ephys_folder)
    
    print(f'Nb of Experiments (blocks): {nb_block}\nNb of segments per block: {nb_segment_per_block}\nDefault exp name: {possible_experiment_names}')

    exps = os.listdir(ephys_base_path / ephys_folder)
    if len(exp) > 1:
        print('Open Ephys folder contains more than one experiment')
    for exp in exps:

#%%
def parse_pycontrol_fn(fn):
    pattern = r'(\w+)-(.*)\.txt'
    m = re.search(pattern, fn.name)
    if m:
        animal_id = m.group(1)
        date_string = m.group(2)
        expt_datetime = datetime.strptime(date_string, "%Y-%m-%d-%H%M%S")
        
        try:
            df = session_dataframe(fn)
            session_length = df.time.iloc[-1]
            
            return {'animal_id':animal_id, 
                    'path':fn,                 
                    'session_id':fn.stem,
                    'filename':fn.stem, 
                    'timestamp':expt_datetime,
                    'session_length': session_length }
        except KeyError:
            return {'animal_id':animal_id, 
                    'path':fn,                 
                    'session_id':fn.stem,
                    'filename':fn.stem, 
                    'timestamp':expt_datetime,
                    'session_length': 0 }
   
    
df_pycontrol = pd.DataFrame(list(map(parse_pycontrol_fn, pycontrol_files)))
df_pycontrol = df_pycontrol[df_pycontrol.session_length>1000*60*5] #remove sessions that are too short
#%% Match
#Try to match pycontrol file together with pyphotometry file

matched_path = []
matched_fn  =[]

for _, row in df_pycontrol.iterrows():
    min_td = np.min(abs(row.timestamp - df_pyphoto.timestamp))
    idx = np.argmin(abs(row.timestamp - df_pyphoto.timestamp))

    if min_td< timedelta(minutes=5):
        matched_path.append(df_pyphoto.iloc[idx].path)
        matched_fn.append(df_pyphoto.iloc[idx].filename)
    else:
        matched_path.append(None)
        matched_fn.append(None)

df_pycontrol['pyphoto_path'] = matched_path
df_pycontrol['pyphoto_filename'] = matched_fn
df_pycontrol = df_pycontrol[df_pycontrol.animal_id!='00'] # do not copy the test data
df_pycontrol = df_pycontrol.dropna(subset='pyphoto_path')

# %%
def create_sync(pycontrol_file, pyphotometry_file):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pyphotometry_file = import_ppd(pyphotometry_file)
        data_pycontrol = session_dataframe(pycontrol_file)

        photo_rsync = pyphotometry_file['pulse_times_2']
        pycontrol_rsync = data_pycontrol[data_pycontrol.name=='rsync'].time
        
        try:
            return Rsync_aligner(pulse_times_A= photo_rsync, 
            pulse_times_B= pycontrol_rsync, plot=False) #align pycontrol time to pyphotometry time
        except (RsyncError, ValueError):
            return None
        
def copy_if_not_exist(src, dest):
    if not (dest/src.name).exists():
        shutil.copy(src, dest)

# %%
# matched = []
# x = df_pycontrol[df_pycontrol.session_id=="RE602-2023-03-17-150753"]

for i in tqdm(range(len(df_pycontrol))):
    row = df_pycontrol.iloc[i]
    session_id = row.session_id
    animal_id = row.animal_id
    
    target_pycontrol_folder = Path(export_base_path, session_id, 'pycontrol')
    target_pyphoto_folder = Path(export_base_path, session_id, 'pyphotometry')
    
    if not target_pycontrol_folder.exists():
        # create the base folder
        target_pycontrol_folder.mkdir(parents=True)
        
    if not target_pyphoto_folder.exists():
        target_pyphoto_folder.mkdir(parents=True)
        
    pycontrol_file = row.path
    pyphotometry_file = row.pyphoto_path
    
    #copy the pycontrol files
    # print(pycontrol_file, target_pycontrol_folder)
    copy_if_not_exist(pycontrol_file, target_pycontrol_folder)
    
    #copy all the analog data
    analog_files = pycontrol_file.parent.glob(f'{session_id}*.pca')
    for f in analog_files:
        copy_if_not_exist(f, target_pycontrol_folder) 
        
    #Copy pyphotometry file if they match
    if create_sync(str(pycontrol_file), str(pyphotometry_file)) is not None:
        copy_if_not_exist(pyphotometry_file, target_pyphoto_folder)

# %%
