'''
Script to create the session folder structure
'''
#%%
import os
import re
import shutil
import warnings
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import time

from trialexp.utils.pycontrol_utilities import parse_pycontrol_fn, create_sync
from trialexp.utils.pyphotometry_utilities import parse_pyhoto_fn
from trialexp.utils.ephys_utilities import parse_openephys_folder, get_recordings_properties


from tqdm import tqdm

        
def copy_if_not_exist(src, dest):
    if not (dest/src.name).exists():
        shutil.copy(src, dest)


#%% Retrieve all task names from the tasks_params.csv
tasks_params_path = Path(os.getcwd()).parent.parent / 'params' / 'tasks_params.csv'
tasks_params_df = pd.read_csv(tasks_params_path)
tasks = tasks_params_df.task.values.tolist()

# %%

for task_id, task in enumerate(tasks):

    print(f'task {task_id+1}/{len(tasks)}: {task}')
    export_base_path = Path(f'/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/by_sessions/{task}')

    pycontrol_folder = Path(f'/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/pycontrol/{task}')
    pyphoto_folder = Path(f'/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/pyphotometry/data/{task}')
    ephys_base_path = '/home/MRC.OX.AC.UK/phar0732/ettin/Data/head-fixed/_Other/test_folder_ephys'

    pycontrol_files = list(pycontrol_folder.glob('*.txt'))
    pyphoto_files = list(pyphoto_folder.glob('*.ppd'))
    open_ephys_folders = os.listdir(Path(ephys_base_path))

    df_pycontrol = pd.DataFrame(list(map(parse_pycontrol_fn, pycontrol_files)))
    try:
        df_pycontrol = df_pycontrol[df_pycontrol.session_length>1000*60*5] #remove sessions that are too short
    except AttributeError:
        print('no session length for task {task}, skipping folder')
        continue

    df_pyphoto = pd.DataFrame(list(map(parse_pyhoto_fn, pyphoto_files)))
    all_parsed_ephys_folders = list(map(parse_openephys_folder, open_ephys_folders))
    # remove unsuccessful ephys folders parsing 
    parsed_ephys_folders = [result for result in all_parsed_ephys_folders if result is not None]
    df_ephys_exp = pd.DataFrame(parsed_ephys_folders)
    # Match
    #Try to match pycontrol file together with pyphotometry file

    matched_path = []
    matched_fn  = []

    for _, row in df_pycontrol.iterrows():
        
        # will only compute time diff on matching subject_id
        if not df_pyphoto.empty:
            df_pyphoto_subject = df_pyphoto[df_pyphoto.subject_id == row.subject_id]
        else:
            matched_path.append(None)
            matched_fn.append(None)
            continue

        if not df_pyphoto_subject.empty:
            min_td = np.min(abs(row.timestamp - df_pyphoto_subject.timestamp))
            idx = np.argmin(abs(row.timestamp - df_pyphoto_subject.timestamp))

            if min_td < timedelta(minutes=15):
                matched_path.append(df_pyphoto_subject.iloc[idx].path)
                matched_fn.append(df_pyphoto_subject.iloc[idx].filename)
            else:
                matched_path.append(None)
                matched_fn.append(None)
        else:
            matched_path.append(None)
            matched_fn.append(None)

    df_pycontrol['pyphoto_path'] = matched_path
    df_pycontrol['pyphoto_filename'] = matched_fn
    df_pycontrol = df_pycontrol[(df_pycontrol.subject_id!='00') & (df_pycontrol.subject_id!='01')] # do not copy the test data
    df_pycontrol = df_pycontrol.dropna(subset='pyphoto_path')

    for i in tqdm(range(len(df_pycontrol))):
        row = df_pycontrol.iloc[i]
        session_id = row.session_id
        subject_id = row.subject_id
        
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
