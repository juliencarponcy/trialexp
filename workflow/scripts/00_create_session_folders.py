'''
Script to create the session folder structure
'''
#%%
import os
import shutil
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from trialexp.utils.pycontrol_utilities import parse_pycontrol_fn
from trialexp.utils.pyphotometry_utilities import parse_pyhoto_fn, create_photo_sync
from trialexp.utils.ephys_utilities import parse_openephys_folder, get_recordings_properties, create_ephys_rsync
        
def copy_if_not_exist(src, dest):
    if not (dest/src.name).exists():
        shutil.copy(src, dest)


#%% Retrieve all task names from the tasks_params.csv
SESSION_ROOT_DIR = Path(os.environ['SESSION_ROOT_DIR'])
ETTIN_DATA_FOLDER = SESSION_ROOT_DIR.parents[1]
PROJECT_ROOT = Path(os.environ['SNAKEMAKE_DEBUG_ROOT'])

tasks_params_path = PROJECT_ROOT / 'params' / 'tasks_params.csv'
tasks_params_df = pd.read_csv(tasks_params_path)
tasks = tasks_params_df.task.values.tolist()


skip_existing = True #whether to skip existing folders
# %%

for task_id, task in enumerate(tasks):

    print(f'task {task_id+1}/{len(tasks)}: {task}')
    export_base_path = SESSION_ROOT_DIR/f'{task}'

    pycontrol_folder = ETTIN_DATA_FOLDER/'head-fixed'/'pycontrol'/f'{task}'
    pyphoto_folder = ETTIN_DATA_FOLDER/'head-fixed'/'pyphotometry'/'data'/f'{task}'
    ephys_base_path = ETTIN_DATA_FOLDER/'head-fixed'/'openephys'

    pycontrol_files = list(pycontrol_folder.glob('*.txt'))
    pyphoto_files = list(pyphoto_folder.glob('*.ppd'))
    open_ephys_folders = os.listdir(ephys_base_path)

    df_pycontrol = pd.DataFrame(list(map(parse_pycontrol_fn, pycontrol_files)))
    try:
        df_pycontrol = df_pycontrol[df_pycontrol.session_length>1000*60*5] #remove sessions that are too short
    except AttributeError:
        print(f'no session length for task {task}, skipping folder')
        continue

    df_pyphoto = pd.DataFrame(list(map(parse_pyhoto_fn, pyphoto_files)))
    all_parsed_ephys_folders = list(map(parse_openephys_folder, open_ephys_folders))
    # remove unsuccessful ephys folders parsing 
    parsed_ephys_folders = [result for result in all_parsed_ephys_folders if result is not None]
    df_ephys_exp = pd.DataFrame(parsed_ephys_folders)
    
    # Match
    #Try to match pycontrol file together with pyphotometry file
    matched_photo_path = []
    matched_photo_fn  = []
    matched_ephys_path = []
    matched_ephys_fn  = []
    
    df_pycontrol['do_copy'] = True
    
    if skip_existing:
        
        for i in range(len(df_pycontrol)):
            # filter out folders that are already there
            session_id = df_pycontrol.iloc[i].filename
            if Path(export_base_path, session_id).exists():
                df_pycontrol.loc[i, 'do_copy'] = False
                    
    df_pycontrol = df_pycontrol.loc[df_pycontrol.do_copy==True]
    for _, row in df_pycontrol.iterrows():
        
        # Photometry matching
        # will only compute time diff on matching subject_id
        # First identify the same animal
        if not df_pyphoto.empty:
            df_pyphoto_subject = df_pyphoto[df_pyphoto.subject_id == row.subject_id]
        else:
            matched_photo_path.append(None)
            matched_photo_fn.append(None)

        # find the closet match in time
        if not df_pyphoto_subject.empty:
            min_td = np.min(abs(row.timestamp - df_pyphoto_subject.timestamp))
            idx = np.argmin(abs(row.timestamp - df_pyphoto_subject.timestamp))

            if min_td < timedelta(minutes=15):
                matched_photo_path.append(df_pyphoto_subject.iloc[idx].path)
                matched_photo_fn.append(df_pyphoto_subject.iloc[idx].filename)
            else:
                matched_photo_path.append(None)
                matched_photo_fn.append(None)
        
        elif not df_pyphoto.empty and df_pyphoto_subject.empty:
            matched_photo_path.append(None)
            matched_photo_fn.append(None)

        # Ephys matching
        if not df_ephys_exp.empty:
            df_ephys_exp_subject = df_ephys_exp[df_ephys_exp.subject_id == row.subject_id]

            if not df_ephys_exp_subject.empty:
                min_td = np.min(abs(row.timestamp - df_ephys_exp_subject.exp_datetime))
                idx = np.argmin(abs(row.timestamp - df_ephys_exp_subject.exp_datetime))

                if min_td < timedelta(days=0.25):
                    matched_ephys_path.append(ephys_base_path / df_ephys_exp_subject.iloc[idx].foldername)
                    matched_ephys_fn.append(df_ephys_exp_subject.iloc[idx].foldername)
                else:
                    matched_ephys_path.append(None)
                    matched_ephys_fn.append(None)
            
            elif not df_ephys_exp.empty and df_ephys_exp_subject.empty:
                matched_ephys_path.append(None)
                matched_ephys_fn.append(None)

        else:
            matched_ephys_path.append(None)
            matched_ephys_fn.append(None)

    df_pycontrol['pyphoto_path'] = matched_photo_path
    df_pycontrol['pyphoto_filename'] = matched_photo_fn

    df_pycontrol['ephys_path'] = matched_ephys_path
    df_pycontrol['ephys_folder_name'] = matched_ephys_fn
    
    df_pycontrol = df_pycontrol[(df_pycontrol.subject_id!='00') & (df_pycontrol.subject_id!='01')] # do not copy the test data
    
    #TODO need to consider the case where there is only pycontrol data but no photometry
    df_pycontrol = df_pycontrol.dropna(subset='pyphoto_path')

    for i in tqdm(range(len(df_pycontrol))):
        row = df_pycontrol.iloc[i]
        session_id = row.session_id
        subject_id = row.subject_id
        
        target_pycontrol_folder = Path(export_base_path, session_id, 'pycontrol')
        target_pyphoto_folder = Path(export_base_path, session_id, 'pyphotometry')
        target_ephys_folder = Path(export_base_path, session_id, 'ephys')
        
        if not target_pycontrol_folder.exists():
            # create the base folder
            target_pycontrol_folder.mkdir(parents=True)
            
        if not target_pyphoto_folder.exists():
            target_pyphoto_folder.mkdir(parents=True)

        if not target_ephys_folder.exists():
            target_ephys_folder.mkdir(parents=True)
            
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
        if create_photo_sync(str(pycontrol_file), str(pyphotometry_file)) is not None:
            copy_if_not_exist(pyphotometry_file, target_pyphoto_folder)


        #write information about ephys recrodings in the ephys folder
        if row.ephys_folder_name:
            recordings_properties = get_recordings_properties(ephys_base_path, row.ephys_folder_name)
            # try to sync ephys recordings
            recordings_properties['syncable'] = False
            recordings_properties['longest'] = False
            sync_paths = recordings_properties.sync_path.unique()
            for sync_path in sync_paths:
                # copy syncing files in 
                if create_ephys_rsync(str(pycontrol_file), sync_path) is not None:
                    recordings_properties.loc[recordings_properties.sync_path == sync_path, 'syncable'] = True
            
            longest_syncable = recordings_properties.loc[recordings_properties.syncable == True, 'duration'].max()
            recordings_properties.loc[(recordings_properties.duration == longest_syncable) & (recordings_properties.syncable == True), 'longest'] = True

            sync_path = recordings_properties.loc[recordings_properties.longest == True, 'sync_path'].unique()
            
            if len(sync_path) > 1:
                raise NotImplementedError(f'multiple valids sync_path for the session, something went wrong: {row.ephys_folder_name}')
            
            # copy sync files from the longest syncable recording
            elif len(sync_path) == 1:

                copy_if_not_exist(sync_path[0] / 'states.npy', target_ephys_folder)
                copy_if_not_exist(sync_path[0] / 'timestamps.npy', target_ephys_folder)

            else:
                # no syncable recordings
                ...

            recordings_properties.to_csv(target_ephys_folder / 'rec_properties.csv')

            
# %%
