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
from trialexp.process.pycontrol.data_import import session_dataframe
from trialexp.process.pyphotometry.utils import import_ppd_auto, get_dataformat

from trialexp.utils.pycontrol_utilities import match_video, move_folders, parse_pycontrol_fn, parse_video_fn, get_df_video
from trialexp.utils.pyphotometry_utilities import match_photometry, parse_pyhoto_fn, create_photo_sync 
from trialexp.utils.ephys_utilities import match_ephys, parse_openephys_folder, get_recordings_properties, create_ephys_rsync
from trialexp.process.pycontrol.utils import auto_load_dotenv
from loguru import logger
from trialexp import config

#%% Retrieve all task names from the tasks_params.csv
SESSION_ROOT_DIR = Path(config.SESSION_ROOT_DIR)
ETTIN_DATA_FOLDER = Path(config.ETTIN_DATA_FOLDER)
PROJECT_ROOT = Path(config.SNAKEMAKE_DEBUG_ROOT)

tasks_params_path = PROJECT_ROOT / 'params' / 'tasks_params.csv'
tasks_params_df = pd.read_csv(tasks_params_path)
tasks = tasks_params_df.task.values.tolist()

skip_existing = True #whether to skip existing folders

# cohort to copy, if empty then search for all cohorts
cohort_to_copy = ['2025_July_cohort','2025_June_cohort','2025_October_cohort','2025_September_cohort','2026_January_cohort','2026_April_cohort'] 

# %%

for cohort_id, cohort in enumerate(cohort_to_copy):

    print(f'cohort {cohort_id+1}/{len(cohort_to_copy)}: {cohort}')
    export_base_path = SESSION_ROOT_DIR/f'{cohort}'/'by_sessions'

    pycontrol_folder = SESSION_ROOT_DIR/f'{cohort}'/'pycontrol'
    pyphoto_folder = SESSION_ROOT_DIR/f'{cohort}'/'pyphotometry'
    ephys_base_path = ETTIN_DATA_FOLDER/'head-fixed'/'neuropixels'
    video_folder = ETTIN_DATA_FOLDER/'head-fixed'/'videos'

    # Gather all pycontrol, photometry, and ephys files/folders 
    pycontrol_files = list(pycontrol_folder.glob('*.txt')) + list(pycontrol_folder.glob('*.tsv'))
    pyphoto_files = list(pyphoto_folder.glob('*.ppd'))
    open_ephys_folders = os.listdir(ephys_base_path)
    df_video = get_df_video(video_folder)


    # Do some quality checks
    df_pycontrol = pd.DataFrame(list(map(parse_pycontrol_fn, pycontrol_files)))
    assert len(df_pycontrol) == len(pycontrol_files)
    
    df_pycontrol = df_pycontrol[(df_pycontrol.subject_id!='00') & (df_pycontrol.subject_id!='01')] # do not copy the test data

    try:
        df_pycontrol = df_pycontrol[df_pycontrol.session_length>1000*60*3] #remove sessions that are too short, v2 uses second as unit
    except AttributeError:
        print(f'no session length, skipping folder')
        continue

    df_pyphoto = pd.DataFrame(list(map(parse_pyhoto_fn, pyphoto_files)))
        
    all_parsed_ephys_folders = list(map(parse_openephys_folder, open_ephys_folders))
    
    # remove unsuccessful ephys folders parsing 
    parsed_ephys_folders = [result for result in all_parsed_ephys_folders if result is not None]
    df_ephys_exp = pd.DataFrame(parsed_ephys_folders)

    
    # Match
    df_pycontrol['do_copy'] = True
    
    if skip_existing:
        
        for i in df_pycontrol.index:
            # filter out folders that are already there
            session_id = df_pycontrol.loc[i].filename
            task_name = df_pycontrol.loc[i].task_name
            if Path(export_base_path, task_name, session_id).exists():
                df_pycontrol.loc[i, 'do_copy'] = False
                    
    df_pycontrol = df_pycontrol[df_pycontrol.do_copy==True]
    # df_pycontrol= df_pycontrol[df_pycontrol.subject_id == 'TT008']
    # df_pycontrol= df_pycontrol[df_pycontrol.session_id == 'TT008-2024-06-10-153517']
    # df_pycontrol= df_pycontrol[df_pycontrol.session_id.str.startswith('TT008')]
    
    # match photometry signals
    match_photometry(df_pycontrol, df_pyphoto)
    
    # match videos
    match_video(df_pycontrol, df_video)
    
    # match ephys
    match_ephys(df_pycontrol, df_ephys_exp, ephys_base_path)
    
    
    # Actually move the files
    move_folders(df_pycontrol, export_base_path, ephys_base_path)
    
# %%
