'''
Script to create the session folder structure
'''
#%%
from pathlib import Path
from glob import glob
import os
import shutil
from tqdm.auto import tqdm
from datetime import datetime. Tim
import re
import pandas as pd
import numpy as np 
#%% 
export_base_path = Path('Z:\Teris\ASAP\expt_sessions')
pycontrol_folder = Path('Z:\\Teris\\ASAP\\pycontrol\\reaching_go_spout_bar_nov22')
pyphoto_folder = Path('Z:\\Teris\\ASAP\\pyphotometry\\reaching_go_spout_bar_nov22')

pycontrol_files = list(pycontrol_folder.glob('*.txt'))
pyphoto_files = list(pyphoto_folder.glob('*.ppd'))


#%% Build a dataframe of the photometry files for matching later
def parse_pyhoto_fn(fn):
    pattern = r'(\w+)-(.*)\.ppd'
    m = re.search(pattern, fn.name)
    if m:
        animal_id = m.group(1)
        date_string = m.group(2)
        expt_datetime = datetime.strptime(date_string, "%Y-%m-%d-%H%M%S")
        
        return {'animal_id':animal_id, 'path':fn, 'filename':fn.stem, 'timestamp':expt_datetime}

df_pyphoto = pd.DataFrame(list(map(parse_pyhoto_fn, pyphoto_files)))

#%%
def parse_pycontrol_fn(fn):
    pattern = r'(\w+)-(.*)\.txt'
    m = re.search(pattern, fn.name)
    if m:
        animal_id = m.group(1)
        date_string = m.group(2)
        expt_datetime = datetime.strptime(date_string, "%Y-%m-%d-%H%M%S")
        
        return {'animal_id':animal_id, 'path':fn, 'filename':fn.stem, 'timestamp':expt_datetime}
    
df_pycontrol = pd.DataFrame(list(map(parse_pycontrol_fn, pycontrol_files)))

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
df_pycontrol = df_pycontrol.dropna(subset='pyphoto_path')
# %%
for fn in tqdm(pycontrol_files):
    session_id = fn.stem
    animal_id = session_id.split('-')[0]
    
    if not animal_id=='00': # do not copy the test data
        target_pycontrol_folder = Path(export_base_path, session_id, 'pycontrol')
        target_pyphoto_folder = Path(export_base_path, session_id, 'pyphotometry')
        
        if not target_pycontrol_folder.exists():
            # create the base folder
            target_pycontrol_folder.mkdir(parents=True)
            target_pyphoto_folder.mkdir(parents=True)
            
            
            shutil.copy(fn, target_pycontrol_folder)
            
            #copy all the analog data
            analog_files = fn.parent.glob(f'{session_id}*.pca')
            for f in analog_files:
                shutil.copy(f, target_pycontrol_folder)
                
                
            # Next work on matching the pyphotometry folder
            
        else:
            print(f'{session_id} already exist')
        

# %%
