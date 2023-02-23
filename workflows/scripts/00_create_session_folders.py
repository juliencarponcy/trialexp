'''
Script to create the session folder structure
'''
#%%
from pathlib import Path
from glob import glob
import os
import shutil
from tqdm.auto import tqdm
#%% 
export_base_path = Path('Z:\Teris\ASAP\expt_sessions')
pycontrol_folder = Path('Z:\\Teris\\ASAP\\pycontrol\\reaching_go_spout_bar_nov22_test')
pycontrol_files = pycontrol_folder.glob('*.txt')

# %%
for fn in tqdm(pycontrol_files):
    session_id = fn.stem
    animal_id = session_id.split('-')[0]
    
    if not animal_id=='00': # do not copy the test data
        session_path = Path(export_base_path, session_id, 'pycontrol')
        
        if not session_path.exists():
            # create the base folder
            session_path.mkdir(parents=True)
            
            shutil.copy(fn, session_path)
            
            #copy all the analog data
            analog_files = fn.parent.glob(f'{session_id}*.pca')
            for f in analog_files:
                shutil.copy(f, session_path)
        else:
            print(f'{session_id} already exist')
        

# %%
