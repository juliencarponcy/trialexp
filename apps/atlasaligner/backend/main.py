#%%
import fastapi
from fastapi import HTTPException
from enum import Enum
from typing import Optional
from trialexp.process.folder_org.utils import build_session_info_cohort
from trialexp.process.anatomy.utils import load_ccf_data, shift_trajectory_depth, get_region_boundaries,\
    trajectory2probe_coords, get_trajectory_areas, load_probe_dates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
from datetime import datetime
from pydantic import BaseModel
import os 
from dotenv import load_dotenv
from loguru import logger
#%%
load_dotenv()
root_path = os.environ['SESSION_ROOT_DIR']
df_session_info = build_session_info_cohort(root_path)

print('Backend server starting')

app = fastapi.FastAPI(root_path='/api')
app.state.df_session_info = df_session_info.query('neuropixels_sorted==True').sort_values('expt_datetime', ascending=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    
@app.get('/sessions')
async def get_sessions(cohort: Optional[str] = None, animal_id: Optional[str] = None):
    query_str = ''
    
    if cohort:
        query_str += f"cohort=='{cohort}'"

    if animal_id:
        query_str += f'and animal_id=="{animal_id}"'
    
    df = app.state.df_session_info
    
    if len(query_str) > 0:
        df = df.query(query_str)
        
    # only return session with sorted neuropixel data
    # df = df.query('neuropixels_sorted==True')
        
    return {"session_id": df.session_id.unique().tolist()}

@app.get('/cohorts')
async def get_cohort():
    df = app.state.df_session_info
    return {"cohorts": df.cohort.unique().tolist()}


@app.get('/animal_id')
async def get_animal_id(cohort: Optional[str] = None):
    df = app.state.df_session_info
    if cohort:
        df = df.query(f"cohort=='{cohort}'")
    return {"animal_id": df.animal_id.unique().tolist()}


@app.get('/trajectory/{session_id}')
async def get_trajectory(session_id: str, shift:int = 0):
    df = app.state.df_session_info
    df = df.query(f"session_id=='{session_id}'")
    df = df.iloc[0]
    local_results = Path(root_path)/df.cohort/'histology'/df.animal_id/'RGB'/'Processed'
    probe_ccf = sio.loadmat(str(local_results/'probe_ccf.mat'), simplify_cells=True)['probe_ccf']
    
    # Main logic
    probes_name = load_probe_dates(local_results/'probe_names.txt')
    trajectory_areas=get_trajectory_areas(df, shift, root_path)
    return trajectory_areas[['acronym','depth_start','track_date','depth_end','name']].to_dict(orient='records')


@app.get('/cell_metrics/{session_id}')
async def get_firing_rate(session_id:str, bin_size:int=0):
    df = app.state.df_session_info
    df = df.query(f"session_id=='{session_id}'")
    df = df.query('neuropixels_sorted==True')
    
    if len(df)>0:
        df = df.iloc[0]
        df_quality_metrics = pd.read_pickle(df.path/'processed/df_quality_metrics.pkl')
        if bin_size>0:
            bins = np.arange(0, df_quality_metrics['ks_chan_pos_y'].values.max()+bin_size, bin_size)
            df_quality_metrics['pos_y_bin'] = pd.cut(df_quality_metrics['ks_chan_pos_y'], bins)
            df_fr = df_quality_metrics.groupby('pos_y_bin')['firing_rate'].agg(['mean','count']).reset_index()
            df_fr['pos_y_bin'] = bins[:-1]
        else:
            df_fr = df_quality_metrics.groupby('ks_chan_pos_y')['firing_rate'].agg(['mean','count']).reset_index()
            df_fr['pos_y_bin'] = df_fr['ks_chan_pos_y']
        
        df_fr = df_fr.fillna(0)
        d =  df_fr.to_dict(orient='list')
        
        # determine if a aligned_trajectory file is already present, if so , send over the shift
        # Find the correct file to load
        local_results = Path(root_path)/df.cohort/'histology'/df.animal_id/'RGB'/'Processed'
        probes_name = load_probe_dates(local_results/'probe_names.txt')
        probe_idx = np.where(probes_name == df.expt_datetime.date())[0]
        if len(probe_idx) == 0:
            msg = f'Probe for date {df.expt_datetime.date()} not found in probe_names.txt'
            logger.error(msg)
            raise HTTPException(status_code=404, detail=msg)
        probe_date = probes_name[probe_idx[0]]
        date_str = probe_date.strftime('%Y-%m-%d')
        trajectory_file = Path(local_results/f'aligned_trajectory_{date_str}.pkl')
        print(trajectory_file)
        if trajectory_file.exists():
            trajectory_file = pd.read_pickle(trajectory_file)
            d['shift'] = trajectory_file.attrs['shift']
            
        return d

class ShiftData(BaseModel):

    shift: int
    session_id: str


@app.post('/save_shift/')
async def save_shift(data:ShiftData):
    # pydantic model or Body is the resquest body data, argument with default value is query data
    # argument that appears in the end point path is the path variable
    df = app.state.df_session_info.query(f"session_id=='{data.session_id}'")
    df = df.iloc[0]
    
    local_results = Path(root_path)/df.cohort/'histology'/df.animal_id/'RGB'/'Processed'
    
    trajectory_areas = get_trajectory_areas(df, data.shift, root_path)
    track_date = trajectory_areas.iloc[0].track_date.replace('/','-')
    trajectory_areas.to_pickle(local_results/f'aligned_trajectory_{track_date}.pkl')
    
    return {'message':'data saved','shift':data.shift, 'session_id':data.session_id}
    