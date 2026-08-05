from re import search, split
from pathlib import Path
from datetime import datetime, timedelta
import warnings

import pandas as pd
import numpy as np

from neo.rawio.openephysbinaryrawio import explore_folder

from trialexp.process.pycontrol.data_import import session_dataframe
from trialexp.utils.rsync import *

def parse_openephys_folder(fn):
    m = split('_', fn)
    if isinstance(m,list) and len(m) >=3:
        subject_id = m[0]

        date_string = m[1]
        time_string = m[2]
        try:
            expt_datetime = datetime.strptime(date_string + '_' + time_string, "%Y-%m-%d_%H-%M-%S")
            return {'subject_id': subject_id, 
                'foldername':fn, 
                'exp_datetime':expt_datetime}
        except ValueError:
            pass


def get_continuous_stream_names(folder_structure):
    # get the names of the continous stream
    first_expt_key = list(folder_structure['Record Node 101']['experiments'].keys())[0]
    first_expt = folder_structure['Record Node 101']['experiments'][first_expt_key]
    
    first_recording_key = list(first_expt['recordings'].keys())[0]
    first_recording = first_expt['recordings'][first_recording_key]
    return list(first_recording['streams']['continuous'].keys())


def get_recordings_properties(ephys_base_path, fn):
    exp_dict = parse_openephys_folder(fn)

    # Explore folder with neo utilities for openephys
    folder_structure, all_streams, nb_block, nb_segment_per_block,\
        experiment_names = explore_folder(Path(ephys_base_path) / fn)


    # List continuous streams names
    try:
        continuous_streams = get_continuous_stream_names(folder_structure)
        # Only select action potentials streams
        AP_streams = [AP_stream for AP_stream in continuous_streams if 'AP' in AP_stream]
        # print(f'Nb of Experiments (blocks): {nb_block}\nNb of segments per block: {nb_segment_per_block}\nDefault exp name: {experiment_names}\n')
        # print(f'Spike streams:{AP_streams}\n')
    except KeyError:
        print('Key error encountered at ', fn)
        raise KeyError
    
    recordings_properties= []
    # use Neo's indexing logic instead of the folder structure
    
    for block_index in range(nb_block):
        for seg_index in range(nb_segment_per_block[block_index]):
            for AP_stream in AP_streams:
                rec_prop = {}

                cur_stream = all_streams[block_index][seg_index]['continuous'][AP_stream]
                
                rec_prop['AP_stream'] = AP_stream
                rec_prop['AP_folder']= AP_stream.split('#')[1]
                rec_prop['block_index'] = block_index
                rec_prop['seg_index'] = seg_index
                rec_prop['tstart'] = cur_stream['t_start']
                rec_prop['sample_rate'] = cur_stream['sample_rate']
                rec_prop['rec_start_datetime'] = exp_dict['exp_datetime'] + timedelta(0, rec_prop['tstart'])
                rec_prop['full_path'] = Path(cur_stream['raw_filename']).parent
                
                try:
                    event_info = all_streams[block_index][seg_index]['events']
                    if 'Record Node 104#TTL' in event_info:
                        sync_path = Path(all_streams[block_index][seg_index]['events']['Record Node 104#TTL']['timestamps_npy']).parents[2]
                    else:
                        sync_path = Path(all_streams[block_index][seg_index]['events']['Record Node 104#NI-DAQmx-103.PXIe-6341']['timestamps_npy']).parents[2]

                    rec_prop['sync_path'] = sync_path/'NI-DAQmx-103.PXIe-6341' / 'TTL'
                except KeyError:
                    print(f'Cannot find sync info for {fn}')
                    print(f"I only have {all_streams[block_index][seg_index]['events'].keys()}")
                    rec_prop['sync_path'] = None
                    
                rec_prop['duration'] = int(get_recording_duration(rec_prop['full_path'], cur_stream['sample_rate']))
                
                # get the expt_no and recording number from the path
                rec_nb = int(rec_prop['full_path'].parts[-3].replace('recording',''))
                exp_nb  = int(rec_prop['full_path'].parts[-4].replace('experiment',''))
                
                rec_prop['rec_nb'] = rec_nb
                rec_prop['exp_nb'] = exp_nb
                
                recordings_properties.append(rec_prop)
                
    recordings_properties = pd.DataFrame(recordings_properties)
            
    for k, v in exp_dict.items():
        recordings_properties[k] = v
    return pd.DataFrame(recordings_properties)

def get_recording_duration(rec_path: str, sample_rate: int):
    timestamps = np.load(Path(rec_path) / 'timestamps.npy')
    duration = len(timestamps) / sample_rate
    return duration

def create_ephys_rsync(pycontrol_file: str, sync_path: str, ephys_start_time: float = 0, rsync_ephys_chan_idx: int = 2):
    event_array = np.load(Path(sync_path, 'states.npy'))
    ts_array = np.load(Path(sync_path, 'timestamps.npy')) - ephys_start_time
        
    rsync_ephys_ts = ts_array[event_array == rsync_ephys_chan_idx]
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data_pycontrol = session_dataframe(pycontrol_file)
        
        if 'subtype' in data_pycontrol.columns:
            pycontrol_rsync = data_pycontrol[data_pycontrol.subtype=='sync'].time.values
        else:
            pycontrol_rsync = data_pycontrol[data_pycontrol.name=='rsync'].time.values
        # print(pycontrol_rsync)
        
        try:
            return Rsync_aligner(pulse_times_A= rsync_ephys_ts*1000, 
            pulse_times_B= pycontrol_rsync, plot=False) 
        except (RsyncError, ValueError) as e:
            return None
        
  
        
def match_ephys(df_pycontrol, df_ephys_exp, ephys_base_path):
    matched_ephys_path = []
    matched_ephys_fn  = []
    
    for _, row in df_pycontrol.iterrows():
        if not df_ephys_exp.empty:
            df_ephys_exp_subject = df_ephys_exp[df_ephys_exp.subject_id == row.subject_id]
            if not df_ephys_exp_subject.empty:
                
                # need to be more careful about the matching because ephys can start
                # much earlier than pycontrol session
                # find all potential match, choose the one that is earlier and closest
                td = (row.timestamp - df_ephys_exp_subject.exp_datetime)
                td = np.array([t.total_seconds() for t in td])
                df_ephys_exp_subject = df_ephys_exp_subject[td>=-1] # pycontrol is later
                
                if len(df_ephys_exp_subject) > 0:
                    min_td = np.min(abs(row.timestamp - df_ephys_exp_subject.exp_datetime))
                    idx = np.argmin(abs(row.timestamp - df_ephys_exp_subject.exp_datetime))
                    if min_td < timedelta(days=0.25):
                        matched_ephys_path.append(ephys_base_path / df_ephys_exp_subject.iloc[idx].foldername)
                        matched_ephys_fn.append(df_ephys_exp_subject.iloc[idx].foldername)
                        continue
    
        # some error occur, append None
        matched_ephys_path.append(None)
        matched_ephys_fn.append(None)
    
    df_pycontrol['ephys_path'] = matched_ephys_path
    df_pycontrol['ephys_folder_name'] = matched_ephys_fn