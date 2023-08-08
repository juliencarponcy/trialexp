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



def get_recordings_properties(ephys_base_path, fn):
    exp_dict = parse_openephys_folder(fn)

    # Explore folder with neo utilities for openephys
    folder_structure, all_streams, nb_block, nb_segment_per_block,\
        experiment_names = explore_folder(Path(ephys_base_path) / fn)


    # List continuous streams names
    try:
        continuous_streams = list(folder_structure['Record Node 101']['experiments'][1]['recordings'][1]['streams']['continuous'].keys())
        # Only select action potentials streams
        AP_streams = [AP_stream for AP_stream in continuous_streams if 'AP' in AP_stream]
        print(f'Nb of Experiments (blocks): {nb_block}\nNb of segments per block: {nb_segment_per_block}\nDefault exp name: {experiment_names}\n')
        print(f'Spike streams:{AP_streams}\n')
    except KeyError:
        print('Key error encountered at ', fn)
        raise KeyError


    # if len(experiment_names) > 1:
    #     raise NotImplementedError('More than one experiment in the open-ephys folder')
    
    recordings_properties = dict()

    for k in exp_dict.keys():
        recordings_properties[k] = list()

    recordings_properties['AP_stream'] = list()
    recordings_properties['AP_folder'] = list()
    recordings_properties['exp_nb'] = list()
    recordings_properties['rec_nb'] = list()
    recordings_properties['tstart'] = list()
    recordings_properties['sample_rate'] = list()
    recordings_properties['rec_start_datetime'] = list()
    recordings_properties['full_path'] = list()
    recordings_properties['sync_path'] = list()
    recordings_properties['duration'] = list()

    exp_keys = list(folder_structure['Record Node 101']['experiments'].keys())
    
    for exp_idx, exp_nb in enumerate(exp_keys):
        
        rec_keys = list(folder_structure['Record Node 101']['experiments'][exp_nb]['recordings'].keys())
        for idx, rec_nb in enumerate(rec_keys):

            for AP_stream in AP_streams:
                for k, v in exp_dict.items():
                    recordings_properties[k].append(v)

                recordings_properties['AP_stream'].append(AP_stream)
                recordings_properties['AP_folder'].append(split('#',AP_stream)[1])
                
                recordings_properties['exp_nb'].append(exp_nb)
                recordings_properties['rec_nb'].append(rec_nb)
                recordings_properties['tstart'].append(
                    folder_structure['Record Node 101']['experiments'][exp_nb]['recordings'][rec_nb]['streams']['continuous'][AP_streams[0]]['t_start']
                )
                recordings_properties['sample_rate'].append(
                    folder_structure['Record Node 101']['experiments'][exp_nb]['recordings'][rec_nb]['streams']['continuous'][AP_streams[0]]['sample_rate']
                )
                recordings_properties['rec_start_datetime'].append(
                    exp_dict['exp_datetime'] + timedelta(0, recordings_properties['tstart'][idx])
                )
                recordings_properties['full_path'].append(
                    Path(ephys_base_path) / fn / 'Record Node 101' / experiment_names[exp_idx] / ('recording' + str(rec_nb)) / 'continuous' / recordings_properties['AP_folder'][idx]
                )

                recordings_properties['sync_path'].append(
                    Path(ephys_base_path) / fn / 'Record Node 104' / experiment_names[exp_idx] / ('recording' + str(rec_nb)) / 'events' / 'NI-DAQmx-103.PXIe-6341' / 'TTL'
                )

                recordings_properties['duration'].append(
                    get_recording_duration(recordings_properties['full_path'][-1] , recordings_properties['sample_rate'][-1])
                )

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

        pycontrol_rsync = data_pycontrol[data_pycontrol.name=='rsync'].time.values
        
        try:
            return Rsync_aligner(pulse_times_A= rsync_ephys_ts*1000, 
            pulse_times_B= pycontrol_rsync, plot=False) #align pycontrol time to pyphotometry time
        except (RsyncError, ValueError) as e:
            return None