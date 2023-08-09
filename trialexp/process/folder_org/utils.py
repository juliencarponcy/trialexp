'''
This contains various helper function for organizing folders and extracting meta data

'''
from glob import glob
from pathlib import Path
from datetime import datetime 
import re 
import pandas as pd 
from tqdm.auto import tqdm
import xarray as xr
import numpy as np 
import os

def load_pycontrol_variables(session_path, parameters, param_extract_method='tail'):
    session_id = session_path.name

    try:
        df_pycontrol = pd.read_pickle(session_path/'processed'/'df_pycontrol.pkl')
        # extract the parameter change, reshape them, and get the first/last value
        df_parameters = df_pycontrol[df_pycontrol.type=='parameters']
        df_parameters= df_parameters[df_parameters['name'].isin(parameters)]
        df_parameters = df_parameters.pivot(columns=['name'], values='value')
        df_parameters = df_parameters.fillna(method='ffill')
        df_parameters = df_parameters.dropna()
        
        if df_parameters.empty:
            # if no parameter is found, return a dataframe filled with NaN
            df_parameters = pd.DataFrame([{p:pd.NA for p in parameters}])
        
        df_parameters['session_id'] = session_id

        if not df_parameters.empty:
            if param_extract_method=='tail':
                return df_parameters.tail(1)
            else:
                return df_parameters.head(1)
    except FileNotFoundError:
        pass

def build_session_info(root_path, load_pycontrol=False, 
                       pycontrol_parameters=None,
                       param_extract_method='tail'):
    """
    This function takes a root path as input and creates a session info from the folders at that path.
    It parses the folder name with a regular expression to extract information about the animal id,
    experiment date/time, and session id. It returns a Pandas dataframe containing this information,
    as well as a calculated session number for each animal based on the experiment date/time.

    Args:
    - root_path (Path): A string representing the root path where the session folders are located.
    - load_pycontrol (bool): whether to load the pycontrol data to extract variable parameters
    - param_extract_method (str): how to extract the parameters from each pycontrol file, can be tail or head
    
    Returns:
    - df_session_info: A Pandas dataframe containing the session information.
    """
    paths = Path(root_path).glob('*/*')
    paths = [Path(p) for p in paths]

    # parse the folder name
    # TODO: extract the task name
    def parse_session_path(session_path):
        session_id = session_path.name
        task_name = session_path.parent.name

        pattern = r'(\w+)-(.*)'
        m = re.search(pattern, session_id)
        
        
        if m:
            animal_id = m.group(1)
            date_string = m.group(2)
            expt_datetime = datetime.strptime(date_string, "%Y-%m-%d-%H%M%S")

            return {'animal_id':animal_id, 
                    'expt_datetime': expt_datetime,
                    'session_id':session_id,
                    'task_name':task_name,
                    'path':session_path}

    session_info = [parse_session_path(p) for p in paths]
    df_session_info = pd.DataFrame(session_info)
    

    
    # Calculate the session number
    df_session_info['session_no'] = df_session_info.groupby(['animal_id','task_name'])['expt_datetime'].rank()

    
    if load_pycontrol:
        if pycontrol_parameters is None:
            raise ValueError('You need to set the parameters to extract')
        
        df_parameters = pd.concat([load_pycontrol_variables(p,pycontrol_parameters,param_extract_method) for p in paths])

        
        #merge the paramteres to df_session_info
        df_session_info = df_session_info.merge(df_parameters, on='session_id')
    
    return df_session_info

def load_datasets(session_paths, load_behaviour_dataset=False):
    ds_list=[]
    for p in tqdm(session_paths):
        fn = p/'processed'/'xr_session.nc'
        try:
            ds = xr.open_dataset(fn) 
            # print(np.unique(np.diff(ds.event_time.data)), ds.event_time.data[-1], p)
            ds = ds.drop_dims('time') # for performance reason
            
            
            if load_behaviour_dataset:
                fn_be = p/'processed'/'xr_behaviour.nc'
                if fn_be.exists():
                    ds_be = xr.open_dataset(fn_be)
                    ds = xr.merge([ds,ds_be])
            
            ds_list.append(ds)
            ds.close()
        except FileNotFoundError:
            print(f'{fn} not found! skipping')

    return ds_list


def load_and_concat_dataset(session_paths, load_behaviour_dataset=False):
    ds_list = load_datasets(session_paths, load_behaviour_dataset)
    print('Concating datasets...')
    return xr.combine_nested(ds_list,'session_id')

def filter_sessions(df_session_info, animal_id=None, 
                    session_no=None, session_method='exact', task_name=None,
                    query=None):
    """
    Filter a dataframe of session information based on various criteria
    
    Parameters:
    df_session_info (pd.DataFrame): The dataframe to filter
    animal_id (str or list of str, optional): Filter by animal ID(s)
    session_no (int or list of int, optional): Filter by session number(s)
    session_method (str, optional): How to interpret session_no. 
        Must be one of 'exact', 'tail', 'head', or 'between'.
        'exact' will choose the exact session_no specified,
        'tail' will choose the last n of sessions
        'head' will choose the first n of sessions
        'between' will choose the session between two specified number
    task_name (str or list of str, optional): Filter by task name(s)
    query (str): additional query for the pandas query function
    Returns:
    pd.DataFrame: A filtered version of the input dataframe
    """

    
    df = df_session_info.copy()
    
    if animal_id is not None:
        if type(animal_id) is str:
            df = df[df.animal_id.str.contains(animal_id)]
        else:
            mask = np.zeros((len(df),)).astype(bool)
            for id in animal_id:
                mask = mask | df.animal_id.str.contains(id)
            df = df[mask]

        
    if session_no is not None:
        if session_method == 'exact':
            if type(session_no) is not list:
                session_no = [session_no]
            df = df[df.session_no.isin(session_no)]
        elif session_method =='tail':
            assert type(session_no) is int, 'if session_method is tail, you have to supply an int to session_no'
            df = df.sort_values('session_no').groupby(['animal_id','task_name']).tail(session_no)
        elif session_method =='head':
            df = df.sort_values('session_no').groupby(['animal_id','task_name']).head(session_no)
        elif session_method =='between':
            df = df[df.session_no.between(session_no[0], session_no[1])]
        else:
            raise NotImplementedError(f'session_method {session_method} is not supported')
    
    if task_name is not None:
        if type(task_name) is not list:
            task_name = [task_name]
        df = df[df.task_name.isin(task_name)]
        
    if query is not None:
        df = df.query(query)
        
    return df

def get_session_path(row, root_path):
    return Path(os.path.join(root_path, row.task_name, row.session_id))