import shutil
from os.path import join, isfile 
from os import listdir, walk

import pandas as pd

from trialexp.utils.pycontrol_utilities import get_datetime_from_datestr, get_datestr_from_filename
#----------------------------------------------------------------------------------
# Data reorganization
#----------------------------------------------------------------------------------

def copy_files_to_horizontal_folders(root_folders, horizontal_folder_pycontrol, horizontal_folder_photometry):
    '''
    Browse sub-folders (in a single root folder or within a list of root folder)
    and copy them in a separate horizontal folders (no subfolders). The main
    purpose is for easier match between pycontrol and photometry files
 
    '''
    
    if isinstance(root_folders, str):
        root_folders = [root_folders]

    for root in root_folders:
        for path, subdirs, files in walk(root):
            for name in files:

                if name[-4:] == '.txt':
                    if not isfile(join(horizontal_folder_pycontrol,name)):
                        print(join(path, name))
                        shutil.copyfile(join(path, name),join(horizontal_folder_pycontrol, name))
                elif name[-4:] == '.ppd':
                    if not isfile(join(horizontal_folder_photometry, name)):
                        print(join(path, name))
                        shutil.copyfile(join(path, name),join(horizontal_folder_photometry, name))


#----------------------------------------------------------------------------------
# Helpers
#----------------------------------------------------------------------------------

def match_sessions_to_files(experiment, files_dir, ext='mp4', verbose=False) -> str:
    '''
    Refactoring note: need to be implemented as a standalone method for single
    py control files but would probably be much slower as it will have to 
    constantly browse folders of other data modalities to list all the files. 
    Can be useful as is for data reorganization purposes.
    
    Take an experiment instance and look for files within a directory
    taken the same day as the session and containing the subject_ID,
    store the filename(s) with the shortest timedelta compared to the
    start of the session in exp.sessions[x].files["ext"] as a list
    
            Parameters:
                    file_name (str): name of the file to look for
                    files_dir (str): path of the directory to look into to find a match
                    ext (str): extension used to filter files within a folder
                        do not include the dot. e.g.: "mp4"

            Returns:
                    str (store list in sessions[x].file["ext"])
    ''' 

    # subject_IDs = [session.subject_ID for session in self.sessions]
    # datetimes = [session.datetime for session in self.sessions]
    files_list = [f for f in listdir(files_dir) if isfile(
        join(files_dir, f)) and ext in f]

    if len(files_list) == 0:
        raise Exception(f'No files with the .{ext} extension where found in the following folder: {files_dir}')

    files_df = pd.DataFrame(columns=['filename','datetime'])

    files_df['filename'] = pd.DataFrame(files_list)
    files_df['datetime'] = files_df['filename'].apply(lambda x: get_datetime_from_datestr(get_datestr_from_filename(x)))
    # print(files_df['datetime'])
    for s_idx, session in enumerate(experiment.sessions):
        match_df = find_matching_files(session.subject_ID, session.datetime, files_df, ext)
        if verbose:
            print(session.subject_ID, session.datetime, match_df['filename'].values)
        
        if not hasattr(experiment.sessions[s_idx], 'files'):
            experiment.sessions[s_idx].files = dict()
        
        experiment.sessions[s_idx].files[ext] = [join(files_dir, filepath) for filepath in match_df['filename'].to_list()]

    return experiment

def find_matching_files(subject_ID, datetime_to_match, files_df, ext):
    """
    Helper function for match_sessions_to_files, find files with
    the same subject_ID in the filename, and taken the same day
    as the pycontrol session, return the file(s) with the shortest
    timedelta compared to the start of the session.
    
    
            Parameters:
                    subject_ID (int): from session.subject_ID (need to be converted
                        from string to int if int_subject_IDs=False at Session object creation)
                    datetime_to_match (datetime): from session.datetime
                    files_df (pd.Dataframe): Created from a list of files in 
                        match_sessions_to_files function
                    ext (str): extension used to filter files within a folder
                        do not include the dot. e.g.: "mp4"

            Returns:
                    match_df (pd.Dataframe): containing filenames of matching files
    """ 

    if ext not in ['nwb','h5']:
        # for videos, avoid integrating DeepLabCut labelled videos "['filename'].str.contains('DLC')"
        #TODO match_df is not a view or copy
        match_df = files_df.loc[(files_df['datetime'].apply(lambda x: pd.Timestamp.date(x)) == datetime_to_match.date()) &
            (files_df['filename'].str.contains(str(subject_ID))) &
            ~(files_df['filename'].str.contains('DLC'))].copy() 

        # will not avoid DLC-containing filenames in case of searching DLC .nwb data files
    else:
        match_df = files_df.loc[(files_df['datetime'].apply(lambda x: pd.Timestamp.date(x)) == datetime_to_match.date()) &
                                (files_df['filename'].str.contains(str(subject_ID)))].copy() #TODO match_df is not a view or copy

    # match_df = match_df.to_frame(name='matching_filename')
    if ~match_df.empty:
      
        # Compute time difference between the files
        match_df['timedelta'] = match_df['datetime'].apply(
            lambda x: abs(datetime_to_match-x))

        # Take the file with the minimum time difference
        match_df = match_df[match_df['timedelta'] == match_df['timedelta'].min()]
        #print(match_df['timedelta'])
        match_df['timedelta'] = match_df['timedelta'].apply(lambda x: x.seconds)
    
    return match_df