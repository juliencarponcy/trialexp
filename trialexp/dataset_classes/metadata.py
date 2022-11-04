from dataclasses import dataclass
from datetime import datetime

from pandas import DataFrame

@dataclass(frozen=True) # frozen True to emulate pseudo-immutability
class Metadata:
    '''
    Parent class for metadata of Dataset classes
.
       Attributes
    ----------
    metadata_dict : dict
        a dictionary containing pairs of key:values of metadata
        valid for the entire dataset

    metadata_df : pd.DataFrame
        Rows for trials, holding colums:
            trial_nb : int
            trigger : str
            success : bool
            valid : bool
            condition_ID : int 
            condition : str
            group_ID : int
            session_nb : int
            subject_ID : int or str
            keep : bool
            trial_ID : int

    metadata_type : str
        The type name of the underlying data


    Methods
    -------
    get_shape()
    get_subject_IDs()
        get the subjects included 
    get_groups()
        The values of 'keep' column of metadata_df are all set to True
    get_conditions
    etc. to be implemented...

    '''
    metadata_dict: dict
    metadata_df : DataFrame
    metadata_creation: datetime
    metadata_type: str = 'ND'
    def verify(self, data) -> bool:
        ...
    
    def get_shape(self) -> tuple:
        ...
    def get_subject_IDs(self, unique:bool = False) -> tuple:
        ...
    def get_tasks(self, unique:bool = False) -> tuple:
        ...
    def get_groups(self) -> tuple:
        ...
    def get_conditions(self) -> tuple:
        ...

    class 