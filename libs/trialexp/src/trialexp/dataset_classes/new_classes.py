# Adapted from
# https://refactoring.guru/design-patterns/abstract-factory/python/example

from abc import abstractmethod
from dataclasses import ABC, dataclass
from datetime import datetime
from typing import Any

from importlib.metadata import metadata

###############################################
#                                             #
#          Proposal for Metadata classes      #
#                                             #
###############################################

@dataclass(frozen=True) # frozen True to emulate pseudo-immutability
class Metadata:
    '''
    Parent class for metadata of Dataset classes

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
    metadata_df : pd.DataFrame
    metadata_creation: datetime
    metadata_type: str = 'ND'

    def get_shape() -> tuple:
        ...
    def get_subject_IDs(unique:bool = False) -> tuple:
        ...
    def get_tasks(unique:bool = False) -> tuple:
        ...
    def get_groups() -> tuple:
        ...
    def get_conditions() -> tuple:
        ...

# class PhotometryMetadata(Metadata): 
#     def __post_init__(self):
#         photo_dict = {'getThisKey':'GetThatValue'}
#         #
#         # extract Metadata without data
#         #
#         self.__setitem__('metadata_dict', photo_dict)

# class DlcMetadata(Metadata):

#     def get_scorer():
#         ...

# class PyControlMetadata(Metadata):
    
#     ...

class TrialAxes:
    """
    fake class for axes objects resulting from plot functions
    """
    ...

class TrialFigure:
    """
    fake class to assemble layout of Axes
    """
    ...

###############################################
#                                             #
# Refactoring attempt with abstract factory   #
#                                             #
###############################################


class DatasetFactory(ABC):
    """
    Abstract Base Class for all trialexp datasets
    
    """
    data: Any
    metadata: Metadata
    source: str

    @abstractmethod
    def create_dataset(self, data, metadata, source) -> ContinuousDataset:
        pass

    @abstractmethod
    def create_event_dataset(self) -> EventDataset:
        pass

    @abstractmethod
    def join_dataset(self, continuous_ds_to_join) -> ContinuousDataset:
        ...


#Abstract datasets types
class ContinuousDatasetFactory(DatasetFactory):
    """
    Abstract Base Class for all continuous datasets
    """

    def create_dataset(self, data, metadata, source) -> ContinuousDataset:
        return PycontrolDataset()

    if source == 'deeplabcut':
        ...
        
    def join_dataset(self, continuous_ds_to_join) -> ContinuousDataset:
        ...

    def get_metadata(self) -> Metadata:
        ...

class EventDatasetFactory(DatasetFactory):
    """
    Abstract Base Class for all events datasets
    """
    
    data: pd.DataFrame
    metadata : Metadata

    ...

# Concrete datasets types
@dataclass
class PycontrolDataset(EventDataset):
    """
    subclass for pycontrol datasets
    """
    def plot_raster() -> TrialAxes:
        # specific raster function for events out of pycontrol
        ...

    def get_metadata() -> PhotometryMetadata:
        # fetch metadata here
        ...

@dataclass
class DlcDataset(ContinuousDataset):
    """
    subclass for DLC datasets
    """
    ...

@dataclass
class PhotometryDataset(ContinuousDataset):
    """
    subclass for photometry datasets
    """
    metadata: PhotometryMetadata
    
    def get_metadata() -> PhotometryMetadata:
        ...



# rest of the example


class ConcreteFactory1(DatasetFactory):
    """
    Concrete Factories produce a family of products that belong to a single
    variant. The factory guarantees that resulting products are compatible. Note
    that signatures of the Concrete Factory's methods return an abstract
    product, while inside the method a concrete product is instantiated.
    """

    def create_product_a(self) -> AbstractProductA:
        return ConcreteProductA1()

    def create_product_b(self) -> AbstractProductB:
        return ConcreteProductB1()




def dataset_creation(factory: DatasetFactory) -> None:
    """
    The client code works with factories and products only through abstract
    types: AbstractFactory and AbstractProduct. This lets you pass any factory
    or product subclass to the client code without breaking it.
    """
    dataset_photo = factory.create_continuous_dataset(data, metadata, source)
    dataset_pycontrol = factory.create_event_dataset(data, metadata, source)

    print(f"{product_b.useful_function_b()}")
    print(f"{product_b.another_useful_function_b(product_a)}", end="")


dataset_creation(ContinuousDatasetFactory())

dataset_creation(EventDatasetFactory())