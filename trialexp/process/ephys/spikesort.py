import spikeinterface.sorters as ss
import spikeinterface.extractors as se


def path_to_recording_openephys(
        experiment_nb: str, 
        recording_nb: int, 
        stream_name: str):
    pass


def sort(
        rec_path: str, 
        output_folder: str, 
        sorter_name: str = 'kilosort3',
        verbose: bool = True):
    
    
    sorting = ss.run_sorter(
        sorter_name = sorter_name,
        recording = rec_path, 
        output_folder = output_folder,
        # remove_existing_folder: bool = True, 
        # delete_output_folder: bool = False, 
        verbose = verbose)
        # raise_error: bool = True, 
        # docker_image: Optional[Union[bool, str]] = False, 
        # singularity_image: Optional[Union[bool, str]] = False, 
        # with_output: bool = True, 
        # **sorter_params)
