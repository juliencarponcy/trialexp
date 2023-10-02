## Sorting workflow

### Environmental variables

You will need to specify the following environmental variables according to the location of various paths in your system

```
CODE_ROOT_FOLDER = <path to parent folder of your repositories>
RAW_DATA_ROOT_DIR = <path to the head-fixed folder on the server>
TEMP_DATA_PATH = <path to the temp folder>
```

You also need to make sure your MATLAB environment is setup correctly. 

### Kilosort
Some recording will generate a `The CUDA error was: invalid configuration argument` error. Follow the github issue to apply a temporary [fix](https://github.com/MouseLand/Kilosort/issues/383)

### Output files
- Reponse curves for cells are stored in `processed/figures/ephys/response_curves`
- `processed/xr_spikes_fr.nc` is an xarray Dataset containing the instantaneous firing rate of all cells
- `processed/xr_spikes_trials.nc` is an array Dataset containing the event-triggered firing rates