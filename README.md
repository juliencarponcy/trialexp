# Trialexp

This repository contains the analysis pipeline for pycontrol, pyphotometry and Neuropixels data.

## Installation
This project manages its dependencis via uv
1. Install `uv` following instruction [here](https://docs.astral.sh/uv/getting-started/installation/)
2. At the current project directory, run `uv sync`
   1. For gpu support and snakemake support do `uv sync --extra full`

## Configurations
The pipeline relies on a set of environmental variables for the directory location. Choose an example that is closest to your system under `env` folder, fill out the content and rename it to `.env`, and copy it to the project root e.g. as `trialexp\.env`.

## Usage
This pipline uses [just](https://github.com/casey/just) to handle receipes to run the pipeline
1. Run `just run-pipeline` to run the full pipeline
2. To force rerun a certain session use `just make-session <search term> -F`  
 e.g. `just make-session TT021-2025-06-04 -F`

## Adding new cohort
Files to modify:
1. `params/animal_info.csv`
2. `workflow/scripts/00_create_session_folders.py`
3. `workflow/pycontrol.smk`

Note: If you want to add suffix to animal ID, please make sure it doesn't have special characters, use
`TT001DLS` instead of `TT001_DLS` or `TT001-DLS`.

## Adding new task
Files to modify
1.  `params/tasks_params.csv`
2. the `compute_success` function in `trialexp/process/pycontrol/session_analysis.py`
3. `workflow/scripts/07_time_warping.py`
4. the `compute_trial_outcome` function in `trialexp/process/pycontrol/session_analysis.py`


## Inspecting raw photometry data
We have phased out the export to spike2 file because `sonpy`, the package needed for the export, depends on very old Python version. Now the photometry raw data are exported to a `photometry.parquet` file which can be read by [plotJuggler](https://github.com/facontidavide/PlotJuggler).
