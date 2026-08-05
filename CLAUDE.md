# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Trialexp is an analysis pipeline for processing pycontrol (behavioral task control), pyphotometry (fiber photometry), and Neuropixels (electrophysiology) data. The pipeline is orchestrated using Snakemake workflows and uses `just` for common task execution.

## Development Setup

### Installation
1. Install `uv`: https://docs.astral.sh/uv/getting-started/installation/
2. Sync dependencies: `uv sync`
3. For GPU and Snakemake support: `uv sync --extra full`

### Configuration
Create a `.env` file from one of the examples in the `env/` folder. Required environment variables include:
- `SESSION_ROOT_DIR`: Root directory for session data
- `ETTIN_MOUNT_PATH`, `ETTIN_DATA_FOLDER`: Server data paths
- Sorting-specific: `SORTING_ROOT_DATA_PATH`, `CODE_ROOT_FOLDER`, `RAW_DATA_ROOT_DIR`, `TEMP_DATA_PATH`
- Optional: `VIDEO_DIR`, `DLC_CONFIG_PATH` for DeepLabCut

## Common Commands

### Running Pipelines
```bash
# Run full pycontrol pipeline
just run-pipeline

# Run pipeline for specific session (interactive confirmation)
just make-session <search_term> [FLAGS]
# Example: just make-session TT021-2025-06-04 -F

# Run spike sorting for specific session
just sort <search_term> [FLAGS]

# Find sessions or data files
just find-session <pattern>
just find-data <pattern>

# Copy raw data into session folders (prerequisite)
just copy-data
```

### Development with Snakemake
- Direct Snakemake execution: `uv run snakemake --snakefile workflow/pycontrol.smk [OPTIONS]`
- Common flags: `-k` (keep going), `-c20` (20 cores), `-F` (force rerun), `-q rules` (quiet), `--rerun-incomplete`

### Atlas Aligner App
```bash
just run-aligner
```

## Architecture

### Directory Structure
```
src/trialexp/           # Main Python package
  process/              # Data processing modules
    pycontrol/          # Behavioral data processing
    pyphotometry/       # Photometry data processing
    ephys/              # Electrophysiology processing
    deeplabcut/         # Video analysis
    model/              # Modeling utilities
    figures/            # Plotting utilities
  config.py             # Environment variable loading

workflow/               # Snakemake workflows
  pycontrol.smk         # Main behavioral+photometry pipeline
  spikesort.smk         # Spike sorting pipeline
  deeplabcut.smk        # Video analysis pipeline
  scripts/              # Pipeline step scripts
    spike_sorting/      # Spike sorting workflow steps

params/                 # Configuration files
  animal_info.csv       # Animal metadata
  tasks_params.csv      # Task definitions and parameters
  timewarp_spec.json    # Time warping specifications

apps/                   # Standalone applications
  atlasaligner/         # Brain atlas alignment tool
  signal_viewer/        # Signal visualization tools
```

### Data Flow

#### Pycontrol/Photometry Pipeline (pycontrol.smk)
1. **Session folder creation** (`00_create_session_folders.py`): Organizes raw data
2. **Pycontrol processing** (`01_process_pycontrol.py`):
   - Parses behavioral task files (.txt/.tsv)
   - Extracts trial structure using `extract_trial_by_trigger()`
   - Computes trial conditions and outcomes via `compute_success()` and `compute_trial_outcome()`
   - Handles lick detection from analog signals if needed
3. **Plotting** (`02_plot_pycontrol_data.py`): Event histograms, reach histograms, discrimination scores
4. **Photometry import** (`04_import_pyphotometry.py`):
   - Loads pyphotometry .ppd files
   - Aligns photometry to behavioral events
   - Converts to xarray format
   - Computes z-scored dF/F signals
5. **Photometry plotting** (`05_plot_pyphotometry.py`): Event-triggered averages
6. **Time warping** (`07_time_warping.py`): Aligns trials to common temporal structure

#### Spike Sorting Pipeline (spikesort.smk)
1. **Sorting** (`s01_sort_ks4.py`): Kilosort4-based spike detection
2. **Waveform extraction**: Quality metrics computation
3. **Cell classification**: Cell type identification
4. **Time warping support**: Aligns spikes to behavioral events

### Key Processing Concepts

#### Trial Structure
- **Triggers**: Events that mark trial start (defined in `params/tasks_params.csv`)
- **Trial windows**: Time range around trigger for including events
- **Conditions**: Boolean flags for trial properties (e.g., success, stimulus type)
- **Events**: Timestamped occurrences (licks, reaches, rewards)

#### Session Analysis Flow
The `session_analysis.py` module provides core functions:
- `extract_trial_by_trigger()`: Segments continuous data into trials
- `add_trial_nb()`: Assigns trial numbers to events
- `compute_conditions_by_trial()`: Evaluates condition states per trial
- `compute_success()`: Task-specific success determination (MODIFY WHEN ADDING NEW TASK)
- `compute_trial_outcome()`: Categorizes trial results (MODIFY WHEN ADDING NEW TASK)

#### Data Formats
- **Behavioral**: Pandas DataFrames with `.attrs` for metadata
- **Photometry**: xarray Datasets with time-aligned signals
- **Spikes**: xarray Datasets with firing rates and event-triggered responses
- **Export**: Parquet files for PlotJuggler visualization (replaces Spike2)

### Snakehelper Integration
Scripts use `snakehelper.SnakeIOHelper.getSnake()` to handle Snakemake I/O, enabling both:
- Pipeline execution via Snakemake
- Interactive development in notebooks (using `config.debug_folder`)

## Adding New Features

### Adding New Cohort
1. Update `params/animal_info.csv` with animal metadata
2. Modify `workflow/scripts/00_create_session_folders.py` for folder organization
3. Update `workflow/pycontrol.smk` `session2analyze()` function with new cohort name

**Important**: Animal ID suffixes must not contain special characters (use `TT001DLS`, not `TT001_DLS` or `TT001-DLS`)

### Adding New Task
1. Add task parameters to `params/tasks_params.csv` (triggers, conditions, events)
2. Modify `compute_success()` in `trialexp/process/pycontrol/session_analysis.py`
3. Update time warping specification in `workflow/scripts/07_time_warping.py`
4. Modify `compute_trial_outcome()` in `trialexp/process/pycontrol/session_analysis.py`

## Spike Sorting Notes

### Environment Requirements
- MATLAB environment must be properly configured
- For Kilosort CUDA errors ("invalid configuration argument"), apply the temporary fix from: https://github.com/MouseLand/Kilosort/issues/383

### Output Files
- Response curves: `processed/figures/ephys/response_curves/`
- Instantaneous firing rates: `processed/xr_spikes_fr.nc` (xarray Dataset)
- Event-triggered firing rates: `processed/xr_spikes_trials.nc` (xarray Dataset)

## Development Notes

### Debugging
- Set `debug_folder` in `src/trialexp/config.py` to test individual scripts
- Most workflow scripts can run standalone using the SnakeIOHelper
- Use `uv run python workflow/scripts/<script>.py` for interactive debugging

### Photometry Visualization
Raw photometry data is exported to `photometry.parquet` for visualization with PlotJuggler (https://github.com/facontidavide/PlotJuggler), replacing the legacy Spike2 export.

### Session ID Format
Sessions follow the pattern: `<AnimalID>-<YYYY>-<MM>-<DD>-<HHMMSS>`
Example: `TT021-2025-06-04-130122`
