# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed
- Cosmetics: do not plot the 0 point in time warp (2025-11-20)
- Optimized print statements (2025-11-19)
- Updated logging and improved trial end label in timewarp specification (2025-11-19)
- Use trial end point for time warping (2025-11-19)
- Updated package dependencies (2025-11-19)

### Added
- Added photometry parquet input to photometry pipeline (2025-11-18)
- Better handling for cases with no photometry present (2025-11-18)

### Fixed
- Fixed edge case when there is number in pycontrol dataframe content (2025-11-18)
- Fixed edge case where there are lots of overlapping trials (2025-11-18)
- Fixed scatter plot rendering by adding checks for empty dataframes (2025-11-18)
- Fixed extraction spec to be consistent with task definition (2025-11-17)
- Fixed end event extraction logic to handle cases with no prior break_after events (2025-11-14)
- Fixed trial data extraction by using the last ITI state (2025-11-14)
- Fixed end detection problem (2025-11-14)
- Fixed deprecation error (2025-11-13)
- Fixed problem caused by duplicate events (2025-11-07)
- Fixed typo in animal ID suffix note (2025-11-06)

## [2025.11.0] - 2025-11-14

### Added
- **Spike Sorting Timewarp Support** (Merge: spike_sort_timewarp)
  - Added timewarp functionality to spike sorting pipeline
  - Added support for both standard and neuropixels data
  - Added attributes for better data tracking
  - Enhanced linear_modelling.py with 265 new lines of functionality

- **Extended Time Warping** (Merge: warp_extend)
  - Extended time window after reward in time warp analysis

### Changed
- Optimized config for break2 task
- Refactored time warp data extraction and improved logging verbosity
- Added 'end' event handling and adjusted trial data extraction logic
- Changed spontaneous task to look for spout instead
- Improved import of old file format
- Updated snakehelper dependency source
- Disabled verbose logging in time warping script

### Fixed
- Better handling of edge cases where last event is too close to trial end

## [2025.11.0] - 2025-11-06

### Added
- **November Cohort Support** (PR #6: november_new_mice)
  - Added animal IDs for DMS and DLS analysis
  - Added documentation for acceptable animal ID formats in README
  - Added upcoming mice to configuration
  - Switched to use session name as subject ID for consistency

### Changed
- Updated logging to support individual session folders for better error handling
- Suppressed jobslib verbose output to avoid polluting logs

## [2025.10.0] - 2025-10-20

### Added
- **September 2025 Cohort** (Merge: sep_2025_cohort)
  - Added new cohort to animal configuration

### Changed
- Updated workflow configuration for new cohort

### Fixed
- Fixed path errors
- Bugfix: always re-run incomplete sessions
- Removed spurious animal

## [2025.10.0] - 2025-10-17

### Added
- **October Cohort Support** (PR #5: october_cohort)
  - Added October cohort animals to configuration
  - Updated workflow configuration

- **Atlas Aligner Enhancements** (Merge: atlas_aligner)
  - Added new firing rate analysis code
  - Enhanced ephys utility functions with 114 new lines of functionality
  - Improved anatomy processing utilities

## [2025.01.0] - 2025-01-18

### Added
- **Atlas Aligner Tool Development** (atlas_aligner branch - ongoing work in January 2025)
  - Added comprehensive documentation for common operations
  - Fixed build process and FastAPI root_path issues
  - Configured nginx to serve React frontend directly
  - Backend is fully working
  - Refactored to structure compatible with UV package manager
  - Folder reorganization into apps and libs structure
  - Added Docker compose for running the application
  - Added Dockerfiles for frontend and backend
  - Transition to UV package manager for atlas aligner components
  - Added function to save data
  - Added notification system
  - Added track title display
  - Added cell count display
  - Added probe matching functionality
  - Converted trajectory to probe coordinates to show tip location
  - Added firing rate plots with unified depth representation
  - Improved stack bar plots with proper starting points
  - Added shift control and probe length configuration

### Changed
- Simplified session search during debugging

### Fixed
- Fixed env file name bug

---

# 2024 Changes

## [2024.12.0] - 2024-12-18

### Added
- **Atlas Aligner Web Interface**
  - Added Plotly plots for trajectory location visualization
  - Added code to map trajectory back to the atlas
  - Added session search boxes
  - Added API code with Mantine UI framework
  - Added frontend starting code
  - Show text on bar displays

### Changed
- Refactored functions into separate modules
- Show sorted results only
- Reordered file organization

### Fixed
- Better error handling for NaN data
- Better handling for cases when no event is found
- Added check to avoid saving empty files

## [2024.12.0] - 2024-12-12

### Added
- **Signal Analysis Improvements**
  - Added spectrogram plotting to check for noise
  - Added spectrogram to Snakemake file
  - Added more examples for session filtering
  - Added resampling to speed up plots
  - Disabled check for sessions contaminated by noise near the end

### Changed
- Refactored photom_data handling to remove unnecessary squeezing
- Moved unused scripts
- Disabled manual code

## [2024.11.0] - 2024-11-21

### Added
- **Signal Viewer Tool**
  - Added basic structure for signal viewer
  - Added sidebar template for signal viewer
  - Added support for bleedthrough channel in export
  - Added support for bleedthrough channel data
  - Added photometry info display
  - Added display for pycontrol info

### Changed
- Refactored move_folders function to improve error handling and streamline folder creation
- Added trial type raw count and refactored for cleaner code

## [2024.11.0] - 2024-11-01

### Added
- **New Cohort and Ephys Support**
  - Added support for new cohort with animal specification in pipeline
  - Added support for showing ephys results
  - Added timewarping to unit responses (WIP)
  - Plot the results of unit firing rate time warping

### Changed
- Adjusted plots for better visualization
- Made figures smaller to improve performance
- Made deshadowing process less stringent to accommodate large spike during random reward

### Fixed
- Fixed ylim for lick plot to avoid confusion
- Fixed rate calculation to make it lick/sec

## [2024.10.0] - 2024-10-10

### Changed
- **Time Warp Improvements** (Merge: time_warp_nogap)
  - Removed gaps in reward timing
  - Added random water triggered signal
  - Changed licking rate to running average of 200ms
  - Simplified time warping code (removed 83 lines, added 81)
  - Updated timewarp_spec.json with new configuration

### Added
- Added new cohort animals
- Added new task

### Fixed
- More robust error handling in pycontrol files
- Bugfixes for bad sessions and plotting issues

## [2024.08.0] - 2024-08-22

### Added
- Support for new cohort

## [2024.07.0] - 2024-07-12

### Added
- **Lick Analysis** (lick_detector branch integration)
  - Added lick probability to time warp plots
  - Added functions to analyze lick in analog signal

### Changed
- Relaxed lick error checking to accommodate neuropixel recordings
- Better handling of sync location for neuropixel recordings
- Improved time warp result plotting

## [2024.07.0] - 2024-07-08

### Added
- **Figure Viewer Improvements**
  - Added figure viewer code
  - Base64 image encoding support
  - More flexible plotting functions
  - Improved figure viewer functions
  - Better error checking when importing files

### Changed
- Refactored folder reorganization code
- Performance improvements

## [2024.06.0] - 2024-06-21

### Added
- **Environment and Configuration**
  - Added support for Lapis environment
  - Added Lapis config
  - Added proper sourcing
  - Added ticks to plots
  - Workflow improvements

### Changed
- Migrated to NumPy 2.0
- Updated rerun triggers
- Increased tolerance of event alignment

## [2024.06.0] - 2024-06-16

### Added
- **Kilosort4 and Statistical Analysis**
  - Merged Kilosort4 branch
  - Added mixed effect function
  - Added ability to do interaction analysis
  - Aligned reward to the opening of the valve

### Changed
- Updated requirements.txt

### Fixed
- Used correct file format for data import
- Allowed pipeline to work with earlier cohort data
- Used zigzag mark to indicate compressed axis

## [2024.05.0] - 2024-05-31

### Added
- **Data Import Improvements**
  - Support for importing new pyphotometry format

### Fixed
- Spike2 export bug caused by importing non-analog data
- Time warping bug that occurs when 1s is not divisible by sampling rate

## [2024.05.0] - 2024-05-28

### Added
- **OpenField Analysis**
  - Added scripts for analyzing openfield data
  - Added support for rDA cohort

## [2024.04.0] - 2024-04-30

### Added
- **PyTorch Integration**
  - Added PyTorch model

### Changed
- Added functions to align response curve to centroid
- Updated timewarp specifications

## [2024.03.0] - 2024-03-31

### Added
- **Time Warping Pipeline** (Major feature)
  - Added time warping to pipeline
  - Added functions for time warping and linear regression analysis
  - Added break2 timewarp spec
  - Added function to draw event line in time warping
  - Added doc to functions
  - Support for timewarping in new pycontrol format and task

### Changed
- Fine tuned parameters for timewarping
- Only plot event line when found in trial
- Integrated interpolation results back to xarray
- Changed extraction timing, added support for old task

## [2024.02.0] - 2024-02-29

### Added
- **Trial Analysis Improvements**
  - Added trial parameters to analysis pipeline
  - Added plot of reaching time
  - Improved progress bar
  - Added function for comparing kernel vs mean

### Fixed
- Fixed final padding bug

## [2024.01.0] - 2024-01-31

### Added
- **New Cohort Structure Support**
  - Support for new cohort structure
  - Updated create_session_folders to support new cohort folder structure
  - Added session info function for new cohort structure
  - Added animal info for new cohort
  - Support for lick event in analysis

### Changed
- Updated to use new cohort folder structure
- Do not coarsen if sampling rate is lower than 100Hz
- Removed dependency on space in separator of event names

### Fixed
- Folder path bugfixes
