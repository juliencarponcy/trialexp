#!/bin/bash

# Activate the conda environment
conda activate trialexp

# Run the Python file
python workflow/scripts/00_create_session_folders.py

snakemake --snakefile workflow/pycontrol.smk
