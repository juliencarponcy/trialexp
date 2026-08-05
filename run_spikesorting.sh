#!/bin/bash

# Activate the conda environment
source /home/MRC.OX.AC.UK/ndcn1330/miniconda3/etc/profile.d/conda.sh
conda activate trialexp

# Run the Python file
snakemake --snakefile workflow/spikesort.smk -k -c5 --rerun-triggers mtime
