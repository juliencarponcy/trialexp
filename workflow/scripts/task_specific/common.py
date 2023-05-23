'''
Use to coordinate task specific scripts. it is not intended to run standalone
'''
from pathlib import Path
import os

task = snakemake.wildcards.task


# Search for the task-specific script and execute it if available
task_specific_file = f'workflow/scripts/task_specific/{task}.py'
if Path(task_specific_file).exists():
    with open(task_specific_file, 'r') as f:
        exec(f.read()) # will run in the current scope, which should contains the snakemake object