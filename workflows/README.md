# Snakemake workflow
This folder contains the scripts and workflows of analyzing pycontrol data with snakemake. It assume a session-based folder structure.

The workflows are developed using [snakemake](https://snakemake.github.io/). Snakemake provides several useful features for data analysis, including:
1. Only executing a pipeline when necessary by comparing the modification date of the input and output files. 
2. Automatically running multiple tasks in parallel by analyzing the depedency grapch of the workflow
3. Portability by integration with conda and container virtualization.
4. Ability to automatically generate unit test for workflow
5. Ability to generate detailed reports for each recoding sessions


## Configuration
- Specify the root of your session-based folder in `workflows\config\config.yaml` through the `session_base_dir` setting, e.g.
    ```
    session_root_dir: 'Z:/Teris/ASAP/expt_sessions/'
    ```

## Usage
 Execute the following command to run the workflow. It is strongely recommend to dry-run (simulate without actually executing anything) the first time you use a workflow with the `-n` command

`snakemake --cores -n --snakefile workflows/spout_bar_nov22.smk`

To actually execute the workflow, remove the `-n` option.

`snakemake --cores --snakefile workflows/spout_bar_nov22.smk`


Since the workflow is based on snakemake, you can also use any of the advanced option supported by snakemake. For detail please consult the snakemake [documentations](https://snakemake.readthedocs.io/en/stable/executing/cli.html)