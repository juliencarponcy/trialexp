# Snakemake workflow
This folder contains the scripts and workflows of analyzing pycontrol data with snakemake. It assume a session-based folder structure.

The workflows are developed using [snakemake](https://snakemake.github.io/). Snakemake provides several useful features for data analysis, including:
1. Only executing a pipeline when necessary by comparing the modification date of the input and output files. 
2. Automatically running multiple tasks in parallel by analyzing the depedency grapch of the workflow
3. Portability by integration with conda and container virtualization.
4. Ability to automatically generate unit test for workflow
5. Ability to generate detailed reports for each recoding sessions
6. Possibly to run the code in a computing cluster to greatly speed up the analysis


## Configuration
- Specify the root of your session-based folder in `workflows\config\config.yaml` through the `session_base_dir` setting, e.g.
    ```
    session_root_dir: 'Z:/Teris/ASAP/expt_sessions/'
    ```

## Usage

If not target is specified, snakemake will execute the first rule in the snakemake file by default.

Execute the following command to run the workflow. It is strongely recommend to dry-run (simulate without actually executing anything) the first time you use a workflow with the `-n` command. The `--cores` command tells snakemake to use all avaiable cores, you can specify the exact number of core with the `-c` option, e.g. `-c 4`.

`snakemake --cores -n --snakefile workflows/spout_bar_nov22.smk`

To actually execute the workflow, remove the `-n` option.

`snakemake --cores --snakefile workflows/spout_bar_nov22.smk`

It is possible that some session folder does not hvae the photometry data, in this case the analysis will fail. You can ask the workflow to continue with other session folder in case of failure with the `-k` option. By default, it will wait for 5s for the missing file. You can make it shorter by using the `--latency-wait` option

`snakemake --cores --snakefile workflows/spout_bar_nov22.smk -k --latency-wait 1`


Since the workflow is based on snakemake, you can also use any of the advanced option supported by snakemake. For detail please consult the snakemake [documentations](https://snakemake.readthedocs.io/en/stable/executing/cli.html)

