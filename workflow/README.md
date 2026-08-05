# Snakemake workflow

This folder contains the scripts and workflows for analyzing pycontrol data with snakemake. It assumes a session-based folder structure.

The workflows are developed using [**snakemake**](https://snakemake.github.io/). Snakemake provides several useful features for data analysis, including:
1. Only executing a pipeline when necessary by comparing the modification date of the input and output files. 
2. Automatically running multiple tasks in parallel by analyzing the dependency grapch of the workflow
3. Portability by integration with conda and container virtualization.
4. Ability to automatically generate unit tests for workflow
5. Ability to generate detailed reports for each recording sessions
6. Possibility to run the code in a computing cluster to greatly speed up the analysis


## Configuration

- Specify your `debug_folder` in `workflow\scripts\settings.py`

- Check `.env` for folder paths

## Folder structure
The snakemake file (`*.smk`) that defines the workflow is in the `workflow` folder, the scripts are in `workflows/scripts`, config files are in `workflows/config`.



## Advanced Usage

### Copying files to `by_session` folder
We first need to copy files to the `by_session` folder. You can do that by

```
python workflow/scripts/00_create_session_folders.py
```



### Executing snakemake
If no target is specified, snakemake will execute the first rule in the snakemake file by default. By default, it will search for a `Snakefile` under the `workflow` folder. The `Snakefile` is the master workflow file. The master Snakefile contains both the pycontrol and spike sorting workflow.

Execute the following command to run the workflow. It is strongly recommended to **dry-run** (simulate without actually executing anything) the first time you use a workflow with the `-n` command. The `--cores` command tells snakemake to use all available cores, while you can specify the exact number of cores to be used with the `-c` option, e.g. `-c 4`.

```
snakemake --cores -n
```



To actually execute the workflow, remove the `-n` option.

```
snakemake --cores
```



If you want to just run a subset of the workflow, e.g. only pycontrol or spike sorting, specify the workflow file directly:
Note: using all available cores may cause problems to ettin, it is suggested that we only use a small number (e.g. 10). 

```
snakemake -c10 --snakefile workflow/pycontrol.smk
```

### Executing profiles
To avoid specifying the number of cores or whether to skip on error every time you run snakemake, you can provide a profile file. Please consult the [official documentation](https://snakemake.readthedocs.io/en/stable/executing/cli.html#profiles) for details. A sample profile is provided in `workflow/profile/config.yaml`.

#### Task-specific execution

If you want to just analyze a particular task, add the task name to the following line in the `pycontrol.smk` file. If you do not specify any task, it will run the pipeline on all tasks.

```python
task2analyze(['reaching_go_spout_bar_nov22', 'reaching_go_spout_incr_break2_nov22','pavlovian_spontanous_reaching_march23'])
```

#### Spike-sorting

To do spike sorting, you can execute

```
snakemake --cores --snakefile workflow/spikesort.smk
```



#### Other options

Since the workflow is based on snakemake, you can also use any of the advanced options supported by snakemake. For detail please consult the snakemake [documentations](https://snakemake.readthedocs.io/en/stable/executing/cli.html)

#### keep-going option `-k`

Sometimes you will encounter errors in running the pipeline in some of the recordings. The default behaviour of snakemake is to stop when it encounters an error. You can ask snakemake to skip that problematic session and continue by adding the `-k` (keep going) option:

```
snakemake --cores --snakefile workflow/pycontrol.smk -k
```



On other occasions, you may encounter the error`IncompleteFilesException`. In such cases,  you can try the `--rerun-incomplete` flag:

```
snakemake --cores --snakefile workflow/pycontrol.smk -k --rerun-incomplete
```


## Development
The best way to start developing new script is by using the interactive Python session in VS Code. 
1. Open any scripts in the `workflow/scripts` folder
2. The scripts are marked with [cell mode](https://code.visualstudio.com/docs/python/jupyter-support-py) `#%%`. Run the current cell by <kbd>shift</kbd>+<kbd>enter</kbd>. This will open the Python interactive window in the same folder as the script file
3. Change the working directory of the Python session to the root folder of the project.

    ```
    import os;
    os.chdir('../..')
    ```


**Note:** If you want `getSnake` to switch to your project folder directly. Define a `SNAKEMAKE_DEBUG_ROOT` environment variable that points to the `trialexp` folder in your system. Alternatively, you can also create a `.env` file in the project root directory, defining that `SNAKEMAKE_DEBUG_ROOT` variable, e.g.


```
# .env
SNAKEMAKE_DEBUG_ROOT = C:\code\trialexp
```

4. Double check that you are in the root of the project by running `os.getcwd()`. You should be at the `trialexp` folder.
5. Execute and test your code by using the cell mode 

### Helper functions

By default, snakemake will inject a `snakemake` object during execution to give you the input and output file names. However, this object is not available when you execute a script outside of the snakemake environment (e.g. running a script file during development). A helper module [snakehelper](https://github.com/teristam/snakehelper/tree/master) is created to compile the snakemake workflow and return the input and output file names, so that they can be used for development and debugging purpose.

Executing workflow script in interactive mode is very similar to running normal Python scripts, except there is a special helper function `getSnake` that work as a "glue" between the snakemake environment and normal Python. It will compile the workflow according to some target files and specific rules, and return you the input and output file name in two dictionary objects. It is very useful when you are developing new scripts or debugging old ones on some specific data that generate an error. For detailed usage of the `getSnake` function, please consult the  [snakehelper](https://github.com/teristam/snakehelper/tree/master) repository.
