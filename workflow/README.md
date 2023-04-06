# Snakemake workflow
This folder contains the scripts and workflows of analyzing pycontrol data with snakemake. It assume a session-based folder structure.

The workflows are developed using [snakemake](https://snakemake.github.io/). Snakemake provides several useful features for data analysis, including:
1. Only executing a pipeline when necessary by comparing the modification date of the input and output files. 
2. Automatically running multiple tasks in parallel by analyzing the depedency grapch of the workflow
3. Portability by integration with conda and container virtualization.
4. Ability to automatically generate unit test for workflow
5. Ability to generate detailed reports for each recording sessions
6. Possibility to run the code in a computing cluster to greatly speed up the analysis


## Depedencies
The snakemake workflow may contain additional dependency requirement. You will need to update your virutal environment with 

` conda env update -f trialexp.yaml`

## Configuration
Specify the root of your session-based folder and project folder in the `.env` file
    ```
    SNAKEMAKE_DEBUG_ROOT=<your project root folder here>
    SESSION_ROOT_FOLDER=<the path of the by_session folder>
    ```

## Folder structure
The snakemake file (*.smk) that define the workflow is in the `workflow` folder, the scripts are in `workflows/scripts`, config files are in `workflows/config`.

## Usage

Execute the following command to run the workflow. It is strongely recommend to dry-run (simulate without actually executing anything) the first time you use a workflow with the `-n` command. The `--cores` command tells snakemake to use all avaiable cores, you can specify the exact number of core with the `-c` option, e.g. `-c 4`.

`snakemake --cores -n`

To actually execute the workflow, remove the `-n` option.

`snakemake --cores`

It is possible that some session folder does not have the photometry data, in this case the analysis will fail. You can ask the workflow to continue with other session folder in case of failure with the `-k` option. By default, it will wait for 5s for the missing file. You can make it shorter by using the `--latency-wait` option

`snakemake --cores -k --latency-wait 1`


Note: If no snakefile is specified, snakemake will try to search for a `Snakefile` in the current folder, or a folder under `workflow`. If no target is specified, snakemake will execute the first rule in the snakemake file by default.



Since the workflow is based on snakemake, you can also use any of the advanced option supported by snakemake. For detail please consult the snakemake [documentations](https://snakemake.readthedocs.io/en/stable/executing/cli.html)

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
By default, snakemake will inject a `snakemake` object during execution to give you the input and output file names. However, this object is not available when you execute script outside of the snakemake environment (e.g. running a script file during development). A helper module [snakehelper](https://github.com/teristam/snakehelper/tree/master) is created to compile the snakemake workflow and return the input and output file names, so that they can be used for development and debugging purpose.

Executing workflow script in interactive mode is very similar to running normal Python scripts, except there is a special helper function `getSnake` that work as a "glue" between the snakemake environment and normal Python. It will compile the workflow according to some target files and specific rules, and return you the input and output file name in two dictionary objects. It is very useful when you are developing new scripts or debugging old ones on some specific data that generate a error. For detailed usage of the `getSnake` function, please consult the  [snakehelper](https://github.com/teristam/snakehelper/tree/master) repository.
