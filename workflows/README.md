### Configuration
System-specific configurations are in `C:\ProgramData\snakemake\snakemake` in Windows, or in `$HOME/.config/snakemake` under Linux

## Execute
Execute the following command to run the workflow. It is strongely recommend to dry-run (simulate without actually executing anything) the first time you use a workflow with the `-n` command
`snakemake --cores -n --snakefile workflows/spout_bar_nov22.smk`
To actually execute the workflow, remove the `-n` option.
`snakemake --cores --snakefile workflows/spout_bar_nov22.smk`
