# HE_ana
Repository with scripts and notebooks for high energy runs analysis.

### Scripts

It includes 2 scripts to prepare and process Sophronia data. They require having IC installed.
* `process_HE_runs.py`: it takes Sophronia files from a run and creates a summary of each event. 
* `cut_HE_runs.py`: it takes output files from `process_HE_runs.py` and performs the desired cuts on them.

**CAUTION**: the path to the files in both scripts has to be changed. Also, check the possible arguments for running them.

### Notebooks

It includes 1 notebook that analyzes the outputs from previous scripts (`HE_run_analysis.ipynb`). It is made to work on the `cut_HE_runs.py` output, because it does not perform any cut. It includes the computation of the parameters for the absolute energy scale of each run, the shifting of the spectra, and other corrections for dependencies of the energy with different variables.
