# Stability analysis of framework mutations in nanobodies

This repository contains source code for performing MD simulations and corresponding analyses on the stability implications of framework mutations in nanobodies (Nbs). Particular focus is on the L69F mutation on a GFP-binding nanobody Lag21. 

#### Dependencies
To run any code in this repository, please install the conda environment provided in `environment.yml` by running:
```
conda env create -f environment.yml
```

This will create a conda environment called `nbenv`. The only other dependency is pyRosetta, which can be obtained from [here](https://www.pyrosetta.org/downloads).

#### Python scripts
- `cdr_sampler.py` performs MCMC sampling of alternate CDR3 conformations using pyRosetta. This generates a pool of initial Nb conformations for MD simulations.

- `mdsim.py` runs the MD simulation protocol starting from conformations generated in the above step.

- `analysis.py` contains utility functions for performing RMS fluctuation and mutual correlation analysis on trajectory data from the MD simulations.

#### Jupyter notebooks
- `notebooks/lag21_analysis.ipynb` contains the complete analysis of Lag21 MD simulation data analysis using the building blocks defined in `analysis.py`. 

- `notebooks/lag21_plots.ipynb` contains python code to create plots of RMS fluctuations and mutual correlation analysis.

#### Lag21 data
- `lag21/data` contains the bare and GFP bound structures of the Lag21 Nb.

- `lag21/mdsims` contains a shell script to launch the MD simulation protocol on a HPC cluster for the wild type and the L69F mutant.

- `lag21/rosetta_conformations` contains CDR3 conformations obtained as the output of `cdr_sampler.py`, as well as shell scripts to launch CDR3 sampling jobs on a HPC cluster.

- `lag21/results` contains the mutual correlation maps, csv file of the per-residue RMS fluctuations, and these fluctuations heat-mapped on to the structures (wild-type and mutant) of Lag21.