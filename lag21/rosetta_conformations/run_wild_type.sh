#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --job-name=lag21w
#SBATCH --nice=200

# conda environment
eval "$(conda shell.bash hook)"
conda activate peptract

# output path
output_path=wild_type
mkdir -p ${output_path}

# run
python ../../cdr_sampler.py ../data/lag21.pdb -o $output_path -n 8
