#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --job-name=lag21m
#SBATCH --nice=200

# conda environment
eval "$(conda shell.bash hook)"
conda activate peptract

# output path
output_path=L69F
mkdir -p ${output_path}

# run
python ../../cdr_sampler.py ../data/lag21.pdb -m L69F -o $output_path -n 8
