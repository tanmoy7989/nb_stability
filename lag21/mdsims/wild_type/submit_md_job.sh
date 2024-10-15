#!/bin/bash
#SBATCH --time=300:00:00
#SBATCH --job-name=lag21wa
#SBATCH --output=logs/sim_%a.log
#SBATCH --nice=200
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --array=0-55

# base paths
PROJECT_HOME=../../../
NB_HOME=$PROJECT_HOME/lag21

# conda environment
eval "$(conda shell.bash hook)"
conda activate nbenv

# gpu
module load cuda/11.0
export OPENMM_CUDA_COMPILER=$(which nvcc)

# indexes
n_conformations=8
idx_conf=$(( ${SLURM_ARRAY_TASK_ID} % ${n_conformations} ))
idx_temp_=$(( ${SLURM_ARRAY_TASK_ID} - ${idx_conf} ))
idx_temp=$(( ${idx_temp_} / ${n_conformations} ))

# temperature
TEMPS=(290 300 317 336 355 398 421)
my_temp=${TEMPS[${idx_temp}]}

# random number seed
my_rng=$(( $RANDOM % 1000 + $idx_conf ))

# input file
input_pdb=$NB_HOME/rosetta_conformations/wild_type/conf_$idx_conf.pdb

# output path
output_path=${my_temp}K/run_$idx_conf
mkdir -p $output_path

# run
python $PROJECT_HOME/mdsim.py $input_pdb -o $output_path/nb \
       -t $my_temp -r $my_rng

