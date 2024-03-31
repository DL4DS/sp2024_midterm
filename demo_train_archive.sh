#!/bin/bash -l

# Set SCC project
#$ -P ds598

module load miniconda
conda activate dl4ds

## Change this path to point to your project directory
export PYTHONPATH="/projectnb/ds598/students/yukez/midterm/sp2024_midterm:$PYTHONPATH"

python src/demo_model/train_archive.py

### The command below is used to submit the job to the cluster
### qsub -pe omp 4 -P ds598 -l gpus=1 demo_train_archive.sh
### The result run as demo_train.sh before