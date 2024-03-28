#!/bin/bash -l

# Set SCC project
#$ -P ds598
#$ -m beas
#$ -M xyz0906@bu.edu

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

# Change this path to point to your project directory
export PYTHONPATH="/usr4/ds598/xyz0906/sp2024_midterm:$PYTHONPATH"

python src/demo_model/train.py

### The command below is used to submit the job to the cluster
### qsub -pe omp 4 -P ds598 -l gpus=1 demo_train.sh
