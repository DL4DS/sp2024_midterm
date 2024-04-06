#!/bin/bash -l

# Set SCC project
#$ -P ds598

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

# Change this path to point to your project directory
export PYTHONPATH="/projectnb/ds598/students/lilinj/sp2024_midterm:$PYTHONPATH"

#python -m spacy download en_core_web_sm   # download spacy model
python src/demo_model/train.py

### The commands below are used to submit the job to the cluster
### qsub -pe omp 4 -P ds598 -l gpus=1 demo_train.sh
### qsub -l gpus=1 -l gpu_c=7.0 -pe omp 8 demo_train.sh
