#!/bin/bash -l

# Set SCC project
#$ -P ds598
#$ -l h_rt=30:00:00
#$ -m beas
#$ -M jawicamp@bu.edu

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

# Change this path to point to your project directory
export PYTHONPATH="/projectnb/ds598/students/jawicamp/sp2024_midterm:$PYTHONPATH"

python -m spacy download en_core_web_sm   # download spacy model
python src/demo_model/train.py

### The command below is used to submit the job to the cluster
### qsub -pe omp 4 -P ds598 -l gpus=2 demo_train.sh