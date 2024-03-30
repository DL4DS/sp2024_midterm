#!/bin/bash -l

# Set SCC project
#$ -P ds598

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

# Change this path to point to your project directory
export PYTHONPATH="/projectnb/ds598/admin/tgardos/sp2024_midterm:$PYTHONPATH" # Set this!!!

python -m spacy download en_core_web_sm   # download spacy model
python /projectnb/ds598/students/nikhitam/sp2024_midterm/src/cnn_lstm/test.py


### The command below is used to submit the job to the cluster
### qsub -pe omp 4 -P ds598 -l gpus=1 cnnlstm_train.sh
