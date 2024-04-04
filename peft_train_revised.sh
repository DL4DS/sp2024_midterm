#!/bin/bash -l

# Set SCC project
#$ -P ds598

# load and activate the academic-ml conda environment on SCC
module load miniconda
module load academic-ml/spring-2024
conda activate spring-2024-pyt

# Change this path to point to your project directory
export PYTHONPATH="/projectnb/ds598/students/spark618/sp2024_midterm:$PYTHONPATH"

python -m spacy download en_core_web_sm   # download spacy model
pip install peft # install peft package

huggingface-cli login --token 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

python src/peft/train_revised_continued.py

### The command below is used to submit the job to the cluster
### qsub -pe omp 4 -P ds598 -l gpus=1 peft_train_revised.sh

