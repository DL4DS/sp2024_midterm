#!/bin/bash -l

# Set SCC project
#$ -P ds598

#$ -m beas
#$ -M faridkar@bu.edu

module load miniconda
module load academic-ml/spring-2024

conda activate spring-2024-pyt

# Change this path to point to your project directory
export PYTHONPATH="/projectnb/ds598/students/faridkar/sp2024_midterm:$PYTHONPATH"

python src/demo_model/train.py