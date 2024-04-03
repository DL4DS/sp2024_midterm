#!/bin/bash -l

# Set SCC project
#$ -P ds598

# load and activate the academic-ml conda environment on SCC
module load miniconda
module load academic-ml/spring-2024
conda activate spring-2024-pyt

# Add the path to your source project directory to the python search path
# so that the local `import` commands will work.
export PYTHONPATH="/projectnb/ds598/students/weimai/sp2024_midterm:$PYTHONPATH"

# Update this path to point to your training file
python -m spacy download en_core_web_sm 
# python src/demo_model/wei-train.py
python src/demo_model/train.py

# After updating the two paths above, run the command below from an SCC
# command prompt in the same directory as this file to submit this as a
# batch job.
### qsub -pe omp 4 -P ds598 -l gpus=1 train.sh