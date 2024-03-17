# DS598 DL4DS Midterm Project

## Introduction
For this project, you will train a network to generate captions for the 
[VizWiz Image Captioning dataset](https://vizwiz.org/tasks-and-datasets/image-captioning/).
The images are taken by people who are blind and typically rely on
human-based image captioning services.  Your objective will be to beat a
a baseline score on the [test set leaderboard](https://eval.ai/web/challenges/challenge-page/739/leaderboard/2006).

## Developer Setup

Clone this repo to your directory on the SCC DS598 project space, e.g.
`/projectnb/ds598/students/<userid>`.

Once you have a training script setup, create a shell script, e.g. `train.sh`,
that loads and activates a conda environment and then runs your training
script. An example shell script is below.

```sh
#!/bin/bash -l

# Set SCC project
#$ -P ds598

# load and activate the academic-ml conda environment on SCC
module load miniconda
module load academic-ml/spring-2024
conda activate spring-2024-pyt

# Add the path to your source project directory to the python search path
# so that the local `import` commands will work.
export PYTHONPATH="/projectnb/ds598/students/<userid>/<yourdir>:$PYTHONPATH"

# Update this path to point to your training file
python path/to/train.py

# After updating the two paths above, run the command below from an SCC
# command prompt in the same directory as this file to submit this as a
# batch job.
### qsub -pe omp 4 -P ds598 -l gpus=1 train.sh
```

## Run Example Scripts

When you run the example scripts, make sure to add the path to the repo
folder before running the script. 

```export PYTHONPATH="/projectnb/ds598/path/to/folder:$PYTHONPATH"```

The example shell scripts include this command.


Set the paths in `src/base/constants.py` to the correct paths on your system.

Follow the .sh files to run the code. As an example, to run the `cnnlstm_train.sh`
script, you would run at the command prompt from the base of your local repo
folder:

```sh
$ qsub -pe omp 4 -P ds598 -l gpus=1 cnnlstm_train.sh
Your job 5437870 ("cnnlstm_train.sh") has been submitted
```
As shown, you should get notification that your job was submitted and get a 
job ID number.

You can check your job status by typing:

```sh
$ qstat -u <userid>
ob-ID  prior   name       user         state submit/start at     queue                          slots ja-task-ID 
-----------------------------------------------------------------------------------------------------------------
5437870 0.00000 cnnlstm_tr tgardos      qw    03/14/2024 09:40:24 
```

The above is showing the example output from user `tgardos`.

## Dataset

The dataset is downloaded to 
`/projectnb/ds598/materials/datasets/vizwiz/captions`. There is no need to 
download the dataset again and the path has already been defined in the 
accompanying code.

## Evaluation

In the VizWiz challenge evaluation they refer to five different evaluation
metrics although they use CIDr-D as their primary evaluation.

They reference the BLUE metric, but there are limitations to that metric as
described in [2] below.

**TODO** Explain how to get validation results from your own run.

**TODO** Explain how to submit test json to evaluation server to get test results.

**TODO** Set test result threshold that must be passed.

References:
1. [CIDEr: Consensus-based image description evaluation](https://ieeexplore.ieee.org/document/7299087)
2. [BLEU: A Misunderstood Metric from Another Age](https://towardsdatascience.com/bleu-a-misunderstood-metric-from-another-age-d434e18f1b37), Medium Post
3. [BLEU Metric](https://huggingface.co/spaces/evaluate-metric/bleu), HuggingFace space



