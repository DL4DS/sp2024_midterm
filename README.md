# DS598 DL4DS Midterm Project

## Introduction
For this project, we had to train a model to generate captions for the 
[VizWiz Image Captioning dataset](https://vizwiz.org/tasks-and-datasets/image-captioning/).
The images are taken by people who are blind and typically rely on
human-based image captioning services. We choose a model from Hugging Face, and then train and test. Our objective is to to beat a
a baseline score on the [test set leaderboard](https://eval.ai/web/challenges/challenge-page/739/leaderboard/2006). For this project we aimed for 50 as Cider Score to pass.


## Dataset
### Overview
The VizWiz-Captions dataset is tailored for image captioning tasks, containing a rich collection of images and captions aimed at training and evaluating machine learning models.

### Dataset Composition

#### Images
- **Training Images**: 23,431
- **Validation Images**: 7,750
- **Test Images**: 8,000

#### Captions
- **Training Captions**: 117,155
- **Validation Captions**: 38,750
- **Test Captions**: 40,000 (Note: Captions for the test set are not publicly available)

### Dataset Organization

The dataset files are organized into three main categories: training, validation, and test sets, with each category containing its respective images.

#### Annotations and APIs

- Annotations for images and captions are split across two JSON files for the training and validation datasets. 
- The test split captions are kept private to prevent data leakage and ensure a fair evaluation of models.
- Each image in the dataset is associated with a "text_detected" flag. This flag is set to `true` if text is detected in the image by at least three out of five crowdsourced annotators. Otherwise, it is set to `false`.
- APIs are included to demonstrate how to efficiently parse the JSON annotation files and to evaluate models against the provided ground truth.


## Chosen Model and Fine Tuning

### Microsoft Git Base

The Generative Image-to-Text (GIT) model combines the Transformer decoder architecture with CLIP image tokens and text tokens, aimed at generating text based on both visual and textual inputs. Trained through "teacher forcing" on numerous image-text pairs, its primary function is to predict subsequent text tokens based on given image tokens and the sequence of preceding text tokens.

Unlike traditional models that might only focus on text or images, GIT utilizes a bidirectional attention mask for image patches, allowing full visibility and contextual understanding of the visual content. However, for text tokens, it employs a causal attention mask, meaning the model can only "see" the text tokens that precede the one it's currently predicting, ensuring a coherent and contextually relevant textual output.

### Microsoft Git Large:
The Generative Image-to-Text (GIT) model, particularly its large-sized variant, is a cutting-edge Transformer-based architecture designed for converting visual inputs into textual descriptions. GIT leverages CLIP image tokens along with text tokens, training through a method known as "teacher forcing" with numerous image-text pairs. Its primary objective is to sequentially predict text tokens by integrating the information from both the provided image tokens and the sequence of previously generated text tokens.

### Fine Tuning and Score

## Experiment Phases

#### Phase 1: Initial Setup with Git Base Model
- **Model:** Microsoft Git Base
- **Optimizer:** Stochastic Gradient Descent (SGD)
- **Learning Rate:** 0.01
- **Momentum:** 0.9
- **Outcome:** The initial CIDEr Score obtained was 10.52, which was below our expectations.

#### Phase 2: Transition to Git Large Model
- **Model:** Microsoft Git Large
- **Optimizer:** AdamW
- **Learning Rate:** 0.00001
- **Outcome:** A significant improvement was observed with a CIDEr Score of 68.69, surpassing our baseline target. This highlighted the Git Large model's superior capability in handling the task.

#### Phase 3: Performance Enhancement
- To further enhance the model's performance, we introduced a modification in the validation process within `test.py`. By implementing a more sophisticated generation strategy, we aimed to refine the output quality. The CIDEr Score is 79.53
- **Code Addition:**
  ```sh
  output = model.generate(num_beams = 10, pixel_values=pixel_values, max_length=50)

## Microsoft Git Models: Base vs Large - A Comparative Overview

### Model Size and Capacity

#### Git Base Model
- **Size:** Smaller, optimized for a balance between performance and computational efficiency.
- **Capacity:** Designed for environments with limited resources, offering faster inference times without significantly compromising result quality.

#### Git Large Model
- **Size:** Larger, with substantially more parameters than the base model.
- **Capacity:** Enhanced ability to comprehend and generate complex code and text, suitable for a broader range of intricate tasks.

### Performance and Accuracy

- **Git Large Model:** Delivers higher accuracy and superior performance on sophisticated code-related tasks, attributed to its expanded capacity for capturing complex data patterns. The trade-off is higher computational demand and slower inference speeds.
- **Git Base Model:** While it may be slightly less accurate on complex tasks, it maintains robust performance with the advantage of quicker inference times, ideal for real-time applications or environments with constrained computational resources.

### Computational Resources and Inference Time

- **Git Large Model:** Necessitates greater computational power and memory, which may increase costs and slow down response times, particularly in real-time applications.
- **Git Base Model:** More computationally economical, facilitating easier deployment across various environments, especially those with limited hardware.

## Conclusion

In conclusion, the experiment clearly demonstrates the superior performance of the Git Large model, as evidenced by its impressive CIDEr score of 75.93. This indicates a significantly enhanced capability in generating relevant and contextually accurate captions compared to the Git Base model. However, this improved performance comes at the cost of considerably higher computational resource requirements, as reflected in the computation time cost, which was notably higher than that of the Git Base model. Therefore, while the Git Large model offers advanced capabilities for complex tasks, its deployment should be carefully considered in scenarios where computational efficiency and time constraints are critical factors.



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

Note that there are train and test scripts for the two folders already.

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

### Validation Results

Validation set results are reported in the CNN-LSTM example and code for reporting validation results are in the demo model code.

### Test Results

As is typically the case, the test dataset labels are withheld, and so the only way to get test results is to produce predicted captions and
then submit them to the VizWiz Image Captioning [Evaluation Server](https://eval.ai/web/challenges/challenge-page/739/overview). There are
scripts in both model directories to create the test submission file, although the demo model test script will have to be updated with model 
information.

Create an account on the [Evaluation Server](https://eval.ai/web/challenges/challenge-page/739/overview) and submit your test predictions
to get your result.

Step-by-step instructions will be added here shortly.

State-of-the-art CIDEr-D scores on VizWiz Image Captioning is ~125. We're asking that you get a **minimum CIDEr-D test score of 50**.

## References

1. [CIDEr: Consensus-based image description evaluation](https://ieeexplore.ieee.org/document/7299087)
2. [BLEU: A Misunderstood Metric from Another Age](https://towardsdatascience.com/bleu-a-misunderstood-metric-from-another-age-d434e18f1b37), Medium Post
3. [BLEU Metric](https://huggingface.co/spaces/evaluate-metric/bleu), HuggingFace space



