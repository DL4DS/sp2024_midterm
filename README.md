# DS598 DL4DS Midterm Project

## Introduction
The project aims to provide image-to-caption services for blind people using Transformer technology. The project employs the [blip-image-captioning-base model](https://huggingface.co/Salesforce/blip-image-captioning-base), fine-tuned on the [VizWiz Image Captioning dataset](https://vizwiz.org/tasks-and-datasets/image-captioning/). The optimizer is [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html) with a learning rate of 2e-5 and a weight decay of 5e-4. The model is set to train for up to 16 epochs, but training is stopped early at epoch 7, since it is overfitting afterwards. The batch sizes of training and validation are 6 and 32 respectively. The model achieved a CIDEr-D score of 75.37 on the [test dataset](https://eval.ai/web/challenges/challenge-page/739/leaderboard/2006).

## Dataset
The dataset used in this project is the VizWiz-Captions dataset, which includes 39,181 images sourced from individuals who are blind. Each image is accompanied by 5 descriptive captions. 

Download the dataset from the website [VizWiz Image Captioning dataset](https://vizwiz.org/tasks-and-datasets/image-captioning/) and update the paths of annotation_file and image_folder in `src/base/dataset.py`.

## Evaluation

In the VizWiz challenge evaluation they refer to five different evaluation metrics although they use CIDr-D as their primary evaluation.

They reference the BLUE metric, but there are limitations to that metric as described in [2] below.

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

## Limitation and Reflection
1. Facing with challenges such as debugging empty predictions, CUDA version mismatches, limited computational resources, and long training times, my experimentation was limited to a few models such as [blip-image-captioning-base model](https://huggingface.co/Salesforce/blip-image-captioning-base), [blip-image-captioning-large model](https://huggingface.co/Salesforce/blip-image-captioning-large), and [git-base](https://huggingface.co/microsoft/git-base) for fine-tuning. 

2. I didn't try methods like data augmentation and dropout that could have potentially improved the model's robustness and generalization capabilities.

## References
1. [CIDEr: Consensus-based image description evaluation](https://ieeexplore.ieee.org/document/7299087)
2. [BLEU: A Misunderstood Metric from Another Age](https://towardsdatascience.com/bleu-a-misunderstood-metric-from-another-age-d434e18f1b37), Medium Post
3. [BLEU Metric](https://huggingface.co/spaces/evaluate-metric/bleu), HuggingFace space
4. [image-to-text models](https://huggingface.co/models?pipeline_tag=image-to-text&sort=trending)
5. [image_captioning](https://huggingface.co/docs/transformers/main/en/tasks/image_captioning)
6. [BlipForConditionalGeneration](https://huggingface.co/docs/transformers/en/model_doc/blip#transformers.BlipForConditionalGeneration)



