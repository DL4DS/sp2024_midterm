# DS598 DL4DS Midterm Project

## Introduction
The project aims to provide image-to-captioning services for blind people using AI technology. The project employs the [blip-image-captioning-base model](https://huggingface.co/Salesforce/blip-image-captioning-base), fine-tuned on the [VizWiz Image Captioning dataset](https://vizwiz.org/tasks-and-datasets/image-captioning/). The optimizer is AdamW with specific settings: a learning rate of 2e-5 and a weight decay of 5e-4. I trained it 15 epochs, and stopped it early at epoch 3, since it was overfitting afterwards. My best CIDEr-D score on the [test dataset](https://eval.ai/web/challenges/challenge-page/739/leaderboard/2006) is 75.37.
