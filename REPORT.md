# DS598 DL4DS Midterm Project

## Introduction
The project aims to provide image-to-captioning services for blind people using AI technology. The project employs the [blip-image-captioning-base model](https://huggingface.co/Salesforce/blip-image-captioning-base), fine-tuned on the [VizWiz Image Captioning dataset](https://vizwiz.org/tasks-and-datasets/image-captioning/). The optimizer is AdamW with specific settings: a learning rate of 2e-5 and a weight decay of 5e-4. I trained it 15 epochs, and stopped it early at epoch 6, since it was overfitting afterwards. My best CIDEr-D score on the [test dataset](https://eval.ai/web/challenges/challenge-page/739/leaderboard/2006) is 75.37.

## Model performance
### 


## Implementation Details and Challanges

1. I browsed through the image-to-text models on the [huggingface website](https://huggingface.co/models?pipeline_tag=image-to-text&sort=trending) for basic information about these models, and fed dataset images into the reference API to evaluate the pre-trained models' outputs. Then, I selected the models like [blip-image-captioning-base model](https://huggingface.co/Salesforce/blip-image-captioning-base), [blip-image-captioning-large model](https://huggingface.co/Salesforce/blip-image-captioning-large), and [git-base](https://huggingface.co/microsoft/git-base) for fine-tuning.

2. I experimented with various optimizers, including SGD, Adam, and AdamW. Since a high default learning rate would cause all inputs to yield few or identical outputs, I reduced and fine-tuned the learning rate to 2e-5. I also fine-tuned the weight decay to 5e-4. 

3. To prevent model overfitting, I adopted measures such as early stopping, batch size reduction, and L2 regularization.

## Limitation and Reflection
1. Facing with issues like debugging empty outputs, CUDA version mismatches, limited computational resources, and long training times, my exploration of diverse models was constrained. 

2. I didn't try methods like data augmentation and dropout that could have potentially improved the model's robustness and generalization capabilities.

## References
1. [CIDEr: Consensus-based image description evaluation](https://ieeexplore.ieee.org/document/7299087)
2. [BLEU: A Misunderstood Metric from Another Age](https://towardsdatascience.com/bleu-a-misunderstood-metric-from-another-age-d434e18f1b37), Medium Post
3. [BLEU Metric](https://huggingface.co/spaces/evaluate-metric/bleu), HuggingFace space
4. [AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
5. [image_captioning](https://huggingface.co/docs/transformers/main/en/tasks/image_captioning)
6. [BlipForConditionalGeneration](https://huggingface.co/docs/transformers/en/model_doc/blip#transformers.BlipForConditionalGeneration)
