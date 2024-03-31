# DS598 DL4DS Midterm Project


## Introduction
The goal of this task is to obtain the highest out-of-sample performance in captioning given images by fine tuning a pre-trained multimodal model. The performance metric is CIDEr, and the test performance is evaluated by the VizWiz-Captions Challenge 2021. For this task, I basically started from the source code given by the instructor (BU DS598 DL4DS Spring 2024).


## Approach
In this task, I adopted two approaches: ordinary fine-tuning and PEFT.

### Fine-tuning
Fine tune all parameters in a pre-trained model using the given training data. The best model is selected via the CIDEr score of validation dataset.

```python
from src.base.constants import *
from src.base.helpers import *
from transformers import AutoProcessor
from transformers import BlipForConditionalGeneration


CACHE_DIR = os.environ.get("TRANSFORMERS_CACHE")

# Save paths for fine-tuning
create_directory(FT_SAVE_PATH)
create_directory(FT_SAVE_PATH + "/examples")

### Modified codes ###
# Due to the limits in the memory of available gpus, the following model is used for fine-tuning
checkpoint = "Salesforce/blip-image-captioning-base"

# Instantiate the processor for BLIP model
try:
    processor = AutoProcessor.from_pretrained(checkpoint, cache_dir=CACHE_DIR)
except Exception as e:
    print("You need to pick a pre-trained model from HuggingFace.")
    print("Exception: ", e)

# Instantiate the model for BLIP model
try:
    model = BlipForConditionalGeneration.from_pretrained(checkpoint, cache_dir=CACHE_DIR)
except Exception as e:
    print("You need to pick a pre-trained model from HuggingFace.")
    print("Exception: ", e)
```

### PEFT (Parameter-efficient fine-tuning)
Train a LoRA using the training data. The best model is selected in the same way. The LoRA configuration is as follows.

```python
from peft import LoraConfig, get_peft_model


# Save paths for PEFT
create_directory(PEFT_SAVE_PATH)
create_directory(PEFT_SAVE_PATH + "/examples")

# For the PEFT, checkpoint should be replaced with the following
checkpoint = "Salesforce/blip-image-captioning-large"

# In addition to the different checkpoint, the LoraConfig should be defined
config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "value"]
)


model = get_peft_model(model, config)

model.print_trainable_parameters()
```
trainable params: 614,400 || all params: 470,347,324 || trainable%: 0.13062687266399756

### Resuming PEFT training
```python
from peft import PeftModel, PeftConfig


# Instantiate the BLIP model in the same way
m = BlipForConditionalGeneration.from_pretrained(checkpoint, cache_dir=CACHE_DIR)
# Load the saved lora configuration
PEFT_CONFIG_PATH = f"{PEFT_SAVE_PATH}/best_model"
# This will load the trained LoRA parameters saved via safetensor.
model = PeftModel.from_pretrained(m, PEFT_CONFIG_PATH, is_trainable=True, cache_dir=CACHE_DIR) # Unless is_trainable=True, the loaded parameters cannot be trained.

model.print_trainable_parameters()
```
trainable params: 614,400 || all params: 470,347,324 || trainable%: 0.13062687266399756

### Inference using the trained model vis PEFT (LoRA)
```python
# Unlike resuming training, is_trainable should be False (Default: False)
model = PeftModel.from_pretrained(m, PEFT_CONFIG_PATH, is_trainable=True, cache_dir=CACHE_DIR)
```


## Pre-trained model: BLIP
### Model: [BLIP (Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation)](https://huggingface.co/Salesforce/blip-image-captioning-large)
### Checkpoint
- Salesforce/blip-image-captioning-base -> For fine-tuning
- Salesforce/blip-image-captioning-large -> For PEFT


## Modification from the given demo source code
1. Pre-determined directories in train.py, test.py, constants.py
2. Attention mask explicitly passed in forward method
   ```python
   def train(loger, train_dataloader, model, optimizer, device, processor):
    model.train()

    for idx, batch in progress_bar:
        input_ids = batch.pop("input_ids").to(device)
        pixel_values = batch.pop("pixel_values").to(device)
        # Elicit attention masks
        attention_mask = batch.pop("attention_mask").to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            # Explicitly pass attention masks in forward process
            attention_mask=attention_mask,
            labels=input_ids
        )

        loss = outputs.loss
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
        loss.backward()

        optimizer.step()

        # Update progress bar with loss info
        progress_bar.set_postfix({"loss": loss.item()})

    return loss.item()
   ```

## Training
```python
# optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
```

## Performance


## Findings


## Limitations


## Future work


For this project, you will train a network to generate captions for the 
[VizWiz Image Captioning dataset](https://vizwiz.org/tasks-and-datasets/image-captioning/).
The images are taken by people who are blind and typically rely on
human-based image captioning services.  Your objective will be to beat a
a baseline score on the [test set leaderboard](https://eval.ai/web/challenges/challenge-page/739/leaderboard/2006).


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



