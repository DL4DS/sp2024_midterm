# DS598 DL4DS Midterm Project


## Introduction
The goal of this task is to obtain the highest out-of-sample performance in captioning given images by fine tuning a pre-trained multimodal model. The performance metric is CIDEr, and the test performance is evaluated by the VizWiz-Captions Challenge 2021. For this task, I basically started from the source code given by the instructor (BU DS598 DL4DS Spring 2024).


## Approach
In this task, I adopted two approaches: ordinary fine-tuning and PEFT.

### Fine-tuning (src &rarr; ft)
Fine tune all parameters in a pre-trained model using the given training data. The best model is selected via the CIDEr score of validation dataset.

```python
# Please see train_revised.py
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

### PEFT (Parameter-efficient fine-tuning, src &rarr; peft)
Train a LoRA using the training data. The best model is selected in the same way. The LoRA configuration is as follows.

```python
# Please see train_revised.py
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
`trainable params: 614,400 || all params: 470,347,324 || trainable%: 0.13062687266399756`

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
`trainable params: 614,400 || all params: 470,347,324 || trainable%: 0.13062687266399756`

### Inference using the trained model vis PEFT (LoRA)
```python
# Unlike resuming training, is_trainable should be False (Default: False)
model = PeftModel.from_pretrained(m, PEFT_CONFIG_PATH, is_trainable=True, cache_dir=CACHE_DIR)
```


## Pre-trained model: BLIP
### Model: [BLIP (Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation)](https://huggingface.co/Salesforce/blip-image-captioning-large)
### Checkpoints
- `Salesforce/blip-image-captioning-base` &rarr; For fine-tuning
- `Salesforce/blip-image-captioning-large` &rarr; For PEFT


## Modification from the given demo source code
1. Pre-determined directories in train.py, test.py, constants.py
2. **Attention mask** explicitly passed in forward method
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
- Fine-tuning training &rarr; `src/ft/train_revised.py`, `ft_train_revised.sh`
- Resuming fine-tuning &rarr; `src/ft/train_revised_continued.py`, `ft_train_revised.sh`
- PEFT training &rarr; `src/ft/train_revised.py`, `peft_train_revised.sh`
- PEFT resumed training &rarr; `src/ft/train_revised_continued.py`, `peft_train_revised.sh`
For resuming training, just modify the python script path that you are getting at in shell script (PEFT example)
```sh
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

python src/peft/train_revised.py # Write the python script path you are getting at (For resuming training -> python src/peft/train_revised_continued.py)

### The command below is used to submit the job to the cluster
### qsub -pe omp 4 -P ds598 -l gpus=1 peft_train_revised.sh
```

## Performance
- Fine-tuned model test answers &rarr; `src/ft/test_revised.py`, `ft_test.sh`
- PEFT model test answers &rarr; `src/ft/test_revised.py`, `peft_test_revised.sh`


## Findings



## Limitations



## Future work
When the end time of this task was only one day left, the most regrettable thing was not to try more diverse hyperparameters with more diverse models.
1.	Hyperparameter tuning: the hyperparameters of the LoRA, evaluation interval, learning rate, etc.
2.	Should have used array jobs
3.	Dynamic Resource Allocation: Develop a dynamic resource allocation strategy that can adjust the job size based on the available computing resources and time. This could involve creating an algorithm that estimates the optimal batch size, number of epochs, etc., given the constraints.
4.	Performance Modeling: Build a performance model that can predict the execution time of a job based on various factors like batch size, number of epochs, and available resources. This could help in planning and scheduling jobs more effectively.
5.	Efficiency Metrics: Establish efficiency metrics that consider both the computational resources used and the performance of the deep learning model. This could help you balance the trade-off between resource usage and model performance.


## References
1. [CIDEr: Consensus-based image description evaluation](https://ieeexplore.ieee.org/document/7299087)
2. [BLEU: A Misunderstood Metric from Another Age](https://towardsdatascience.com/bleu-a-misunderstood-metric-from-another-age-d434e18f1b37), Medium Post
3. [BLEU Metric](https://huggingface.co/spaces/evaluate-metric/bleu), HuggingFace space



