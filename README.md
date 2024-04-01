# DS598 DL4DS Midterm Project


## Introduction
The goal of this task is to obtain the highest out-of-sample performance in captioning given images by fine-tuning a pre-trained multimodal model. The performance metric is CIDEr, and the test performance is evaluated by the VizWiz-Captions Challenge 2021. For this task, I basically started from the source code given by the instructor (BU DS598 DL4DS Spring 2024).


## Approach
In this task, I adopted two fine-tuning approaches: ordinary fine-tuning and PEFT.

### Ordinary fine-tuning (Path: src/ft)
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

### PEFT (Parameter-efficient fine-tuning, Path: src/peft)
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
- `Salesforce/blip-image-captioning-base` &rarr; For ordinary fine-tuning
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

### Optimizer & Learning rate
```python
# optimizer and learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
```

### Scripts
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

### The number of epochs
- Due to time limit, the number of epoch for fine-tuning was **eight**.
- The number of epoch for PEFT was **21**.


## Performance
(Please see `res/ft` and `res/peft`)

### Validation performance
- Both the fine-tuning and PEFT showed improvement over training.
- The X-axis in images below indicates epoch.

![alt text](https://github.com/S-Park1228/ds598_sp2024_midterm/tree/img/result_ft.png?raw=true)
The plot for the ordinary fine-tuning: results fetched from epochs 1\~4 (`ft_train_revised.sh.e5763909`), epochs 5\~8 (`ft_train_revised.sh.e5790506`)

The plot for the PEFT: results fetched from epochs 1\~13 (`peft_train_revised.sh.e5769591`), epoch 14 (`peft_train_revised.sh.e5779377`), epochs 15\-21 (`peft_train_revised.sh.e5790125`)


### Test performance
- Testing fine-tuned model test answers &rarr; `src/ft/test_revised.py`, `ft_test.sh`
- Testing PEFT model test answers &rarr; `src/ft/test_revised.py`, `peft_test_revised.sh`
- Fine-tuning CIDEr (`test_captions_ft.json`): **56.43**
- PEFT CIDEr (`test_captions_revised_continued2`): **77.17** &rarr; [VizWiz-Captions Challenge 2021 (test-standard2021)](https://eval.ai/web/challenges/challenge-page/739/overview)


## Limitations
Fine-tuning models through either ordinary fine-tuning or PEFT seemingly shows improvement in performance. However, if we compare test performance between the pre-trained model and fine-tuned model, we can see that fine-tuning was not effective in this case.
- `Salesforce/blip-image-captioning-base` test CIDEr (`test_captions_ft_blip.json`): **53.51**
-  `Salesforce/blip-image-captioning-large` test CIDEr (`test_captions_blip.json`): **76.88**
  
The reason seems that the pre-trained model is too large (the number of parameters of large model &rarr; 469,732,924) to be fine-tuned.

## Future work
1. It is important to consider comparing between the size of training data and pre-trained model capacity when it comes to fine-tuning.
2. Hyperparameter tuning: the hyperparameters of the LoRA, evaluation interval, learning rate, etc.
3. Using array jobs necessary


