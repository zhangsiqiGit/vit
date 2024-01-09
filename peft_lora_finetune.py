"""
Introduction
In this notebook, we will learn how to use LoRA from 🤗 PEFT to fine-tune an image classification model by ONLY using 0.77% of the original trainable parameters of the model.

LoRA adds low-rank "update matrices" to certain blocks in the underlying model (in this case the attention blocks) and ONLY trains those matrices during fine-tuning. During inference, these update matrices are merged with the original model parameters. For more details, check out the original LoRA paper.

Let's get started by installing the dependencies.

Note that this notebook builds on top the official image classification example notebook.
"""

"""
Install dependencies
!pip install transformers accelerate evaluate datasets git+https://github.com/huggingface/peft -q
"""

# import necessary
import transformers
import accelerate
import peft
print(f"Transformers version: {transformers.__version__}")
print(f"Accelerate version: {accelerate.__version__}")
print(f"PEFT version: {peft.__version__}")

# select a model checkpoint to fine-tune
model_checkpoint = "google/vit-base-patch16-224-in21k"  # pre-trained model from which to fine-tune

# load a dataset
# dataset is huggingface dataset, which should load from local
from datasets import load_dataset, load_from_disk

# dataset = load_dataset("food101", split="train[:5000]")  # load from hub

# load parquet data files
dataset = load_dataset("parquet", data_files={'train':[
    "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/train-00000-of-00008.parquet", 
    "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/train-00001-of-00008.parquet",
    "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/train-00002-of-00008.parquet",
    "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/train-00003-of-00008.parquet",
    "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/train-00004-of-00008.parquet",
    "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/train-00005-of-00008.parquet",
    "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/train-00006-of-00008.parquet",
    "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/train-00007-of-00008.parquet"],
    'test': [
        "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/validation-00000-of-00003.parquet",
        "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/validation-00001-of-00003.parquet",
        "/opt/tiger/algo/vit_lora/.cache/huggingface/hub/datasets/food101/data/validation-00002-of-00003.parquet"
    ]})

# prepare datasets for training and evaluation
# 1.Prepare label2id and id2label dictionaries. This will come in handy when performing inference and for metadata information.
print('数据：', dataset['train'].num_rows)
dataset = dataset['train'].select(range(5000))
# dataset = dataset["train"][:5000]

labels = dataset.features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = i
    id2label[i] = label

print(id2label[2])

# 2.We load the image processor of the model we're fine-tuning.
from transformers import AutoImageProcessor

image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
print(image_processor)

# 3.Using the image processor we prepare transformation functions for the datasets. These functions will include augmentation and pixel scaling.
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
train_transforms = Compose(
    [
        RandomResizedCrop(image_processor.size["height"]),
        RandomHorizontalFlip(),
        ToTensor(),
        normalize,
    ]
)

val_transforms = Compose(
    [
        Resize(image_processor.size["height"]),
        CenterCrop(image_processor.size["height"]),
        ToTensor(),
        normalize,
    ]
)


def preprocess_train(example_batch):
    """Apply train_transforms across a batch."""
    example_batch["pixel_values"] = [train_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch


def preprocess_val(example_batch):
    """Apply val_transforms across a batch."""
    example_batch["pixel_values"] = [val_transforms(image.convert("RGB")) for image in example_batch["image"]]
    return example_batch

# 4.We split our mini dataset into training and validation.
# split up training into training + validation
splits = dataset.train_test_split(test_size=0.1)
train_ds = splits["train"]
val_ds = splits["test"]
# 5.We set the transformation functions to the datasets accordingly.
train_ds.set_transform(preprocess_train)
val_ds.set_transform(preprocess_val)

# Load and prepare a model
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer

model = AutoModelForImageClassification.from_pretrained(
    model_checkpoint,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
print_trainable_parameters(model)

from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
lora_model = get_peft_model(model, config)
print_trainable_parameters(lora_model)

# Training arguments
from transformers import TrainingArguments, Trainer


model_name = model_checkpoint.split("/")[-1]
batch_size = 128

args = TrainingArguments(
    f"{model_name}-finetuned-lora-food101",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-3,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=batch_size,
    fp16=True,
    # num_train_epochs=5,
    num_train_epochs=1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    label_names=["labels"],
)

# Prepare evaluation metric
import numpy as np
import evaluate

metric = evaluate.load("accuracy")


# the compute_metrics function takes a Named Tuple as input:
# predictions, which are the logits of the model as Numpy arrays,
# and label_ids, which are the ground-truth labels as Numpy arrays.
def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

import torch


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

# Train and evaluate
trainer = Trainer(
    model,
    args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=image_processor,
    compute_metrics=compute_metrics,
    data_collator=collate_fn,
)
train_results = trainer.train()
print(train_results)

print(trainer.evaluate(val_ds))
# saved adapter_model.bin & adapter_config.json locally
lora_model.save_pretrained(f"{model_name}-finetuned-lora-food101")


# Inference
from peft import PeftConfig, PeftModel


config = PeftConfig.from_pretrained("vit-base-patch16-224-in21k-finetuned-lora-food101")
model = model = AutoModelForImageClassification.from_pretrained(
    config.base_model_name_or_path,
    label2id=label2id,
    id2label=id2label,
    ignore_mismatched_sizes=True,  # provide this in case you're planning to fine-tune an already fine-tuned checkpoint
)
# Load the Lora model
inference_model = PeftModel.from_pretrained(model, "vit-base-patch16-224-in21k-finetuned-lora-food101")

from PIL import Image
import requests

url = "https://huggingface.co/datasets/sayakpaul/sample-datasets/resolve/main/beignets.jpeg"
image = Image.open(requests.get(url, stream=True).raw)
# image
image_processor = AutoImageProcessor.from_pretrained("vit-base-patch16-224-in21k-finetuned-lora-food101/checkpoint-9")
# prepare image for the model
encoding = image_processor(image.convert("RGB"), return_tensors="pt")
print(encoding.pixel_values.shape)

# forward pass
with torch.no_grad():
    outputs = inference_model(**encoding)
    logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", inference_model.config.id2label[predicted_class_idx])
