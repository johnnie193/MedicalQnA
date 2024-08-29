from mindnlp.peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    PeftType,
    PeftConfig,
    PeftModel,
)
import argparse
import os

import mindspore
from mindspore.common.api import _no_grad

from tqdm import tqdm
from mindnlp.core.optim import AdamW
from mindnlp import evaluate
from mindnlp.dataset import load_dataset
from mindnlp.engine import set_seed
from mindnlp.transformers import AutoModelForSequenceClassification, AutoTokenizer
from mindnlp.transformers.optimization import get_linear_schedule_with_warmup
from mindnlp.peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    PeftType,
    LoraConfig,
)

import mindspore
import mindnlp
from mindnlp.core import no_grad
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from mindspore._c_expression import _framework_profiler_step_start
from mindspore._c_expression import _framework_profiler_step_end
# 加载tokenizer文件

batch_size = 32
model_name_or_path = "01ai/Yi-6B-Chat"
task = "mrpc"
peft_type = PeftType.PROMPT_TUNING
num_epochs = 20
peft_config = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=8, lora_alpha=16, lora_dropout=0.1)
lr = 3e-4
max_length = 128

tokenizer = AutoTokenizer.from_pretrained("01ai/Yi-6B-Chat", mirror="modelscope")
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    mirror='modelscope',
    ms_dtype=mindspore.float16,
).eval()

if any(k in model_name_or_path for k in ("gpt", "opt", "bloom")):
    padding_side = "left"
else:
    padding_side = "right"
if getattr(tokenizer, "pad_token_id") is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
model = get_peft_model(model, peft_config)
# print(model)
model.print_trainable_parameters()

import mindspore.dataset as ds
dataset = ds.CSVDataset(dataset_files="./dataset/output.csv", column_names=['questions', 'answers'], column_defaults=[])
train_dataset, validation_dataset = dataset.shuffle(64).split([0.9, 0.1])

import numpy as np
from mindnlp.dataset import BaseMapFunction
from threading import Lock
lock = Lock()

class MapFunc(BaseMapFunction):
    def __call__(self, inputs, outputs):
        lock.acquire()
        model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True)
        labels = tokenizer(outputs, max_length=max_length, padding="max_length", truncation=True)
        lock.release()
        labels = labels['input_ids']
        labels = np.where(np.equal(labels, tokenizer.pad_token_id), -100, labels)
        return model_inputs['input_ids'], model_inputs['attention_mask'], labels


def get_dataset(dataset, tokenizer, shuffle=True):
    input_colums=['questions', 'answers']
    output_columns=['input_ids', 'attention_mask', 'labels']
    dataset = dataset.map(MapFunc(input_colums, output_columns),
                          input_colums, output_columns)
    if shuffle:
        dataset = dataset.shuffle(64)
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = get_dataset(train_dataset, tokenizer)
eval_dataset = get_dataset(validation_dataset, tokenizer, shuffle=False)

optimizer = mindnlp.core.optim.AdamW(model.trainable_params(),lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataset) * num_epochs),
)

def forward_fn(**batch):
    outputs = model(**batch)
    loss = outputs.loss
    return loss

grad_fn = mindspore.value_and_grad(forward_fn, None, model.trainable_params())

def train_step(**batch):
    loss, grads = grad_fn(**batch)
    optimizer(grads)
    return loss

for epoch in range(num_epochs):
    model.set_train()
    total_loss = 0
    train_total_size = train_dataset.get_dataset_size()
    for step, batch in enumerate(tqdm(train_dataset.create_dict_iterator(), total=train_total_size)):
        loss = train_step(**batch)
        total_loss += loss.float()
        lr_scheduler.step()

    model.set_train(False)
    eval_loss = 0
    eval_preds = []
    eval_total_size = eval_dataset.get_dataset_size()
    for step, batch in enumerate(tqdm(eval_dataset.create_dict_iterator(), total=eval_total_size)):
        with mindspore._no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.float()
        eval_preds.extend(
            tokenizer.batch_decode(ops.argmax(outputs.logits, -1).asnumpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataset)
    eval_ppl = ops.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataset)
    train_ppl = ops.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")