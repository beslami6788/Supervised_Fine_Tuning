from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_config, get_peft_model, TaskType, PeftType, LoraConfig, prepare_model_for_kbit_training
import torch
import transformers
import bitsandbytes as bnb
from datasets import load_dataset
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import DatasetDict
import argparse
import sys
import evaluate
import numpy as np
from accelerate import Accelerator

# input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--train', '-i', help="training file in JSON format", required=True)
parser.add_argument('--dev', '-d', help="dev training file in JSON format, if not provided then will do a 70/30 split on training data", nargs="?", const="", default="")
parser.add_argument('--maxlen', help="max input length, default is 1024", type=int, default=1024, const=1024, nargs="?")
parser.add_argument('--finetuned', '-f', help="output fine tuning directory after training", required=True)
parser.add_argument('--model', '-m', help="model directory", required=True)
parser.add_argument('--batchsize', help="batch size, default is 8", type=int, default=2, const=2, nargs="?")
parser.add_argument('--loraconfigR', help="LoraConfig r parameter", type=int, nargs='?', const=8, default=8)
parser.add_argument('--loraalpha', help="LoraConfig lora_alpha parameter", type=int, nargs="?", const=2, default=2)
parser.add_argument('--loradropout', help="LoraConfig lora_dropout parameter", type=float, nargs="?", const=0.00, default=0.00)
parser.add_argument('--epochct', help="number of epochs", type=int, nargs="?", const=2, default=2)
parser.add_argument('--lr', help="training learning rate", type=float, nargs="?", const=0.0001, default=0.0001)
parser.add_argument('--outputdir', help="output directory name", nargs="?", const="outputs", default="outputs")
parser.add_argument('--quantlevel', choices=["4bit", "8bit"], nargs="?", const="4bit", default="4bit", help="quantization level, if using parallelization this must be set to 4bit")
parser.add_argument('--parallelization', '-p', choices=["naivepp", "ddp", "none"], help="parallelization to use with quantization, please review the documentation", nargs="?", const="none", default="none")
args = parser.parse_args()

# functions
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def preprocess_function(examples):
    text_column = "input"
    label_column = "output"
    batch_size = len(examples[text_column])
    inputs = [f"{text_column} : {x} Label : " for x in examples[text_column]]
    targets = [str(x) for x in examples[label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_length - len(sample_input_ids)) + label_input_ids
        try:
            model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_length])
        except RuntimeError:
            print(model_inputs)
            sys.exit()
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    rouge = evaluate.load("rouge")
    predictions, labels = eval_pred

    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]

    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k,v in result.items()}

def preprocess_logits_for_metrics(logits, labels):
    # pred_ids = torch.argmax(logits[0], dim=-1)
    # return pred_ids, labels
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


# setup
output_dir = './' + str(args.outputdir)

model_name = str(args.model) # "/workspace/llama2-70b-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# training parameters and hyperparameters
max_length = args.maxlen # global variable
lr = args.lr
num_epochs = args.epochct
batch_size = args.batchsize
device = "cuda"

# name for output directory
saved_model_dir = args.finetuned

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# LoRA Config
peft_config = LoraConfig(
    r=int(args.loraconfigR),
    lora_alpha=int(args.loraalpha),
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=float(args.loradropout),
    bias="none",
    task_type="CAUSAL_LM"
)

dataset_name = "prob_summ"
checkpoint_name = f"{dataset_name}_{model_name}_{peft_config.peft_type}_{peft_config.task_type}_v1.pt".replace(
    "/", "_"
)

print('processing files')


data_files = {'train' : str(args.train), 'test' : str(args.dev)}
dataset = DatasetDict.from_json(data_files)

processed_datasets = dataset.map(
    preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)


train_dataset = processed_datasets["train"]
eval_dataset = processed_datasets["test"]


train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

print('loading model')
model = None 


if args.quantlevel == "8bit":
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map={"":0})
else:
    if args.parallelization == "none":
        model=AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"":0})
    elif args.parallelization == "ddp":
        device_index= Accelerator().process_index
        device_map = {"": device_index}
        model=AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map=device_map)
    elif args.parallelization == "naivepp":
        model=AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
    else:
        print("Error, unknown parallelization option, exiting now.")
        sys.exit()

model = get_peft_model(model, peft_config)

# use naive pp parallelization
if args.parallelization == "naivepp":
    accelerator = Accelerator()
    model = accelerator.prepare(model)

print('setting up peft_config for model')

# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
optimizer = bnb.optim.Adam8bit(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

training_args = TrainingArguments(
    output_dir = output_dir,
    #max_steps=200,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=0,
    learning_rate=lr,
    weight_decay=0.001,
    # eval_accumulation_steps=2,
    evaluation_strategy="steps", # epoch
    logging_steps=20,
    fp16=True,
    report_to="none", # include to prevent call to WANDB
    eval_steps=0.25,
    save_strategy="steps", # epoch
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_metrics
)


# training code block
print('starting training...')
trainer.train()

trainer.save_model(saved_model_dir)
