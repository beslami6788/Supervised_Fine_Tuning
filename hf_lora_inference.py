from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import os, sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import DatasetDict
from peft import PeftModel, PeftConfig
import argparse
from pathlib import Path
import pandas as pd
import json
from rouge_score import rouge_scorer
from tqdm import tqdm
from huggingface_hub import login


# setup
# access_token= os.environ["HUGGING_FACE_HUB_TOKEN"]
# login(token=access_token)

parser = argparse.ArgumentParser()
parser.add_argument('--inputFile', '-i', help="input test dataset name, must be in jsonl format", required=True)
parser.add_argument('--finetunedmodeldir', '-d', help="directory where trained model is", required=True)
parser.add_argument('--outputFile', '-o', help="ouput filename for predictions as CSV file", required=True)
parser.add_argument('--model', '-m', help="directory with llama2-hf model", required=True)
parser.add_argument('--maxnewtok', type=int, help="maximum number of new tokens to generate, ignoring the number of tokens in the prompt", nargs="?", default=200, const=200)
parser.add_argument('--temperature', '-t', type=float, help="value for temperature when running inference", nargs="?", default=0.4, const=0.4)
parser.add_argument('--topk', '-k', type=int, help="top_k value when running inference", nargs="?", default=40, const=40)
parser.add_argument('--topp', type=float, help="top_p value when running inference", nargs="?", default=0.95, const=0.95)
parser.add_argument('--quantlevel', choices=["4bit", "8bit"], nargs="?", const="4bit", default="4bit")

args = parser.parse_args()

outFileName = str(args.outputFile)
if not str(args.outputFile).endswith('csv'):
    print('output will be in CSV format, appending csv to output filename')
    outFileName = outFileName + '.csv'

if Path(outFileName).exists():
    print('Error, filename exists in current directory with provided output filename:')
    print(outFileName)
    print('Exiting Now.')
    sys.exit()

if not Path(args.finetunedmodeldir).exists():
    print("Error, the fine-tuned directory provided:")
    print(str(args.finetunedmodeldir))
    print("was not found. Exiting now...")
    sys.exit()

if not Path(args.model).exists():
    print("Error, the llama2-hf model directory provided:")
    print(str(args.model))
    print("was not found. Exiting now...")
    sys.exit()

if not Path(args.inputFile).exists():
    print("Error, the input file:")
    print(str(args.inputFile))
    print("was not found. Exiting now...")
    sys.exit()

from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4_bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    # bnb_4bit_compute_dtype=torch.float32
)


# end setup

# functions
def calc_rougel(generated_text, reference_text):
    """Compute Rouge-L score"""
    # {'rougeL': Score(precision=0.5, recall=0.6, fmeasure=0.5)}
    
    scorer = rouge_scorer.RougeScorer(['rougeL'])
    scores = scorer.score(reference_text, generated_text)
    f1 = scores['rougeL'].fmeasure
    
    return f1



# Prompt-Tuning directory
peft_model_id = args.finetunedmodeldir

# import text to use to test prompt-tuning
print('Loading prompts from file ')
textList = []

jList = []
outputList = []
with open(args.inputFile) as json_file:
    jList = list(json_file)

for j in jList:
    result=json.loads(j)
    textList.append(result)


config = PeftConfig.from_pretrained(peft_model_id)
if args.quantlevel == "8bit":
    model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map='auto')
else:
    model=AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=nf4_config, device_map="auto")

model = PeftModel.from_pretrained(model, peft_model_id)

model = model.bfloat16() # debugging

tokenizer = AutoTokenizer.from_pretrained(args.model)

# added this to resolve RuntimeError
tokenizer.pad_token = tokenizer.bos_token
tokenizer.padding_side="left"
model.config.pad_token_id = model.config.bos_token_id

device = "cuda"

model.to(device)

results = []
for i in tqdm(textList):
    p_text = i["input"]
    p_len = len(p_text)
    inputs = tokenizer(
        i["input"],
        return_tensors="pt"
    )
    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = None
        try:
            outputs = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=args.maxnewtok, do_sample=True, top_p=args.topp, temperature=args.temperature, top_k=args.topk # pad_token_id=tokenizer.eos_token_id, do_sample=True
            )
        except RuntimeError as e:
            print(inputs)
            print(len(inputs["input_ids"][0]))
            print(len(inputs["attention_mask"][0]))
            print(p_text)
            print(e)
            continue
            # sys.exit()
    a = i["output"]
    # predictedText = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    predictedText = tokenizer.batch_decode(outputs)
    ppredictedText = predictedText[0][p_len:]
    ppredictedText = ppredictedText.replace("<|eot_id|>", "")
    ppredictedText = ppredictedText.replace("end_header_id|>", "")
    ppredictedText = ppredictedText.replace("nd_header_id|>", "")   
    ppredictedText = ppredictedText.replace("d_header_id|>", "") 
    ppredictedText = ppredictedText.replace("_header_id|>", "")    
    ppredictedText = ppredictedText.replace("der_id|>", "")    
    ppredictedText = ppredictedText.strip()
    # print('prediction:')
    print(ppredictedText)
    score = calc_rougel(ppredictedText, a)
    results.append((ppredictedText, predictedText[0], score, a))

df = pd.DataFrame({
      'PredictedText' : [x[0] for x in results],
      'ParsedPrediction' : [x[1] for x in results],
      'RougeL' : [x[2] for x in results],
      'GoldTruth' : [x[3] for x in results]
    })

df.to_csv(str(args.outputFile))
print("Created output file named " + str(args.outputFile))
