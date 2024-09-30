from dataclasses import dataclass, field
from typing import Optional
from transformers import LlamaTokenizerFast
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from peft import PeftModel
from tqdm import tqdm
from transformers import Adafactor, AutoTokenizer, HfArgumentParser, pipeline, BertForSequenceClassification, BertTokenizer
import wandb
import pandas as pd
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
from trl.core import LengthSampler
from nltk.tokenize import word_tokenize
import os
import argparse

@dataclass
class ScriptArguments:
    quantized: Optional[bool] = field(default=True, metadata={"help": "quantized or not"})
    data_amount: Optional[int] = field(default=100000, metadata={"help": "how many new data wanted"})
    challenging_data_amount: Optional[int] = field(default=10000, metadata={"help": "number of challenging sentences"})
    data_folder: Optional[str] = field(default="data", metadata={"help": "the folder for saving generated data"})
    start_with_prefix: Optional[bool] = field(default=True, metadata={"help": "Starts with a prefix"})
    generation_model: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "which generation model to use"})
    complete_or_not: Optional[bool] = field(default=False, metadata={"help": "whether to use complete sentence or not"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

# Create the data folder
os.makedirs(script_args.data_folder, exist_ok=True)

tqdm.pandas()

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define arguments for the sentiment analysis pipeline
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

tokenizer = AutoTokenizer.from_pretrained(script_args.generation_model)

# Target classification model
classifier_target = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True, device=device)

def get_pos_scores_tar(text):
    result = classifier_target(text)[0]
    if result[0]['label'] == "POSITIVE":
        return result[0]['score']
    else:
        return 1 - result[0]['score']

if getattr(tokenizer, "pad_token", None) is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to build dataset
def build_dataset(dataset_name="imdb", input_min_text_length=25, input_max_text_length=25):
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    def tokenize(sample):
        words = word_tokenize(sample["review"])
        first_five_words = ' '.join(words[:5])
        if script_args.start_with_prefix:
            sample["input_ids"] = tokenizer.encode(first_five_words)
            sample["query"] = tokenizer.decode(sample["input_ids"])
        else:
            sample["input_ids"] = []
            sample["query"] = ""
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds

dataset = build_dataset()

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

current_device = Accelerator().local_process_index

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

print("Quantized:", script_args.quantized)
if script_args.quantized:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.generation_model,
        load_in_4bit=True,
        device_map="balanced",
        peft_config=lora_config,
    )
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.generation_model,
        device_map="balanced",
    )

# Reference classification model
def get_pos_scores_ref(text):
    target = f"""Question: Find the sentiment of this text. Answer with positive or negative: that is far too tragic to merit such superficial treatment
Answer: negative
Question: Find the sentiment of this text. Answer with positive or negative: a smile on your face
Answer: positive
Question: Find the sentiment of this text. Answer with positive or negative: saw how bad this movie was
Answer: negative
Question: Find the sentiment of this text. Answer with positive or negative: the greatest musicians
Answer: positive
Question: Find the sentiment of this text. Answer with positive or negative: {text}
Answer:"""
    encoding = tokenizer.encode_plus(target, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encoding)[0]
    probs = torch.softmax(outputs[0][-1].detach().to("cpu"), 0)
    neg_prob_raw = probs[8178]
    pos_prob_raw = probs[6374]
    pos_prob = pos_prob_raw / (pos_prob_raw + neg_prob_raw).item()
    return pos_prob

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "max_new_tokens": 50,
}

# Function to return complete sentences
def get_complete_sentences(text):
    end_positions = [text.rfind('.'), text.rfind('!'), text.rfind('?')]
    end_positions = [pos for pos in end_positions if pos != -1]
    if not end_positions:
        return text
    return text[:max(end_positions) + 1]

# Generate new data
bs = script_args.data_amount
game_data = dict()
dataset.set_format("pandas")
df_batch = dataset[:].sample(bs, replace=True)
game_data["query"] = df_batch["query"].tolist()
query_tensors = df_batch["input_ids"].tolist()

response_tensors = []

for i in tqdm(range(bs)):
    output = model.generate(
        input_ids=torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device),
        **generation_kwargs
    ).squeeze()
    response_tensors.append(output)

if script_args.complete_or_not:
    game_data["response"] = [get_complete_sentences(tokenizer.decode(response_tensors[i])) for i in range(bs)]
else:
    game_data["response"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]

df = pd.DataFrame(game_data)

# Save the generated data
df.to_pickle(script_args.data_folder + '/generation_raw.pkl')
csv_filename = script_args.data_folder + '/generation_raw.csv'
df.to_csv(csv_filename, index=False)

# Initialize DataFrame
df['score_ref'] = None
df['score_tar'] = None

for i in tqdm(range(len(df['query']))):
    df.at[i, 'score_ref'] = get_pos_scores_ref(df['response'][i])
    df.at[i, 'score_tar'] = get_pos_scores_tar(df['response'][i])
df['score_diff'] = df['score_ref'] - df['score_tar']

# Save the score data
df.to_pickle(script_args.data_folder + '/score.pkl')
csv_filename = script_args.data_folder + '/score.csv'
df.to_csv(csv_filename, index=False)

# Sort the DataFrame based on 'score_diff' in descending order and save top challenging data
df_sorted = df.sort_values(by='score_diff', ascending=False)
top_challenging = df_sorted.head(script_args.challenging_data_amount)
challenging_csv_filename = script_args.data_folder + '/challenging_score.csv'
top_challenging.to_csv(challenging_csv_filename, index=False)
