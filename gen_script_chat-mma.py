import os

import random
import numpy as np
import torch
import sys

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch

import numpy as np

import torch.nn as nn
import torch.optim as optim

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
        
from transformers import WEIGHTS_NAME, CONFIG_NAME
import json

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
def load_data(path):
    with open(path, 'rb') as file:
        data = json.load(file)
    return data

def load_model(name, dir_ = 'models/'):
    output_dir = "{}/{}/".format(dir_, name)
    model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    return model, tokenizer
        
if __name__ == "__main__":
    set_seed(42)
    
    print('-------------Initialize the model and tokenizer-------------')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "RLHF/t5_small" 

    rlhf_model, rlhf_tokenizer = load_model(model_name)
    
    rlhf_model.to(device)
      
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'prompts.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'answers_chat-mma.json'
    
    print('-------------Loading dataset from {}--------------------'.format(input_file))   
    
    prompts = load_data(input_file)
    
    MAX_SEQ_LENGTH= 512
    
    generated_answers = []
    
    print('-------------Starting generation--------------------')    
    
    for sample in prompts:
        question = "Answer to the question: " + sample["question"] + '\n'
        if "choices" in sample and sample["choices"] is not None:
            question += "Answer options: " + " ".join("{}) {}".format(i, choice) for i, choice in enumerate(sample["choices"]))
            
        model_inputs = rlhf_tokenizer(question, padding='max_length', truncation=True,
                                      max_length=MAX_SEQ_LENGTH, return_tensors="pt")
        
        beam_output = rlhf_model.generate(
                        input_ids=model_inputs['input_ids'].to(rlhf_model.device),
                        attention_mask=model_inputs['attention_mask'].to(rlhf_model.device),
                        num_beams = 5,
                        no_repeat_ngram_size = 2, 
                        # early_stopping = True,
                        max_length=512)
        
        sample['model_answer'] = rlhf_tokenizer.batch_decode(beam_output, skip_special_tokens=True)[0]
        generated_answers.append(sample)
    
    with open(output_file, 'w') as file:
        json.dump(generated_answers, file)
        
    print('Answers have been successfully generated. Check the file {}'.format(output_file))
    
