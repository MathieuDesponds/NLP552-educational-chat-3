import json 
from langdetect import detect
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gensim
import torch
import math
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoConfig, AutoModelWithLMHead, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments


def split_train_test(path):
    """
        Reads the JSON list file given in argument and output the result
    """
    with open(path, 'rb') as f :
        data = json.load(f)
    
    questions = set()
    for sample in data:
        try:
            if detect(sample['question']) == 'en':
                questions.add(sample['sol_id'])
        except:
            pass

    questions = list(questions)

    train_size = int(0.9*len(questions))

    train_questions = questions[:train_size]
    val_questions = questions[train_size:]
    
    train = []
    train_has = []

    val = []
    val_has = []
    for sample in data:
        if sample['sol_id'] in train_questions and sample['sol_id'] not in train_has:
            train.append(sample)
            train_has.append(sample['sol_id'])
        if sample['sol_id'] in val_questions and sample['sol_id'] not in val_has:
            val.append(sample)
            val_has.append(sample['sol_id'])
        
    return train, val