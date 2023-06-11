import json
from turtle import shape
import nltk
import torch
import logging
import random
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.dataloader import MReDDataset
from utils.logger import set_logger
from torch import nn
from typing import Callable, Iterable, List, Optional
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BartTokenizer,
    BartModel,
    BartConfig
)
import matplotlib.pyplot as plt
import os

aspect_to_id={
    'history':0,
    'description':1,
    'description and history': 2,
    'heritage listing':3,
    'architecture':4,
    'location':5,
    'historic uses':6,
    'preservation':7,
    'geography':8,
    'interior':9
}

def fun(m):
    print(m.argmax(axis=0))

def sharpen(arr,T):
    T=1/T
    sum_arr=np.sum(np.power(arr,T),axis=1,keepdims=True)
    return arr/sum_arr



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained_model", default='/data/aspect_mred/our_results/detectorNum_1_distributionType_2_docType_1_ctrlType_1_aspectWeight_1.0/1.pt', type=str)
    parser.add_argument("--raw_model", default='', type=str)
    parser.add_argument("--test_file", default='/data/aspect_mred/processed_data/test.json', type=str)
    parser.add_argument("--max_source_length", default=2048, type=int)
    parser.add_argument("--max_target_length", default=400, type=int)

    args = parser.parse_args()

    metric = load_metric("./utils/rouge.py")

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    test_dataset = MReDDataset(args.test_file)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    config = AutoConfig.from_pretrained(
            "facebook/bart-large-cnn"
        )
    config.max_position_embeddings=2048
    model = torch.load(args.trained_model).to(device)
    bart=model.bart
    decoder_aspect_token_detector=model.decoder_aspect_token_detector
    # encoder_aspect_token_detector=model.encoder_aspect_token_detector
    tokenizer=model.tokenizer

    # trained_model.eval()
    with torch.no_grad():
        for sample in tqdm(test_dataloader):
            # print(sample)
            doc=[i["doc"] for i in sample]
            summary = [i["summary"] for i in sample]    
            summary_with_aspect = [i["summary_with_seg_aspect"] for i in sample]

            decoder_aspect_last_pos = []  
            for i in range(len(summary_with_aspect[
                                   0])): 
                summary_to_ids = \
                tokenizer(summary_with_aspect[0][i][1], max_length=args.max_target_length, padding=False,
                          truncation=True, return_tensors="pt")["input_ids"].to(device)
                length = summary_to_ids.shape[1]
                if i == 0:
                    decoder_aspect_last_pos.append(length - 2)  
                else:
                    decoder_aspect_last_pos.append(decoder_aspect_last_pos[i - 1] + length - 2)
            decoder_aspect_last_pos = torch.unsqueeze(torch.tensor(decoder_aspect_last_pos), 1).to(device)  
            # print("\n")
            # print(decoder_aspect_last_pos)
            # print("aspect_id:",sample[0]['seg_ap_id'])

            inputs = tokenizer(doc, max_length=args.max_source_length, padding=False, truncation=True, return_tensors="pt").to(device)
            labels = tokenizer(summary, return_tensors="pt", max_length=args.max_target_length, padding=False, truncation=True).to(device)

            result = bart(input_ids=inputs['input_ids'], labels=labels['input_ids'])

            decoder_aspect = torch.squeeze(decoder_aspect_token_detector(result.decoder_hidden_states[12]))
            decoder_aspect= decoder_aspect.cpu().numpy().transpose()

            
            for i in range(1,10):
                j=i/10
                # os.mkdir("./allpic/fig"+str(j))
                m=sharpen(decoder_aspect,j)
                plt.matshow(m, cmap=plt.get_cmap('Greens'), alpha=0.1)  # , alpha=0.3
                plt.title(sample[0]["sample_id"])
                name="./allpic/fig"+str(j)+"/"+sample[0]["sample_id"]+".png"
                plt.savefig(name)
                plt.show()