import json
import tokenize
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
    BartConfig,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import matplotlib.pyplot as plt
import os

ghj_model_list_detectorNum_1={
    0.1:"2.pt",
    0.2:"2.pt",
    0.3:"2.pt",
    0.4:"2.pt",
    0.5:"2.pt",
    0.6:"2.pt",
    0.7:"2.pt",
    0.8:"2.pt",
    0.9:"2.pt",
    1.0:"2.pt"
}

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

def compute_metrics(preds, labels):
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    p_result = {key+'_p': value.mid.precision * 100 for key, value in result.items()}
    r_result = {key+'_r': value.mid.recall * 100 for key, value in result.items()}
    f_result = {key+'_f': value.mid.fmeasure * 100 for key, value in result.items()}
    # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    # result["gen_len"] = np.mean(prediction_lens)
    p_result = {k: round(v, 4) for k, v in p_result.items()}
    r_result = {k: round(v, 4) for k, v in r_result.items()}
    f_result = {k: round(v, 4) for k, v in f_result.items()}
    return p_result, r_result, f_result




if __name__=="__main__":
    log_file=os.path.join("./unctrl_results","ghj_model_list_detectorNum_2"+'.log')
    set_logger(save=True, log_path=log_file)
    
    
    test_dataset = MReDDataset("/home/spark/zhanghan/aspect_mred/processed_data/test.json")
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn', use_fast=True, local_files_only = True)

    # model='/home/spark/zhanghan/aspect_mred/results/concat_2048_sent-ctrl/checkpoint-16000/optimizer.pt'
    
    # model = torch.load(model).to(device)
    # # bart=model.bart
    
    # # tokenizer = AutoTokenizer.from_pretrained('./bart.large.cnn/', use_fast=True)
    # preds=[]
    # labels=[]
    # with torch.no_grad():
    #     for sample in tqdm(test_dataloader):
    #         doc=[i["doc"] for i in sample]
    #         summary=[i["summary"] for i in sample]

    #         doc_id = tokenizer(doc, max_length=2048, padding=False, truncation=True,
    #                                 return_tensors="pt").to(device)
    #         generate_ids = model.generate(doc_id["input_ids"], min_length=20, num_beams=4, no_repeat_ngram_size=3,
    #                                         max_length=400, early_stopping=True)
        
    #         summary_id = tokenizer(summary, max_length=400, padding=False, truncation=True,
    #                                 return_tensors="pt").to(device)
            
    #         preds.append(generate_ids.cpu().numpy().flatten().tolist())
    #         labels.append(summary_id["input_ids"].cpu().numpy().flatten().tolist())
    # p_rouge, r_rouge, f_rouge = compute_metrics(preds,labels)
    # logging.info("************************")
    # logging.info("Native bart的结果：")
    # logging.info("test result :R1-p:{},R2-p:{},Rl-p:{}".format(p_rouge['rouge1_p'], p_rouge['rouge2_p'], p_rouge['rougeL_p']))
    # logging.info("test result :R1-r:{},R2-r:{},Rl-r:{}".format(r_rouge['rouge1_r'], r_rouge['rouge2_r'], r_rouge['rougeL_r']))
    # logging.info("test result :R1-f:{},R2-f:{},Rl-f:{}".format(f_rouge['rouge1_f'], f_rouge['rouge2_f'], f_rouge['rougeL_f']))



    for key in ghj_model_list_detectorNum_1:
        model='./our_results/detectorNum_1_distributionType_1_docType_1_ctrlType_1_aspectWeight_'+str(key)+'/'+str(ghj_model_list_detectorNum_1[key])
        metric = load_metric("./utils/rouge.py")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = AutoConfig.from_pretrained(
            "facebook/bart-large-cnn",local_files_only = True
        )

        config.max_position_embeddings=2048
        model = torch.load(model).to(device)
        bart=model.bart
        

        preds=[]
        labels=[]
        with torch.no_grad():
            for sample in tqdm(test_dataloader):
                doc=[i["doc"] for i in sample]
                summary=[i["summary"] for i in sample]

                doc_id = tokenizer(doc, max_length=2048, padding=False, truncation=True,
                                     return_tensors="pt").to(device)
                generate_ids = model.generate(doc_id["input_ids"], min_length=20, num_beams=4, no_repeat_ngram_size=3,
                                              max_length=400, early_stopping=True)
                summary_id = tokenizer(summary, max_length=400, padding=False, truncation=True,
                                       return_tensors="pt").to(device)
                
                preds.append(generate_ids.cpu().numpy().flatten().tolist())
                labels.append(summary_id["input_ids"].cpu().numpy().flatten().tolist())
        p_rouge, r_rouge, f_rouge = compute_metrics(preds,labels)
        logging.info("************************")
        logging.info(str(key))
        logging.info("测试结果:R1-p:{},R2-p:{},Rl-p:{}".format(p_rouge['rouge1_p'], p_rouge['rouge2_p'], p_rouge['rougeL_p']))
        logging.info("测试结果:R1-r:{},R2-r:{},Rl-r:{}".format(r_rouge['rouge1_r'], r_rouge['rouge2_r'], r_rouge['rougeL_r']))
        logging.info("测试结果F:R1-f:{},R2-f:{},Rl-f:{}".format(f_rouge['rouge1_f'], f_rouge['rouge2_f'], f_rouge['rougeL_f']))