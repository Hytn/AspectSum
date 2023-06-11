import json
import nltk
import torch
import logging
import random
import argparse
import numpy as np
from model import aspect_summarizaiton_model
from tqdm import tqdm
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.dataloader import MReDDataset
from utils.logger import set_logger
import os



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


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


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch_num", default=1, type=int)
    parser.add_argument("--train_file", default="./processed_data/train.json", type=str)
    parser.add_argument("--val_file", default="./processed_data/val.json", type=str)
    parser.add_argument("--test_file", default="./processed_data/test.json", type=str)
    parser.add_argument("--detector_num", default=2, type=int)                   # 1 means only decoder detector, 2 means detector with encoder and decoder
    parser.add_argument("--use_distribution_type", default=2, type=int)      # two calculation methods   
    parser.add_argument("--whether_input_ctrl", default=1, type=int)      # If the type is 1, the doc input in the test set has aspect-ctrl; if the type is 0, the doc input in the test set is aspect-unctrl
    parser.add_argument("--sent_or_seg_ctrl", default=1, type=int)      # If the type is 1, it is sent-control; if the type is 2, it is seg-control. This field affects the selection of output fields
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--log_dir", default="./our_results", type=str)
    parser.add_argument("--aspect_weight", default=0.5, type=float)
    parser.add_argument("--pretrained_model_path", default="./results/concat_2048_sent-ctrl", type=str)
    parser.add_argument("--max_source_length", default=2048, type=int)
    parser.add_argument("--max_target_length", default=400, type=int)
    parser.add_argument("--use_consist_loss", default=0, type=int)

    args = parser.parse_args()

    name='detectorNum_'+str(args.detector_num)+'_distributionType_'+str(args.use_distribution_type)+'_docType_'+str(args.whether_input_ctrl)+'_ctrlType_'+str(args.sent_or_seg_ctrl)+'_aspectWeight_'+str(args.aspect_weight)
    dir=os.path.join(args.log_dir, name)
    if not os.path.exists(dir):
        os.mkdir(dir)
    log_file=os.path.join(dir,name+'.log')

    setup_seed(args.seed)

    metric = load_metric("./utils/rouge.py")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = MReDDataset(args.train_file)
    val_dataset = MReDDataset(args.val_file)
    test_dataset = MReDDataset(args.test_file)

    # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn', use_fast=True)
    model = aspect_summarizaiton_model(pretrained_model_path=args.pretrained_model_path).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    set_logger(save=True, log_path=log_file)

    logging.info('start training')
    logging.info('The weight of aspect is'+str(args.aspect_weight))

    for epoch in range(args.epoch_num):
        logging.info("Epoch {}/{}".format(epoch+1, args.epoch_num))
        train_epoch_loss = 0   # The total loss used to accumulate the training process
        torch.cuda.empty_cache()
        
        model.train()
        for sample in tqdm(train_dataloader):
            if args.sent_or_seg_ctrl==1:
                doc=[i["sent_controlled_doc"] for i in sample]  # The input doc documents and the input doc documents of the training set all have aspect
            else:
                doc=[i["seg_controlled_doc"] for i in sample]
            doc_with_sent_aspect = [i["doc_with_sent_aspect"] for i in sample]
            if args.sent_or_seg_ctrl==1:
                summary_with_aspect = [i["summary_with_sent_aspect"] for i in sample]    # summary with sent-aspect tag
            else:
                summary_with_aspect=[i["summary_with_seg_aspect"] for i in sample]
            summary = [i["summary"] for i in sample]    
            if args.sent_or_seg_ctrl==1:
                decoder_aspect_id=[i["sent_ap_id"] for i in sample]
            else:
                decoder_aspect_id=[i["seg_ap_id"] for i in sample]
            encoder_aspect_id=[i["doc_ap_id"] for i in sample]
        
            # Calculate the correspondence of the summary aspect
            decoder_aspect_last_pos=[]  # The subscript of the last token of the sentence corresponding to each aspect
            for i in range(len(summary_with_aspect[0])):    # only for batch_size = 1
                summary_to_ids = tokenizer(summary_with_aspect[0][i][1], max_length=args.max_target_length, padding=False, truncation=True,
                                 return_tensors="pt")["input_ids"].to(device)
                length=summary_to_ids.shape[1]
                if i==0:
                    decoder_aspect_last_pos.append(length-2)    # To subtract the start and end characters before and after
                else:
                    decoder_aspect_last_pos.append(decoder_aspect_last_pos[i-1] + length - 2)
            decoder_aspect_last_pos = torch.unsqueeze(torch.tensor(decoder_aspect_last_pos), 1).to(device)  

            inputs = tokenizer(doc, max_length=args.max_source_length, padding=False, truncation=True, return_tensors="pt").to(device)
            labels = tokenizer(summary, return_tensors="pt", padding=False).to(device)

            loss = model(inputs=inputs, labels=labels, encoder_aspect_id=encoder_aspect_id, decoder_aspect_last_pos=decoder_aspect_last_pos, decoder_aspect_id=decoder_aspect_id, aspect_weight=args.aspect_weight,use_distribution_type=args.use_distribution_type,detector_num=args.detector_num,use_consist_loss=args.use_consist_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss+=loss.item()
        logging.info("train loss:{}".format(train_epoch_loss))

        # save model
        torch.save(model,os.path.join(dir,'{}.pt'.format(epoch)))

        model.eval()

        # Evaluating Models
        preds=[]
        labels=[]
        with torch.no_grad():
            for sample in tqdm(val_dataloader):
                dict = {}
                if args.whether_input_ctrl==1:
                    if args.sent_or_seg_ctrl==1:
                        doc=[i["sent_controlled_doc"] for i in sample]
                    else:
                        doc=[i["seg_controlled_doc"] for i in sample]
                else:   
                    doc=[i["doc"] for i in sample]
                summary=[i["summary"] for i in sample]
                doc_id = tokenizer(doc, max_length=args.max_source_length, padding=False, truncation=True,
                                     return_tensors="pt").to(device)
                generate_ids = model.generate(doc_id["input_ids"], min_length=20, num_beams=4, no_repeat_ngram_size=3,
                                              max_length=400, early_stopping=True)
                summary_id = tokenizer(summary, max_length=args.max_source_length, padding=False, truncation=True,
                                       return_tensors="pt").to(device)
                
                preds.append(generate_ids.cpu().numpy().flatten().tolist())
                labels.append(summary_id["input_ids"].cpu().numpy().flatten().tolist())

        p_rouge, r_rouge, f_rouge = compute_metrics(preds,labels)
        logging.info("Validation results:R1-p:{},R2-p:{},Rl-p:{}".format(p_rouge['rouge1_p'], p_rouge['rouge2_p'], p_rouge['rougeL_p']))
        logging.info("Validation results:R1-r:{},R2-r:{},Rl-r:{}".format(r_rouge['rouge1_r'], r_rouge['rouge2_r'], r_rouge['rougeL_r']))
        logging.info("Validation results:R1-f:{},R2-f:{},Rl-f:{}".format(f_rouge['rouge1_f'], f_rouge['rouge2_f'], f_rouge['rougeL_f']))

        # Testing Models
        preds=[]
        labels=[]
        with torch.no_grad():
            for sample in tqdm(test_dataloader):
                dict = {}
                if args.whether_input_ctrl==1:
                    if args.sent_or_seg_ctrl==1:
                        doc=[i["sent_controlled_doc"] for i in sample]
                    else:
                        doc=[i["seg_controlled_doc"] for i in sample]
                else:   
                    doc=[i["doc"] for i in sample]
                summary=[i["summary"] for i in sample]
                # doc=sample[0]["sent_controlled_doc"]
                # summary=sample[0]["summary"]
                doc_id = tokenizer(doc, max_length=args.max_source_length, padding=False, truncation=True,
                                     return_tensors="pt").to(device)
                generate_ids = model.generate(doc_id["input_ids"], min_length=20, num_beams=4, no_repeat_ngram_size=3,
                                              max_length=400, early_stopping=True)
                summary_id = tokenizer(summary, max_length=args.max_source_length, padding=False, truncation=True,
                                       return_tensors="pt").to(device)
                
                preds.append(generate_ids.cpu().numpy().flatten().tolist())
                labels.append(summary_id["input_ids"].cpu().numpy().flatten().tolist())

        # rouge
        p_rouge, r_rouge, f_rouge = compute_metrics(preds,labels)
        logging.info("Test results:R1-p:{},R2-p:{},Rl-p:{}".format(p_rouge['rouge1_p'], p_rouge['rouge2_p'], p_rouge['rougeL_p']))
        logging.info("Test results R1-r:{},R2-r:{},Rl-r:{}".format(r_rouge['rouge1_r'], r_rouge['rouge2_r'], r_rouge['rougeL_r']))
        logging.info("Test results:R1-f:{},R2-f:{},Rl-f:{}".format(f_rouge['rouge1_f'], f_rouge['rouge2_f'], f_rouge['rougeL_f']))