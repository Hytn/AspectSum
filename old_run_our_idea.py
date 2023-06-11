import json
import nltk
import torch
import logging
import random
import argparse
import numpy as np
from model import aspect_summarizaiton_model
from rouge import Rouge
from tqdm import tqdm
from datasets import load_metric
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.dataloader import MReDDataset
from utils.logger import set_logger
import os
# from sumeval.metrics.bleu import BLEUCalculator



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
    parser.add_argument("--detector_num", default=1, type=int)                   # 1 means only decoder detector, 2 means detector with encoder and decoder
    parser.add_argument("--use_distribution_type", default=1, type=int)      # two calculation methods 
    parser.add_argument("--use_input_doc_type", default=1, type=int)      # If the type is 1, the doc input in the test set has aspect-ctrl; if the type is 0, the doc input in the test set is aspect-unctrl
    parser.add_argument("--use_output_ctrl_type", default=1, type=int)      # If the type is 1, it is sent-control; if the type is 2, it is seg-control. This field affects the selection of output fields
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--log_dir", default="./our_results", type=str)
    parser.add_argument("--aspect_weight", default=0, type=float)
    parser.add_argument("--pretrained_model_path", default="./results/concat_1024_sent-ctrl", type=str)
    parser.add_argument("--max_source_length", default=1024, type=int)

    args = parser.parse_args()

    dir=os.path.join(args.log_dir, 'detectorNum_'+str(args.detector_num)+'_distributionType_'+str(args.use_distribution_type)+'_docType_'+str(args.use_input_doc_type)+'_ctrlType_'+str(args.use_output_ctrl_type)+'_aspectWeight_'+str(args.aspect_weight))
    if not os.path.exists(dir):
        os.mkdir(dir)
    log_file=os.path.join(dir,str(dir)+'.log')

    setup_seed(args.seed)

    metric = load_metric("./utils/rouge.py")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = MReDDataset(args.train_file)
    val_dataset = MReDDataset(args.val_file)
    test_dataset = MReDDataset(args.test_file)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn', use_fast=True)
    model = aspect_summarizaiton_model(pretrained_model_path=args.pretrained_model_path).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)

    set_logger(save=True, log_path=args.log_file)

    train_loss = []
    val_rouge_avg = []
    rouge = Rouge()
    logging.info('Start train...')
    logging.info('The weight of aspect is'+str(args.aspect_weight))

    for epoch in range(args.epoch_num):
        logging.info("Epoch {}/{}".format(epoch+1, args.epoch_num))
        train_epoch_loss = []
        torch.cuda.empty_cache()
        
        model.train()
        for X in tqdm(train_dataloader):
            train_features = [a[0] for a in X]
            summary_with_sent_aspect = [a[1] for a in X]
            train_labels = [a[2] for a in X]
            aspect_id = [a[3] for a in X]
            lenofaspect = []
            idofasp = []
            for i in range(len(summary_with_sent_aspect[0])):
                temp = tokenizer(summary_with_sent_aspect[0][i][1], max_length=args.max_source_length, padding=False, truncation=True,
                                 return_tensors="pt")["input_ids"].to(device)
                lenofaspect.append(temp.shape[1])
                if i == 0:
                    idofasp.append(lenofaspect[i] - 2)
                else:
                    idofasp.append(idofasp[i - 1] + lenofaspect[i] - 2)
            idofasp = torch.unsqueeze(torch.tensor(idofasp), 1).to(device)
            # leninsum=torch.tensor(lenofaspect).to(device)
            in1 = tokenizer(train_features, max_length=args.max_source_length, padding=False, truncation=True, return_tensors="pt").to(device)
            in2 = tokenizer(train_labels, return_tensors="pt", padding=False).to(device)
            r = model(inputs=in1, labels=in2, idofasp=idofasp, aspect_id=aspect_id, aspect_weight=args.aspect_weight)

            loss = r[4]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
        logging.info("train loss:{}".format(sum(train_epoch_loss)))
        train_loss.append(sum(train_epoch_loss))
        # test
        model.eval()
        list = []
        test_rouge = {
            "rouge-1": 0,
            "rouge-2": 0,
            "rouge-L": 0
        }
        preds=[]
        labels=[]
        with torch.no_grad():
            for Y in tqdm(test_dataloader):
                dict = {}
                train_features = Y[0][0]
                train_labels = Y[0][2]
                features = tokenizer(train_features, max_length=args.max_source_length, padding=False, truncation=True,
                                     return_tensors="pt").to(device)
                summary_id = tokenizer(train_labels, max_length=args.max_source_length, padding=False, truncation=True,
                                       return_tensors="pt").to(device)
                generate_ids = model.generate(features["input_ids"], min_length=20, num_beams=4, no_repeat_ngram_size=3,
                                              max_length=400, early_stopping=True)
                presum = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
                preds.append(generate_ids.cpu().numpy().flatten().tolist())
                labels.append(summary_id["input_ids"].cpu().numpy().flatten().tolist())
                p,r,f = compute_metrics(generate_ids, summary_id["input_ids"])      

                dict['article'] = train_features
                dict['summary'] = train_labels
                dict['presum'] = presum[0]
                dict['scores_p'] = p
                dict['scores_r'] = r
                dict['scores_f'] = f
                list.append(dict)

                # test_rouge['rouge-1'] += f['rouge1_f']
                # test_rouge['rouge-2'] += f['rouge2_f']
                # test_rouge['rouge-L'] += f['rougeL_f']
            
        # test_rouge['rouge-1'] /= 674
        # test_rouge['rouge-2'] /= 674
        # test_rouge['rouge-L'] /= 674
        # dict = {}
        # dict["R1"] = test_rouge['rouge-1']
        # dict["R2"] = test_rouge['rouge-2']
        # dict["RL"] = test_rouge['rouge-L']
        # list.append(dict)

        with open("{}-result-{}.json".format(args.aspect_weight,epoch), 'w') as fp:
            json.dump(list, fp)

        # rouge指标
        p_rouge, r_rouge, f_rouge = compute_metrics(preds,labels)
        logging.info("Test results::R1-p:{},R2-p:{},Rl-p:{}".format(p_rouge['rouge1_p'], p_rouge['rouge2_p'], p_rouge['rougeL_p']))
        logging.info("Test results::R1-r:{},R2-r:{},Rl-r:{}".format(r_rouge['rouge1_r'], r_rouge['rouge2_r'], r_rouge['rougeL_r']))
        logging.info("Test results::R1-f:{},R2-f:{},Rl-f:{}".format(f_rouge['rouge1_f'], f_rouge['rouge2_f'], f_rouge['rougeL_f']))