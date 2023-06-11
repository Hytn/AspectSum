import logging
import os
import sys
from termios import FF1
sys.path.append('/home/ysc/MReD')
# print(sys.path)
from dataclasses import dataclass, field
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, load_metric

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

from transformers.tokenization_utils_base import BatchEncoding


if __name__ == '__main__':
    model=AutoModelForSeq2SeqLM.from_pretrained('/home/spark/zhanghan/aspect_mred/results/concat_2048_sent-ctrl/checkpoint-16000/')

    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn', use_fast=True, local_files_only = True)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer
    )
    test_dataset = load_dataset("csv",data_files='/home/spark/zhanghan/aspect_mred/raw_data/test_concat.csv')

    trainer._load_from_checkpoint('/home/spark/zhanghan/aspect_mred/results/concat_2048_sent-ctrl/checkpoint-16000')
    predict_results = trainer.predict(
            test_dataset,
            metric_key_prefix="predict",
            max_length=400,
            num_beams=4,
        )
    


