# AspectSum
The source code of the paper [SALAS: Supervised Aspect Learning Improves Abstractive Multi-document Summarization Through Aspect Information Loss](https://link.springer.com/chapter/10.1007/978-3-031-43421-1_4).

## Requirements
* Python (tested on 3.8.13)
* CUDA (tested on 11.4)
* [PyTorch](http://pytorch.org/) (tested on 1.8.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.6.0)
* numpy (tested on 1.23.2)
* tqdm

## Datasets
The [MRED](https://arxiv.org/abs/2110.07474) dataset can be downloaded from https://github.com/Shen-Chenhui/MReD
The [WikiAsp](https://arxiv.org/abs/2011.07832) dataset can be downloaded from https://github.com/neulab/wikiasp


## Data Processing
First, place the data in `raw_data`. Next, processing the data by
```bash
python 1_process_data.py
```
The processed files are stored in `processed_data`.
Here are some descriptions.
* **doc**: original document
* **doc_with_sent_aspect**: aspects with each sentence
* **sent_controlled_doc**: `[label1, label2, ... ]`, per-sentence label sequence for the meta-review     where `label1` represents the category label for 1st sentence, `label2` for the 2nd sentence and so on
* **seg_controlled_doc**: `[label1, label2, ... ]`, label sequence for the meta-review on segment level where `label1` represents the category label for 1st segment (the sentences of the same label), `label2` for the 2nd segment and so on
* **summary**: Summary of the document
* **summary_with_seg_aspect**: segment level summaries
* **summary_with_sent_aspect**: Sentence level summaries
* **sample_id**: `yyyy-id`, where `yyyy` is the year

## Training and Evaluation
The training and evaluation are executed by `3_run_our_idea.py`, you can replace the `pretrained_model_path` to use different pre-trained models.

## Case Study
Our implementation of the case study is in `case_study.py`.
