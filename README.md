# AspectSum
Implementation of An Aspect-Guided Joint Learning Generator for Recovering Aspect Information in Abstractive Multi-Document Summarization

## Required Packages
* Python (tested on 3.8.13)
* CUDA (tested on 11.4)
* [PyTorch](http://pytorch.org/) (tested on 1.8.0)
* [Transformers](https://github.com/huggingface/transformers) (tested on 4.6.0)
* numpy (tested on 1.23.2)
* tqdm

## Dataset
The [MRED](https://arxiv.org/abs/2110.07474) dataset can be downloaded following the instructions at https://github.com/Shen-Chenhui/MReD
The [WikiAsp](https://arxiv.org/abs/2011.07832) dataset can be downloaded following the instructions at https://github.com/neulab/wikiasp


## Process data
First, you should prepare the data and place them in the `raw_data`. Next process the data 
```bash
python 1_process_data.py
```
The processed files are in `processed_data`
The following are descriptions of some fields
* **doc**: original document
* **doc_with_sent_aspect**: aspect with each sentence
* **sent_controlled_doc**: `[label1, label2, ... ]`, per-sentence label sequence for the meta-review     where `label1` represents the category label for 1st sentence, `label2` for the 2nd sentence and so on
* **seg_controlled_doc**: `[label1, label2, ... ]`, label sequence for the meta-review on segment level where `label1` represents the category label for 1st segment (the sentences of the same label), `label2` for the 2nd segment and so on
* **summary**: Summary of the document
* **summary_with_seg_aspect**: segment level summaries
* **summary_with_sent_aspect**: Sentence level summaries
* **sample_id**: `yyyy-id`, where `yyyy` is the year

## Training and Evaluation
The training and verification codes are in `3_run_our_idea.py`, you can change the `pretrained_model_path` to replace different pre-trained models, and compare the learning effects of different models.

## case study
If you want to see the effect of different aspect learning, you can run `case_study.py`

## Cite Us