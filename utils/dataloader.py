import torch
import json
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import BartTokenizer

aspect_to_id = {'abstract': 0, 'strength': 1, 'rebuttal_process': 2,
                'decision': 3, 'weakness': 4, 'misc': 5,
                'rating_summary': 6, 'suggestion': 7, 'ac_disagreement': 8}


class MReDDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data=self.data[idx]

        data["sent_ap_id"]=[aspect_to_id[a[0]] for a in data["summary_with_sent_aspect"]]
        data["seg_ap_id"]=[aspect_to_id[a[0]] for a in data["summary_with_seg_aspect"]]
        data["doc_ap_id"]=[aspect_to_id[a[0]] for a in data["doc_with_sent_aspect"]]

        return data
