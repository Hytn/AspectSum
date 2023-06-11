from ast import Or
import os
import shutil
from tqdm import tqdm
import csv
import json
from collections import OrderedDict

raw_data_root_dir='./raw_data'
processed_data_root_dir='./processed_data'
split=['train','val','test']

train_id=[]
val_id=[]
test_id=[]

def process_map_to_json(map_data):
    data=[]
    for key in tqdm(map_data):
        map_data[key]["sample_id"]=key
        data.append(map_data[key])
    return data



def process_txt_to_json(spl,src,tgt):
    data=OrderedDict()
    sample={}
    sample['doc_with_sent_aspect']=[]
    sample['doc']=''
    sample['sent_controlled_doc']=''
    sample['seg_controlled_doc']=''
    sample['summary_with_sent_aspect']=[]
    sample['summary_with_seg_aspect']=[]
    sample['summary']=''
    sent_ctrl_aspect=''
    seg_ctrl_aspect=''

    f_src=open(src,'r',encoding='utf-8')
    f_tgt=open(tgt,'r',encoding='utf-8')
    fout=open(os.path.join(processed_data_root_dir,spl+'.json'),'w')

    for line in tqdm(f_src.readlines()):
        line = line.rstrip()
        if line!='':
            ls=line.split('\t')
            sample_id=ls[0]
            sample['doc_with_sent_aspect'].append([ls[2],ls[1]])
            if sample['doc']=='':
                sample['doc']+=ls[1]
            else:
                sample['doc']+=' '+ls[1]
        else:   # hit a blank line
            if sample_id in eval(spl+'_id'):
                data[sample_id]=sample
                sample={}
                sample['doc_with_sent_aspect']=[]
                sample['doc']=''
                sample['sent_controlled_doc']=''
                sample['seg_controlled_doc']=''
                sample['summary_with_sent_aspect']=[]
                sample['summary_with_seg_aspect']=[]
                sample['summary']=''
    
    summary_with_seg_aspect=OrderedDict()
    for line in tqdm(f_tgt.readlines()):
        line = line.rstrip()
        if line!='':
            ls=line.split('\t')
            sample_id=ls[0]
            if sample_id not in eval(spl+'_id'):
                continue
            ls=line.split('\t')
            sample_id=ls[0]
            data[sample_id]['summary_with_sent_aspect'].append([ls[2],ls[1]])
            if ls[2] not in summary_with_seg_aspect:
                summary_with_seg_aspect[ls[2]]=ls[1]
            else:
                summary_with_seg_aspect[ls[2]]+=' '+ls[1]
            if data[sample_id]['summary']=='':
                data[sample_id]['summary']+=ls[1]
            else:
                data[sample_id]['summary']+=' '+ls[1]
            
        else:   # hit a blank line
            if sample_id in eval(spl+'_id'):
                sent_str=[]
                seg_str=[]
                for i in data[sample_id]['summary_with_sent_aspect']:
                    sent_str.append(i[0])
                for key in summary_with_seg_aspect:
                    data[sample_id]['summary_with_seg_aspect'].append([key,summary_with_seg_aspect[key]])
                    seg_str.append(key)

                data[sample_id]['sent_controlled_doc']+=' | '.join(sent_str)+' ==> '+data[sample_id]['doc']
                data[sample_id]['seg_controlled_doc']+=' | '.join(seg_str)+' ==> '+data[sample_id]['doc']

                summary_with_seg_aspect=OrderedDict()
                sample={}
                sample['doc_with_sent_aspect']=[]
                sample['doc']=''
                sample['sent_controlled_doc']=''
                sample['seg_controlled_doc']=''
                sample['summary_with_sent_aspect']=[]
                sample['summary_with_seg_aspect']=[]
                sample['summary']=''
    
    data=process_map_to_json(data)
    json.dump(data,fout)
    f_src.close()
    f_tgt.close()
    fout.close()



if __name__=="__main__":
    # Get the id of the sample to be kept
    f=open(os.path.join(raw_data_root_dir,'full_data_info.txt'),'r',encoding='utf-8')
    for line in tqdm(f.readlines()):
        line = line.rstrip().split('\t')
        if 20<=int(line[9])<=400:
            if line[0]=='val':
                val_id.append(line[1])
            elif line[0]=='test':
                test_id.append(line[1])
            elif line[0]=='train':
                train_id.append(line[1])

    for spl in split:
        src=os.path.join(raw_data_root_dir,spl+'_src.txt')
        tgt=os.path.join(raw_data_root_dir,spl+'_tgt.txt')
        process_txt_to_json(spl,src,tgt)
