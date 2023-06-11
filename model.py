import torch
from torch import nn
from typing import Callable, Iterable, List, Optional
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

class aspect_summarizaiton_model(nn.Module):
    def __init__(self,pretrained_model_path):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn', use_fast=True)
        self.bart = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=pretrained_model_path)
        self.bart.config.output_hidden_states = True
        self.decoder_aspect_token_detector = nn.Linear(1024, 9, bias=True)
        self.encoder_aspect_token_detector = nn.Linear(1024, 9, bias=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,
                inputs,
                labels,
                encoder_aspect_id,
                decoder_aspect_last_pos,
                decoder_aspect_id,
                aspect_weight,
                use_distribution_type,
                detector_num,
                use_consist_loss,
                **kwargs
                ):
        result = self.bart(input_ids=inputs['input_ids'], labels=labels['input_ids'])

        # Calculate the distribution of the aspect of the decoder
        decoder_aspect = torch.squeeze(self.decoder_aspect_token_detector(result.decoder_hidden_states[12]))
        deocder_aspect_label = torch.zeros(len(labels['input_ids'][0]), 9).to(self.device)
        assert(decoder_aspect_last_pos.size(dim=0) == len(decoder_aspect_id[0]))
        for i in range(len(decoder_aspect_last_pos)):
            if i==0:
                deocder_aspect_label[1:decoder_aspect_last_pos[i]+1, decoder_aspect_id[0][i]] = 1 / decoder_aspect_last_pos[i]
            else:
                deocder_aspect_label[decoder_aspect_last_pos[i-1]+1:decoder_aspect_last_pos[i]+1, decoder_aspect_id[0][i]] = 1 / (decoder_aspect_last_pos[i]-decoder_aspect_last_pos[i-1])
        if use_distribution_type==2:
            for i in range(1,len(deocder_aspect_label)-1):
                for j in range(9):
                    deocder_aspect_label[i][j]=min(1,deocder_aspect_label[i-1][j]+deocder_aspect_label[i][j])

        loss_mse = nn.MSELoss()
        loss = result.loss + aspect_weight * loss_mse(decoder_aspect, deocder_aspect_label)

        # Calculate the distribution of the aspect of the encoder
        if detector_num==2:
            encoder_aspect = torch.squeeze(self.encoder_aspect_token_detector(result.encoder_hidden_states[12]))
            encoder_aspect_label = torch.zeros(1, 9).to(self.device)
            for i in encoder_aspect_id[0]:
                encoder_aspect_label[0,i]=1
            if use_distribution_type==1:
                encoder_aspect=encoder_aspect.mean(dim=0).view(1,-1)    # 按行取平均
                loss+=aspect_weight*loss_mse(encoder_aspect,encoder_aspect_label)
            elif use_distribution_type==2:
                encoder_aspect=encoder_aspect[-1:]  # 取最后一行
                loss += aspect_weight * loss_mse(encoder_aspect, encoder_aspect_label)
        if use_consist_loss==1:
            if use_distribution_type==1:
                decoder_aspect=decoder_aspect.mean(dim=0).view(1,-1)
            elif use_distribution_type==2:
                decoder_aspect=decoder_aspect[-1:]
            loss+=aspect_weight*loss_mse(encoder_aspect,decoder_aspect)
        return loss

    def generate(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            max_length: Optional[int] = None,
            min_length: Optional[int] = None,
            do_sample: Optional[bool] = None,
            early_stopping: Optional[bool] = None,
            num_beams: Optional[int] = None,
            temperature: Optional[float] = None,
            top_k: Optional[int] = None,
            top_p: Optional[float] = None,
            repetition_penalty: Optional[float] = None,
            bad_words_ids: Optional[Iterable[int]] = None,
            bos_token_id: Optional[int] = None,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            length_penalty: Optional[float] = None,
            no_repeat_ngram_size: Optional[int] = None,
            encoder_no_repeat_ngram_size: Optional[int] = None,
            num_return_sequences: Optional[int] = None,
            max_time: Optional[float] = None,
            decoder_start_token_id: Optional[int] = None,
            use_cache: Optional[bool] = None,
            num_beam_groups: Optional[int] = None,
            diversity_penalty: Optional[float] = None,
            prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_scores: Optional[bool] = None,
            return_dict_in_generate: Optional[bool] = None,
            forced_bos_token_id: Optional[int] = None,
            forced_eos_token_id: Optional[int] = None,
            remove_invalid_values: Optional[bool] = None,
            synced_gpus: Optional[bool] = None,
            **model_kwargs,
    ):
        return self.bart.generate(input_ids=input_ids,
                                   max_length=max_length,
                                   min_length=min_length,
                                   do_sample=do_sample,
                                   early_stopping=early_stopping,
                                   num_beams=num_beams,
                                   temperature=temperature,
                                   top_k=top_k,
                                   top_p=top_p,
                                   repetition_penalty=repetition_penalty,
                                   bad_words_ids=bad_words_ids,
                                   bos_token_id=bos_token_id,
                                   pad_token_id=pad_token_id,
                                   eos_token_id=eos_token_id,
                                   length_penalty=length_penalty,
                                   no_repeat_ngram_size=no_repeat_ngram_size,
                                   encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
                                   num_return_sequences=num_return_sequences,
                                   max_time=max_time,
                                   decoder_start_token_id=decoder_start_token_id,
                                   use_cache=use_cache,
                                   num_beam_groups=num_beam_groups,
                                   diversity_penalty=diversity_penalty,
                                   prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   output_scores=output_scores,
                                   return_dict_in_generate=return_dict_in_generate,
                                   forced_bos_token_id=forced_bos_token_id,
                                   forced_eos_token_id=forced_eos_token_id,
                                   remove_invalid_values=remove_invalid_values,
                                   synced_gpus=synced_gpus,
                                   **model_kwargs)