'''
model with provided pre-generated table embeddings
need a db_path in config and a db_id in each input to retrieve the table embeddings
'''
from model.multihead_projector import MultiHeadProjector
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoConfig, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Dict, Sequence, List, Tuple, Union
import json, os
import torch
from model.utils import find_correct_case_file_name, tokenize_insert
from torch import nn
import numpy as np
from model.utils import *
import torch.nn.functional as F
import pandas as pd
from sentence_transformers import SentenceTransformer
from safetensors.torch import load_file
from Table_Encoder.model.load_encoder import load_encoder
from config import SPIDER_CSV_PATH, SENTENCE_TRANSFORMER_PATH, SPIDER_EMB_PATH, MAX_ROW, BASE_MODEL_PATH
if 'MAX_COL' in os.environ:
    MAX_COL = int(os.environ['MAX_COL'])
    print(f'find new MAX_COL in environ: {MAX_COL}')
    
class Model(nn.Module):

    # 将成员属性decoder指向model

    def __init__(self, *, decoder, tokenizer, encoder_tokenizer, torch_dtype):
        super().__init__()
        self.encoder = nn.Parameter(torch.randn(MAX_COL, 10, 256, dtype=torch.bfloat16))
        self.projector = MultiHeadProjector(projector_type="mlp2x_gelu", encoder_hidden_size=256, decoder_hidden_size=3584, num_heads=1, torch_dtype=torch.bfloat16, multihead=False)
        self.decoder = decoder
        self.torch_dtype = torch_dtype
        self.tokenizer = tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.gradient_checkpointing_enable = self.decoder.gradient_checkpointing_enable
        
        
    @classmethod
    def from_pretrained(cls, path):
        raise NotImplementedError
        # encoder = load_encoder().to(dtype = torch.bfloat16)
        # projector = MultiHeadProjector(
        #     projector_type="mlp2x_gelu",
        #     encoder_hidden_size=3584,
        #     decoder_hidden_size=3584,
        #     num_heads=1,
        #     torch_dtype=torch.bfloat16,
        #     multihead=False
        # )
        # decoder = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(dtype = torch.bfloat16)
        # tokenizer = AutoTokenizer.from_pretrained(path)
        # encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_PATH)
        # model = cls(encoder=encoder, projector=projector, decoder=decoder, tokenizer=tokenizer, torch_dtype=torch.bfloat16, encoder_tokenizer=encoder_tokenizer).to(dtype = torch.bfloat16)
        # print('model initialized')

        # model.load_state_dict(load_file(os.path.join(path, 'model.safetensors'))) 
        # print('model loaded')
        # # tokenizer = AutoTokenizer.from_pretrained(decoder_path)
        # # decoder = AutoModelForCausalLM.from_pretrained(decoder_path, torch_dtype = torch_dtype)
        # # encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_PATH)
        # # model = cls(encoder=encoder, projector=projector, decoder=decoder, tokenizer=tokenizer, torch_dtype=torch_dtype, encoder_tokenizer=encoder_tokenizer)
        # return modelq

    def get_encoder_output(self, path_csv, path_emb):
        batched_encoder = torch.stack([self.encoder for _ in range(len(path_csv))], dim=0).clone()
        return batched_encoder
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        path_emb: Optional[str] = None,
        path_csv: Optional[str] = None,
        insert_embs = None # 是否往中间插入embedding
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # get table embeddings
        bs = input_ids.shape[0]
        table_embeds = self.get_encoder_output(path_csv, path_emb)
        prepare_embs_func = self.projector.prepare_embeds if insert_embs == None or insert_embs[0] == False else self.projector.prepare_insert_embeds
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = prepare_embs_func(
            decoder = self.decoder,
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            table_embeds=table_embeds,
        )
        return self.decoder.forward(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds.to(dtype = self.decoder.dtype),
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # cache_position=cache_position,
            return_dict=return_dict
        )
    
    @torch.inference_mode()
    def generate(self, input_str = None, input_ids = None, path_emb = None, path_csv = None, insert_embs = None, max_new_tokens = 1024, **kwargs):
        if input_ids is None:
            if insert_embs is not None and insert_embs == True:
                input_ids = tokenize_insert(input_str, self.tokenizer).unsqueeze(0)
            else: 
                input_ids = self.tokenizer(input_str, return_tensors='pt')['input_ids']
            input_ids = input_ids.to(self.decoder.model.device)

        if type(path_emb) == str:
            path_emb = [path_emb]
            path_csv = [path_csv]
            
        table_embeds = self.get_encoder_output(path_csv, path_emb)
        
        prepare_embs_func = self.projector.prepare_embeds if insert_embs == False else self.projector.prepare_insert_embeds
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = prepare_embs_func(
            decoder = self.decoder,
            input_ids=input_ids,
            # position_ids,
            table_embeds=table_embeds,
        )
        # print('inputs_embeds', inputs_embeds.shape)
        attention_mask = torch.ones(inputs_embeds.shape[:-1], device=inputs_embeds.device)
        
        return self.decoder.generate(
            max_new_tokens=max_new_tokens,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds.to(dtype = self.decoder.dtype),
            use_cache=True,
            **kwargs
        )
    
    
    