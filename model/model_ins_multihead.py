'''
model with provided pre-generated table embeddings
need a db_path in config and a db_id in each input to retrieve the table embeddings
'''
import torch.distributed
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
if 'MAX_ROW' in os.environ:
    MAX_ROW = int(os.environ['MAX_ROW'])
    print(f'find new MAX_ROW in environ: {MAX_ROW}')
    
    
class Model(nn.Module):

    # 将成员属性decoder指向model

    def __init__(self, *, encoder, projector, decoder, tokenizer, encoder_tokenizer, torch_dtype):
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.decoder = decoder
        self.torch_dtype = torch_dtype
        self.tokenizer = tokenizer
        self.encoder_tokenizer = encoder_tokenizer
        self.gradient_checkpointing_enable = self.decoder.gradient_checkpointing_enable
        
        
    @classmethod
    def from_pretrained(cls, path, qformer_qnum = 10):
        print("!!", path)
        assert os.path.exists(path), f"model path {path} not exists"
        
        encoder = load_encoder(qformer_qnum=qformer_qnum).to(dtype = torch.bfloat16)
        projector = MultiHeadProjector(
            projector_type="mlp2x_gelu",
            encoder_hidden_size=3584,
            decoder_hidden_size=3584,
            num_heads=1,
            torch_dtype=torch.bfloat16,
            multihead=False
        )
        decoder = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH).to(dtype = torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(path)
        encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_PATH)
        model = cls(encoder=encoder, projector=projector, decoder=decoder, tokenizer=tokenizer, torch_dtype=torch.bfloat16, encoder_tokenizer=encoder_tokenizer).to(dtype = torch.bfloat16)
        print('model initialized')

        model.load_state_dict(load_file(os.path.join(path, 'model.safetensors'))) 
        print('model loaded')
        # tokenizer = AutoTokenizer.from_pretrained(decoder_path)
        # decoder = AutoModelForCausalLM.from_pretrained(decoder_path, torch_dtype = torch_dtype)
        # encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(SENTENCE_TRANSFORMER_PATH)
        # model = cls(encoder=encoder, projector=projector, decoder=decoder, tokenizer=tokenizer, torch_dtype=torch_dtype, encoder_tokenizer=encoder_tokenizer)
        return model

    
    def get_embedded_table(self, path_csv, path_emb):
        def process_table_df(table_df):
            numeric_columns = table_df.select_dtypes(include=["number"]).columns
            numeric_indices = [
                table_df.columns.get_loc(col) for col in numeric_columns
            ]
            
            # fill missing values with mean
            table_df[numeric_columns] = table_df[numeric_columns].apply(
                lambda col: col.fillna(col.mean() if not col.isna().all() else 0)
            )
            if len(table_df) > MAX_ROW:
                table_df = table_df.sample(n=MAX_ROW)
                
            
            table_np = table_df.to_numpy().astype(str)
            
            return table_np
        def load_tokenized_table(anchor_table):
            anchor_table = process_table_df(anchor_table)
            num_rows, num_cols = anchor_table.shape[0], anchor_table.shape[1]
            anchor_row_num = anchor_table.shape[0]
            anchor_table = anchor_table.reshape(-1)
            max_length = 64
            tokenized_anchor_table = self.encoder_tokenizer(anchor_table.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')                
            tokenized_anchor_table = {k: v.reshape(anchor_row_num, num_cols, -1).to(self.decoder.device) for k, v in tokenized_anchor_table.items()}
            return tokenized_anchor_table


        # print(f'loading csv from {path_csv}') 
        table_df = pd.read_csv(
            path_csv,
            encoding="utf-8",
            low_memory=False,
            nrows=500
        )
        # if table_df.shape[0] > MAX_ROW:
        #     table_df = table_df.sample(n=MAX_ROW)
        #     table_df.to_csv(path_csv, index=False)
        anchor_table = load_tokenized_table(table_df)
        num_cols = anchor_table['input_ids'].shape[1]
        anchor_table_row_num = anchor_table['input_ids'].shape[0]
        anchor_table_padded = {k: F.pad(v, (0, 0, 0, MAX_COL - v.shape[1], 0, MAX_ROW - v.shape[0]), "constant", 1) for k, v in anchor_table.items()}
        anchor_table_mask = np.zeros((MAX_ROW, MAX_COL))

        anchor_table_mask[:anchor_table_row_num, : num_cols] = 1
        ret = (
            anchor_table_padded['input_ids'].to(device = self.decoder.device),
            anchor_table_padded['attention_mask'].to(device = self.decoder.device),
            anchor_table_padded['token_type_ids'].to(device = self.decoder.device),
            torch.tensor(anchor_table_mask, device = self.decoder.device),
        )
        return ret
            

    
    def get_encoder_output(self, path_csv, path_emb):
        anchor_table_input_ids = []
        anchor_table_attention_mask = []
        anchor_table_token_type_ids = []
        anchor_table_mask = []
        for c, e in zip(path_csv, path_emb):
            p, q, r, s = self.get_embedded_table(c, e)
            anchor_table_input_ids.append(p)
            anchor_table_attention_mask.append(q)
            anchor_table_token_type_ids.append(r)
            anchor_table_mask.append(s)
        
        anchor_table_input_ids = torch.stack(anchor_table_input_ids, dim=0)
        anchor_table_attention_mask = torch.stack(anchor_table_attention_mask, dim=0)
        anchor_table_token_type_ids = torch.stack(anchor_table_token_type_ids, dim=0)
        anchor_table_mask = torch.stack(anchor_table_mask, dim=0)
        # table_embeds = self.encoder(anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask, inference=True)            
        table_embeds = self.encoder(anchor_table_input_ids, anchor_table_attention_mask, anchor_table_token_type_ids, anchor_table_mask, inference=True)            
        # table_embeds = self.encoder(anchor_table_input_ids, 1, 1, anchor_table_mask, inference=True)Î
        # table_embeds = F.normalize(table_embeds,dim=-1)
        return table_embeds
    
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
        reorder_idx = None,
        insert_embs = None # 是否往中间插入embedding
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # get table embeddings
        # print(next(self.encoder.transformer.parameters())[:10])
        bs = input_ids.shape[0]
        table_embeds = self.get_encoder_output(path_csv, path_emb)
        if reorder_idx is not None:
            table_embeds = torch.gather(table_embeds, dim = 1, index = reorder_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, table_embeds.shape[-2], table_embeds.shape[-1]))
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
    
    
    