'''
model with provided pre-generated table embeddings
need a db_path in config and a db_id in each input to retrieve the table embeddings
'''
from model.decoder import TableDecoder
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
from model.encoder import TableEncoder
from sentence_transformers import SentenceTransformer
from config import SPIDER_CSV_PATH, SENTENCE_TRANSFORMER_PATH, SPIDER_EMB_PATH
class MyConfig(PretrainedConfig):
    model_type = "my_emb_model"

    def __init__(self, device='cuda', mm_projector_type='linear', encoder_hidden_size=384,
                 decoder_hidden_size=4096, base_model_name_or_path=None, model_path=None, db_path=None, torch_dtype="float32", **kwargs):
        super().__init__(torch_dtype=torch_dtype) 
        self.device = device
        self.mm_projector_type = mm_projector_type
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.base_model_name_or_path = base_model_name_or_path
        self.model_path = model_path
        self.db_path = db_path
        
    @classmethod
    def from_pretrained(cls, model_path = None, config_dict = None, **kwargs): # if model_path is provided, it means the config, model, learnable_embedding are pretrained; otherwise a new model will be created with the provided config_dict
        if model_path is not None:
            with open(os.path.join(model_path, 'config.json'), 'r') as f:
                config_dict = json.load(f)
        
        elif config_dict is None:
            raise ValueError("The 'config_dict' must be provided if 'model_path' is None.")            
        
        base_model_name_or_path = config_dict.get('base_model_name_or_path')
        if base_model_name_or_path is None:
            raise ValueError("The config file must contain the 'base_model_name_or_path' field.")
        
        # base_config = AutoConfig.from_pretrained(base_model_name_or_path)
        
        all_dict = {
            **config_dict,
            **kwargs,
            'model_path': model_path
        }
        return cls(**all_dict)


class Model(nn.Module):

    # 将成员属性decoder指向model

    def __init__(self, *, config, encoder, decoder = None, learnable_embedding = None, learnable_embedding_projector = None, tokenizer, decoder_config_dict = None):
        super().__init__()
        self.config = config
        self.gradient_checkpointing_enable = decoder.model.gradient_checkpointing_enable
        if learnable_embedding is not None:
            self.learnable_embedding = nn.parameter.Parameter(learnable_embedding)
        else:
            self.learnable_embedding = None

        if learnable_embedding_projector is not None:
            self.learnable_embedding_projector = learnable_embedding_projector
        else:
            self.learnable_embedding_projector = None

        self.encoder = encoder
        self.decoder = decoder
        
        self.tokenizer = tokenizer
        self.torch_dtype = config.torch_dtype
        # self.db_path = config.db_path
        self.sentencetransformer = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)

    # def train(self, mode=True):
    #     self.encoder.train(mode)
    #     self.decoder.projector.train(mode)
    #     self.decoder.model.train(False)
    #     self.sentencetransformer.train(False)
    
    # def eval(self):
    #     self.encoder.eval()
    #     self.decoder.projector.eval()
    #     self.decoder.model.eval()
    #     self.sentencetransformer.eval()
    #     # requires_grad
    #     for param in self.parameters():
    #         param.requires_grad = False
    
    @classmethod
    def from_pretrained(cls, encoder_path=None, decoder_path=None, learnable_embedding_path = None, learnable_embedding_projector_path = None, config_dict = None): # if model_path is provided, it means the config, model, learnable_embedding are pretrained; else a new model will be created with the provided config_dict

        config = MyConfig.from_pretrained(model_path=decoder_path, config_dict=config_dict)
        dtype = config.torch_dtype
        
        encoder = torch.load(encoder_path).to(dtype=config.torch_dtype)
        print('load encoder from', encoder_path)
        decoder = TableDecoder.from_pretrained(config=config)
        print('load decoder from', decoder_path)
        learnable_embedding = None
        
        if learnable_embedding_path is not None:
            learnable_embedding = torch.load(learnable_embedding_path).to(dtype=config.torch_dtype).clone().detach()
            print('load learnable_embedding from', learnable_embedding_path, learnable_embedding.shape)
        else:
            learnable_embedding = None

        if learnable_embedding_projector_path is not None:
            learnable_embedding_projector = torch.load(learnable_embedding_projector_path).to(dtype=config.torch_dtype)
            print('load projector from', learnable_embedding_projector_path)
        else:
            learnable_embedding_projector = None
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        import os
        model = cls(config=config, encoder=encoder, decoder=decoder, tokenizer=tokenizer, learnable_embedding = learnable_embedding, learnable_embedding_projector = learnable_embedding_projector)
        return model

    
    def get_embedded_table(self, db_id, table):
        # torch.manual_seed(11)
        # np.random.seed(11)
        # row_size = 50
        # column_size = 50
        # db_path = SPIDER_CSV_PATH

        # embedding_dim = self.sentencetransformer.get_sentence_embedding_dimension()
        # try:
        #     new_table = find_correct_case_file_name(os.path.join(db_path, db_id), table)
        #     table_df = pd.read_csv(os.path.join(db_path, db_id, new_table + '.csv'), encoding='utf-8', low_memory=False)
        # except:
        #     raise ValueError(f"Table {table} not found in database {db_id}")
        # table = new_table
        # if len(table_df) > row_size:
        #     table_df = table_df.sample(n=row_size)
        # table = table_df.to_numpy()
        # table = table.astype(str)
        # table_emb = np.zeros((table.shape[0], table.shape[1], embedding_dim))
        # for j, row in enumerate(table):
        #     row_emb = self.sentencetransformer.encode(row)
        #     table_emb[j] = row_emb
        
        # column_truncation = np.random.permutation(range(table_emb.shape[1]))[:column_size]
        # table_emb = table_emb[:, column_truncation, :]

        # table_emb_padded = np.zeros((row_size, 100, table_emb.shape[2]))
        # table_emb_padded[:table_emb.shape[0], :table_emb.shape[1], :] = table_emb
        # table_mask = np.zeros((row_size, 100))
        # table_mask[:table_emb.shape[0], :table_emb.shape[1]] = 1
        
        path = os.path.join(SPIDER_EMB_PATH, db_id, table + '.pt')
        table_emb_padded, table_mask = torch.load(path)
        return torch.tensor(table_emb_padded, device=self.decoder.model.device, dtype = self.torch_dtype), torch.tensor(table_mask, device=self.decoder.model.device, dtype = self.torch_dtype)


    
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
        db_ids = None,
        tables = None,
        insert_embs = None # 是否往中间插入embedding
    ) -> Union[Tuple, CausalLMOutputWithPast]: # TODO: 清理一下参数
        if db_ids != None:
            bs = input_ids.shape[0]
            table_embeds = []
            for i in range(bs):
                table_embeds.append(self.get_embedded_table(db_ids[i], tables[i]))
                
            table_embeds = []
            table_masks = []
            for db_id, table in zip(db_ids, tables):
                table_emb, table_mask = self.get_embedded_table(db_id, table)
                table_embeds.append(table_emb)
                table_masks.append(table_mask)
            table_embeds = torch.stack(table_embeds).clone().detach() # prevent any operations on sentencetransformer
            table_masks = torch.stack(table_masks).clone().detach()
            table_embeds = self.encoder(anchor_table=table_embeds, anchor_table_mask=table_masks, inference=True)
            # table_embeds = F.normalize(table_embeds,dim=-1)
        # print('sum', sum(self.learnable_embedding))
        # print('sum en', next(self.encoder.parameters()))
        # print('sum pr', sum(next(self.decoder.projector.parameters())))
        # if self.learnable_embedding is not None:
        #     table_embeds = torch.cat([table_embeds, self.learnable_embedding.repeat(input_ids.shape[0], 1, 1)], dim=1)
        # table_embeds = []
        # for i in range(bs):
        #     table_embeds.append(self.get_generated_db_embeds(db_ids[i], tables[i]))
        # print('sp', self.learnable_embedding.view(-1)[:10])
        # print('spproj', next(self.learnable_embedding_projector.parameters()).view(-1)[:10])
        return self.decoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            table_embeds=table_embeds,
            insert_embs = insert_embs,
            learnable_embedding=self.learnable_embedding_projector(self.learnable_embedding) if self.learnable_embedding is not None else None
        )
    
    def generate(self, input_str = None, input_ids = None, db_id = None, table = None, insert_embs = None, max_new_tokens = 1024, **kwargs):
        if input_ids is None:
            if insert_embs is not None and insert_embs == True:
                input_ids = tokenize_insert(input_str, self.tokenizer).unsqueeze(0)
            else: 
                input_ids = self.tokenizer(input_str, return_tensors='pt')['input_ids']
            input_ids = input_ids.to(self.decoder.model.device)
        bs = input_ids.shape[0]
        assert bs == 1, "batch size must be 1"
        
        
        table_emb, table_mask = self.get_embedded_table(db_id, table)
        table_embeds = self.encoder(anchor_table=table_emb.unsqueeze(0), anchor_table_mask=table_mask.unsqueeze(0), inference=True)
        # table_embeds = F.normalize(table_embeds,dim=-1)
        assert len(table_embeds.shape) == 3, "table_embeds must be 3D" # bs * n_tokens * hidden_size
        # extended_table_embeds = self.learnable_embedding.repeat(bs, 1, 1)

        return self.decoder.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens, 
            table_embeds=table_embeds, 
            learnable_embeds=self.learnable_embedding_projector(self.learnable_embedding) if self.learnable_embedding is not None else None, 
            insert_embs = insert_embs,
            **kwargs)
    
    
    