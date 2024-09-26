'''
在__init__时候生成足够长的随机向量，每次获取embedding时候返回与原有embedding长度相同的随机向量前缀
need a db_path in config and a db_id in each input to retrieve the table embeddings
'''
from model.decoder import TableDecoder
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, AutoConfig, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from typing import Optional, Dict, Sequence, List, Tuple, Union
import json, os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class MyConfig(PretrainedConfig):
    model_type = "my_emb_model"

    def __init__(self, device='cuda', mm_projector_type='linear', encoder_hidden_size=384,
                 decoder_hidden_size=4096, base_model_name_or_path=None, model_path=None, db_path=None, **kwargs):
        super().__init__(**kwargs)
        self.device = device
        self.mm_projector_type = mm_projector_type
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.base_model_name_or_path = base_model_name_or_path
        self.model_path = model_path
        self.db_path = db_path
        
    @classmethod
    def from_pretrained(cls, model_path = None, config_dict = None, **kwargs): # if model_path is provided, it means the config, model, learnable_embedding are pretrained; else a new model will be created with the provided config_dict

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

    def __init__(self, config, decoder, tokenizer, rand_embedding):
        super().__init__()
        self.config = config
        self.gradient_checkpointing_enable = decoder.model.gradient_checkpointing_enable
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.torch_dtype = config.torch_dtype
        self.db_path = config.db_path
        self.rand_embedding = rand_embedding

    @classmethod
    def from_pretrained(cls, model_path=None, config_dict = None): # if model_path is provided, it means the config, model, learnable_embedding are pretrained; else a new model will be created with the provided config_dict
        config = MyConfig.from_pretrained(model_path=model_path, config_dict=config_dict)
        
        print('model_config', config)
        decoder = TableDecoder.from_pretrained(config=config)
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        if model_path is not None:
            rand_embedding = torch.load(os.path.join(model_path, 'rand_embedding.bin')).to(decoder.model.device).requires_grad_(False)
            print('load pretrained rand_embedding from', os.path.join(model_path, 'rand_embedding.bin'))
        else:
            rand_embedding = torch.randn(512, config.encoder_hidden_size).to(decoder.model.device).requires_grad_(False)
            rand_embedding = F.normalize(rand_embedding, p=2, dim=-1)
        model = cls(config, decoder, tokenizer, rand_embedding)
        return model

    def get_db_embeds(self, db_id, table=None):
        # randomly select a db_id from DB_ID_LIST
        # db_id = np.random.choice(DB_ID_LIST)
        db_path = self.config.db_path
        cur_db_path = os.path.join(db_path, db_id)
        # 获取以.npy结尾的所有文件
        assert (table != None)
        if table == None:
            npy_list = [os.path.join(cur_db_path, file) for file in os.listdir(cur_db_path) if file.endswith('.npy')]
            ret = torch.cat([torch.tensor(np.load(os.path.join(cur_db_path, npy_file)), dtype=self.torch_dtype, device=self.decoder.model.device, requires_grad=False).detach() for npy_file in npy_list], dim=0)
        else:
            npy_file = table + '.npy'
            assert os.path.exists(os.path.join(cur_db_path, npy_file))
            ret = torch.tensor(np.load(os.path.join(cur_db_path, npy_file)), dtype=self.torch_dtype, device=self.decoder.model.device, requires_grad=False).detach()
        rand_ret = self.rand_embedding[:ret.shape[0]].clone().detach().to(self.decoder.model.device)
        rand_ret.requires_grad_(False)
        return rand_ret
    
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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        bs = input_ids.shape[0]
        table_embeds = []
        for i in range(bs):
            table_embeds.append(self.get_db_embeds(db_ids[i], tables[i]))
            # db_path = self.config.db_path
            # cur_db_path = os.path.join(db_path, db_ids[i])
            # # 获取以.npy结尾的所有文件
            # npy_list = [os.path.join(cur_db_path, file) for file in os.listdir(cur_db_path) if file.endswith('.npy')]
            # table_embeds.append(torch.cat([torch.tensor(np.load(os.path.join(cur_db_path, npy_file)), dtype=self.torch_dtype, device=self.decoder.model.device, requires_grad=False).detach() for npy_file in npy_list], dim=0))
        # TODO: 防止多次加载相同的表格。似乎表格长度都有限，感觉可以强行存
        
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
            table_embeds=table_embeds
        )
    
    def generate(self, input_str = None, input_ids = None, db_id = None, table = None, max_new_tokens = 114514, **kwargs):
        if input_ids is None:
            input_ids = self.tokenizer(input_str, return_tensors='pt')['input_ids'].to(self.decoder.model.device)
        bs = input_ids.shape[0]
        assert bs == 1, "batch size must be 1"
        
        table_embeds = self.get_db_embeds(db_id, table).unsqueeze(0)
        assert len(table_embeds.shape) == 3, "table_embeds must be 3D" # bs * n_tokens * hidden_size
        # extended_table_embeds = self.learnable_embedding.repeat(bs, 1, 1)
        return self.decoder.generate(input_ids, max_new_tokens, table_embeds, **kwargs)