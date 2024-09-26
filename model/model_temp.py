'''
model with provided encoder and decoder
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
import pandas as pd
from model.encoder import TableEncoder
from sentence_transformers import SentenceTransformer

    
class Model(nn.Module):


    def __init__(self, *, encoder, decoder, tokenizer):
        '''
            __init__只会在from_pretrained中调用，因此上面的参数可以改
        '''
        super().__init__()
        pass

    @classmethod
    def from_pretrained(cls, encoder_path, decoder_path, **kwargs):
        pass

    def generate(self, input_str = None, context_str = None, max_new_tokens = 114514, **kwargs):
        '''
            single batch generation
            args:
                input_str: str
                context_str: str
            return: 
                ModelOutput
        '''
        if input_ids is None:
            input_ids = self.tokenizer(input_str, return_tensors='pt')['input_ids'].to(self.decoder.model.device)
        bs = input_ids.shape[0]
        table_embeds = []
        # TODO: get table embeddings
        return self.decoder.generate(input_ids, max_new_tokens, table_embeds, **kwargs)
    