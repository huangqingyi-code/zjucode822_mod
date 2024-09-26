from sentence_transformers import SentenceTransformer
import os
SENTENCE_TRANSFORMER_PATH = '/data0/pretrained-models/all-MiniLM-L6-v2'
os.environ["SENTENCE_TRANSFORMER"] = 'all-MiniLM-L6-v2'
os.environ['MODELS_PATH'] = '/data0/pretrained-models'
from Table_Encoder.model.encoder import TableEncoder
# from config import SENTENCE_TRANSFORMER_PATH

from transformers import AutoModel
from torch import nn
import torch
import argparse

def load_encoder(path = None, **kwargs):
    args = {
        'num_columns': 20, 
        'embedding_size': 384, 
        'transformer_depth': 12, 
        'attention_heads': 16, 
        'attention_dropout': 0.1, 
        'ff_dropout': 0.1, 
        'dim_head': 64,
        'decode': True,
        'pred_type': 'contrastive',
        'qformer_qnum': 10
    }
    # 用kwargs更新args
    args.update(kwargs)
    print("!!qnum", args['qformer_qnum'])
    # 将args转换为命名空间
    args = argparse.Namespace(**args)
    st = AutoModel.from_pretrained(SENTENCE_TRANSFORMER_PATH)
    args.embedding_size = st.config.hidden_size
    model = TableEncoder(
            num_cols=args.num_columns,
            depth=args.transformer_depth,
            heads=args.attention_heads,
            attn_dropout=args.attention_dropout,
            ff_dropout=args.ff_dropout,
            attentiontype="colrow",
            # decode=args.decode,
            pred_type=args.pred_type,
            dim_head=args.dim_head,
            pooling='mean',
            col_name=False,
            numeric_mlp=False,
            qformer_qnum=args.qformer_qnum
        ).cpu()
    if path is not None:
        model.load_state_dict(torch.load(path), strict=False)
        
    # print(next(model.parameters())[:10])
    # path = '/data0/ll/checkpoints/contrastive-all-MiniLM-L6-v2-7-None-1e-05-0.0001-16-ColMatching-20240717/model_25400.pt'
    # # path = '/data4/sft_output/qwen2-base-0717/checkpoint-2000'
    # import torch
    # # model = nn.DataParallel(model)
    # try:
    #     model.load_state_dict(torch.load(path))
    # except Exception as e:
    #     print(e)
    # print(next(model.parameters())[:10])
    return model