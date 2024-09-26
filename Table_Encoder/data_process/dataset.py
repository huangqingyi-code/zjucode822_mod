import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import os
import math
import random
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from dotenv import load_dotenv

load_dotenv()
DATASETS_PATH = os.environ["DATASETS_PATH"]
MODELS_PATH = os.environ["MODELS_PATH"]
SENTENCE_TRANSFORMER = os.environ["SENTENCE_TRANSFORMER"]

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def list_npy_csv_files(data_path, from_csv=False):
    if not from_csv:
        file_list = os.listdir(data_path)
        npy_files = []
        for n_file in file_list:
            if n_file.endswith(".npy"):
                npy_files.append(os.path.join(data_path, n_file))
        return npy_files
    else:
        # for 2-level directory structure
        # file_list = os.listdir(data_path)
        # csv_files = []
        # for d_file in file_list:
        #     csv_file_list = os.listdir(os.path.join(data_path, d_file))
        #     for c_file in csv_file_list:
        #         if c_file.endswith(".csv"):
        #             csv_files.append(os.path.join(data_path, d_file, c_file))
        # return csv_files
        
        # for 1-level directory structure
        file_list = os.listdir(data_path)
        csv_files = []
        for c_file in file_list:
            if c_file.endswith(".csv"):
                csv_files.append(os.path.join(data_path, c_file))
        return csv_files


def list_2npy_files(data_path):
    file_list = os.listdir(data_path)
    npy_files = []
    for file in file_list:
        if file.endswith(".npy") or file.endswith(".csv"):
            npy_files.append(file)
    res = []
    for i in range(0, len(npy_files)):
        for j in range(i, len(npy_files)):
            res.append([npy_files[i], npy_files[j]])
            if len(res) >= 3000:
                break
        if len(res) >= 3000:
            break
    return res


def list_same_col_files(data_path):
    res = []
    col_dir_list = [
        f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))
    ]
    for col_dir in col_dir_list:
        file_list = os.listdir(os.path.join(data_path, col_dir))
        npy_files = []
        for file in file_list:
            if file.endswith(".npy") or file.endswith(".csv"):
                npy_files.append(os.path.join(data_path, col_dir, file))
        neg_num = 0
        for i in range(len(npy_files)):
            for j in range(i + 1, len(npy_files)):
                res.append([npy_files[i], npy_files[j]])
                neg_num += 1
                if neg_num >= 20:
                    break
            if neg_num >= 20:
                break
        for k in range(neg_num):
            rand = random.randint(0, len(npy_files) - 1)
            res.append([npy_files[rand], npy_files[rand]])
    return res


def generate_special_tokens():
    model = SentenceTransformer(f"{MODELS_PATH}/{SENTENCE_TRANSFORMER}")
    embedding_dim = model.get_sentence_embedding_dimension()
    shuffle_tokens_num = 100
    shuffle_tokens = model.encode(
        [f"[unused{i}]" for i in range(shuffle_tokens_num)]
    )  # [unused1]<col1>[unused1] 50*(100+2st)*384?
    cls_token = model.encode("[CLS]")
    sep_token = model.encode("[SEP]")
    np.savez(
        f"data/special_tokens_{SENTENCE_TRANSFORMER}.npz",
        shuffle_tokens=shuffle_tokens,
        cls_token=cls_token,
        sep_token=sep_token,
    )


def get_special_tokens():
    special_tokens = np.load(f"data/special_tokens_{SENTENCE_TRANSFORMER}.npz")
    return (
        torch.from_numpy(special_tokens["cls_token"]),
        torch.from_numpy(special_tokens["sep_token"]),
        torch.from_numpy(special_tokens["shuffle_tokens"]),
    )


def get_device(module):
    if next(module.parameters(), None) is not None:
        return next(module.parameters()).device
    elif next(module.buffers(), None) is not None:
        return next(module.buffers()).device
    else:
        raise ValueError("The module has no parameters or buffers.")

# 定义一个函数，尝试将列转换为数字类型，如果成功则返回 True，否则返回 False
def is_convertible_to_numeric(series):
    try:
        pd.to_numeric(series)
        return True
    except ValueError:
        return False

class TableDataset(Dataset):
    def __init__(
        self,
        data_path,
        pred_type="contrastive",
        from_csv=False,
        model=None,
        idx=None,
        shuffle_num=3,
        numeric_mlp=False
    ):
        self.data_path = data_path
        self.pred_type = pred_type
        self.numeric_mlp = numeric_mlp
                
        # load the learned speicial tokens from a certain SentenceTransformer
        # self.cls_token, self.sep_token, self.shuffle_tokens = get_special_tokens()
        
        self.from_csv = from_csv
        # self.model = model
        self.shuffle_num = (
            3  # num of cols that need to be shuffled, won't influence contrastive
        )
        
        # read data list
        self.table_list = np.load(self.data_path)
        
        self.max_row = 10
        self.max_col = 20
        
        # scan data file
        # self.table_list = list_npy_csv_files(self.data_path, self.from_csv)
        # self.table_list = np.array(self.table_list)
            
        if idx is not None:
            self.table_list = self.table_list[idx]
        
        self.tokenizer = AutoTokenizer.from_pretrained(f"{MODELS_PATH}/{SENTENCE_TRANSFORMER}", use_fast=False)
        self.st_name = SENTENCE_TRANSFORMER

    def __len__(self):
        return len(self.table_list)
    
    def process_table_df(self, table_df):
        if len(table_df.columns) > self.max_col:
            table_df = table_df.sample(n=self.max_col, axis=1)
        
        numeric_columns = table_df.select_dtypes(include=["number"]).columns
        numeric_indices = [
            table_df.columns.get_loc(col) for col in numeric_columns
        ]
        
        # fill missing values with mean
        table_df[numeric_columns] = table_df[numeric_columns].apply(
            lambda col: col.fillna(col.mean() if not col.isna().all() else 0)
        )
        
        if len(table_df) > self.max_row * 2:
            table_df = table_df.sample(n=self.max_row * 2)
        
        table_np = table_df.to_numpy().astype(str)
        
        return table_np
    
    def load_tokenized_table(self, table_file):
        tokenizer = self.tokenizer
        
        table_df = pd.read_csv(
            table_file,
            encoding="utf-8",
            low_memory=False,
            nrows=500
        )

        # size = [num_rows, num_cols]
        table = self.process_table_df(table_df)
        num_rows, num_cols = table.shape[0], table.shape[1]
        
        anchor_table, shuffled_table = self.split_table(table)
        
        anchor_row_num = anchor_table.shape[0]
        shuffled_row_num = shuffled_table.shape[0]
                
        shuffled_table, shuffle_idx = self.shuffle_table(shuffled_table)
        
        anchor_table, shuffled_table = anchor_table.reshape(-1), shuffled_table.reshape(-1)

        # size = [num_cells, seq_len]
        # 
        
        max_length = 64
        tokenized_anchor_table = tokenizer(anchor_table.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
        tokenized_shuffled_table = tokenizer(shuffled_table.tolist(), padding='max_length', truncation=True, max_length=max_length, return_tensors='pt')
                
        tokenized_anchor_table = {k: v.reshape(anchor_row_num, num_cols, -1) for k, v in tokenized_anchor_table.items()}
        tokenized_shuffled_table = {k: v.reshape(shuffled_row_num, num_cols, -1) for k, v in tokenized_shuffled_table.items()}
                
        assert tokenized_anchor_table['input_ids'].shape[2] == tokenized_shuffled_table['input_ids'].shape[2]
        
        return tokenized_anchor_table, tokenized_shuffled_table, shuffle_idx

    def split_table(self, table):
        num_rows = table.shape[0]
        anchor_table_row_num = num_rows // 2
        shuffled_table_row_num = num_rows - anchor_table_row_num
        
        anchor_table = table[:anchor_table_row_num]
        shuffled_table = table[-shuffled_table_row_num:]
        
        return anchor_table, shuffled_table
    
    def shuffle_table(self, shuffled_table):
        # Shuffle columns
        # Randomly select columns to shuffle
        shuffle_idx = torch.randperm(shuffled_table.shape[1])
        shuffled_table = shuffled_table[:, shuffle_idx]
        
        return shuffled_table, shuffle_idx

    def __getitem__(self, idx):
        anchor_table, shuffled_table, shuffle_idx = self.load_tokenized_table(self.table_list[idx])
        num_cols = anchor_table['input_ids'].shape[1]
        
        anchor_table_row_num = anchor_table['input_ids'].shape[0]
        shuffled_table_row_num = shuffled_table['input_ids'].shape[0]
            
        anchor_table_padded = {k: F.pad(v, (0, 0, 0, self.max_col - v.shape[1], 0, self.max_row - v.shape[0]), "constant", 1) for k, v in anchor_table.items()}
        shuffled_table_padded = {k: F.pad(v, (0, 0, 0, self.max_col - v.shape[1], 0, self.max_row - v.shape[0]), "constant", 1) for k, v in shuffled_table.items()}

        anchor_table_mask = np.zeros((self.max_row, self.max_col))
        shuffled_table_mask = np.zeros((self.max_row, self.max_col))

        anchor_table_mask[:anchor_table_row_num, : num_cols] = 1
        shuffled_table_mask[:shuffled_table_row_num, : num_cols] = 1

        shuffle_idx_padded = F.pad(shuffle_idx, (0, self.max_col - len(shuffle_idx)), "constant", -1)
                
        if self.st_name == 'all-MiniLM-L6-v2' or self.st_name == 'bge-small-en-v1.5':        
            return (
                anchor_table_padded['input_ids'],
                anchor_table_padded['attention_mask'],
                anchor_table_padded['token_type_ids'],
                shuffled_table_padded['input_ids'],
                shuffled_table_padded['attention_mask'],
                shuffled_table_padded['token_type_ids'],
                anchor_table_mask,
                shuffled_table_mask,
                shuffle_idx_padded
            )
        elif self.st_name == 'puff-base-v1':
            return (
                anchor_table_padded['input_ids'],
                anchor_table_padded['attention_mask'],
                torch.zeros_like(anchor_table_padded['input_ids']),
                shuffled_table_padded['input_ids'],
                shuffled_table_padded['attention_mask'],
                torch.zeros_like(anchor_table_padded['input_ids']),
                anchor_table_mask,
                shuffled_table_mask,
                shuffle_idx_padded
            )
    
    # def __getitem__(self, idx):
    #     if self.pred_type == "name_prediction":
    #         table_emb, origin_table_emb, col_name_emb = load_table_embeddings(
    #             self.table_list[idx], add_col_emb=True
    #         )
    #     elif self.pred_type == 'contrastive':
    #         table_emb, col_name_emb = load_table_embeddings(self.table_list[idx])
    #     else:
    #         table_emb = load_table_embeddings(self.table_list[idx])

    #     # Row sampling: No more than self.max_row rows for anchor table and 5 rows for shuffled table (for reconstruction tasks)
    #     if self.pred_type == "name_prediction":
    #         anchor_table_row_num_upper = min(table_emb.shape[0] // 2, self.max_row // 2)
    #         anchor_table_row_num = np.random.randint(4, anchor_table_row_num_upper)
    #         shuffled_table_row_num_upper = min(len(table_emb) - anchor_table_row_num, 5)
    #         shuffled_table_row_num = np.random.randint(1, shuffled_table_row_num_upper)

    #         shuffle_num = max(int(table_emb.shape[1] * 0.6), 3)
    #         # shuffle_num = self.shuffle_num

    #         anchor_table = table_emb[:anchor_table_row_num]
    #         shuffled_table = origin_table_emb[
    #             anchor_table_row_num : anchor_table_row_num + shuffled_table_row_num
    #         ]
    #     # Row sampling: Completely random for contrastive tasks
    #     else:
    #         # anchor_table_row_num_upper = min(table_emb.shape[0] // 2, self.max_row // 2)
    #         # anchor_table_row_num = np.random.randint(4, anchor_table_row_num_upper)
    #         # shuffled_table_row_num_upper = min(
    #         #     len(table_emb) - anchor_table_row_num, self.max_row // 2
    #         # )
    #         # shuffled_table_row_num = np.random.randint(1, shuffled_table_row_num_upper)

    #         # anchor_table = table_emb[:anchor_table_row_num]
    #         # shuffled_table = table_emb[
    #         #     anchor_table_row_num : anchor_table_row_num + shuffled_table_row_num
    #         # ]
            
    #         anchor_table_row_num = min(table_emb.shape[0] // 2, self.max_row)
    #         shuffled_table_row_num = min(len(table_emb) - anchor_table_row_num, self.max_row)

    #         anchor_table = table_emb[:anchor_table_row_num].reshape(anchor_table_row_num, table_emb.shape[1], table_emb.shape[2])
    #         shuffled_table = table_emb[-shuffled_table_row_num:].reshape(shuffled_table_row_num, table_emb.shape[1], table_emb.shape[2])
            

    #     # Shuffle columns
    #     # Randomly select columns to shuffle
    #     shuffle_idx_pre = torch.randperm(shuffled_table.shape[1])
    #     shuffle_idx_post = shuffle_idx_pre[torch.randperm(shuffle_idx_pre.shape[0])]
    #     shuffled_table[:, shuffle_idx_post, :] = shuffled_table[:, shuffle_idx_pre, :]
    #     if self.pred_type == "name_prediction":
    #         col_name_emb[shuffle_idx_post, :] = col_name_emb[shuffle_idx_pre, :]

        
    #     if self.pred_type == "name_prediction":
    #         anchor_table_padded = np.zeros((50, 50, table_emb.shape[2]))
    #         shuffled_table_padded = np.zeros((5, 50, table_emb.shape[2]))
    #         anchor_table_padded[:anchor_table_row_num, : table_emb.shape[1], :] = (
    #             anchor_table
    #         )
    #         shuffled_table_padded[:shuffled_table_row_num, :shuffle_num, :] = (
    #             shuffled_table[:, shuffle_idx_post, :]
    #         )
    #         anchor_table_mask = torch.zeros((50, 50))
    #         shuffled_table_mask = torch.zeros((5, 50))

    #         anchor_table_mask[:anchor_table_row_num, : table_emb.shape[1]] = 1
    #         shuffled_table_mask[:shuffled_table_row_num, :shuffle_num] = 1

    #         target_padded = torch.zeros((50, table_emb.shape[2]))
    #         target_padded[:shuffle_num, :] = col_name_emb[shuffle_idx_post]

    #         target_mask = torch.zeros(50)
    #         target_mask[:shuffle_num] = 1
    #         return (
    #             anchor_table_padded,
    #             shuffled_table_padded,
    #             anchor_table_mask,
    #             shuffled_table_mask,
    #             target_padded,
    #             target_mask,
    #         )
    #     else:
    #         # pad the table to [self.max_row, self.max_col, 384]
    #         # anchor_table_padded = F.pad(anchor_table, (0, 0, 0, self.max_col - anchor_table.shape[1], 0, self.max_row - anchor_table.shape[0]), "constant", 0)
    #         # shuffled_table_padded = F.pad(shuffled_table, (0, 0, 0, self.max_col - shuffled_table.shape[1], 0, self.max_row - shuffled_table.shape[0]), "constant", 0)

    #         # anchor_table_mask = np.zeros((self.max_row, self.max_col))
    #         # shuffled_table_mask = np.zeros((self.max_row, self.max_col))

    #         # anchor_table_mask[:anchor_table_row_num, : table_emb.shape[1]] = 1
    #         # shuffled_table_mask[:shuffled_table_row_num, : shuffled_table.shape[1]] = 1

            
    #         # pre_idx_padded = F.pad(shuffle_idx_pre, (0, self.max_col - len(shuffle_idx_pre)), "constant", -1)
    #         # post_idx_padded = F.pad(shuffle_idx_post, (0, self.max_col - len(shuffle_idx_post)), "constant", -1)
            
    #         # col_name_emb_padded = F.pad(col_name_emb, (0, 0, 0, self.max_col - len(col_name_emb)), "constant", 0)
            
    #         # pad the table to [self.max_row, self.max_col, 384]
    #         anchor_table_padded = F.pad(anchor_table, (0, 0, 0, self.max_col - anchor_table.shape[1], 0, self.max_row - anchor_table.shape[0]), "constant", 0)
    #         shuffled_table_padded = F.pad(shuffled_table, (0, 0, 0, self.max_col - shuffled_table.shape[1], 0, self.max_row - shuffled_table.shape[0]), "constant", 0)

    #         anchor_table_mask = np.zeros((self.max_row, self.max_col))
    #         shuffled_table_mask = np.zeros((self.max_row, self.max_col))

    #         anchor_table_mask[:anchor_table_row_num, : table_emb.shape[1]] = 1
    #         shuffled_table_mask[:shuffled_table_row_num, : shuffled_table.shape[1]] = 1

            
    #         pre_idx_padded = F.pad(shuffle_idx_pre, (0, self.max_col - len(shuffle_idx_pre)), "constant", -1)
    #         post_idx_padded = F.pad(shuffle_idx_post, (0, self.max_col - len(shuffle_idx_post)), "constant", -1)
            
    #         col_name_emb_padded = F.pad(col_name_emb, (0, 0, 0, self.max_col - len(col_name_emb)), "constant", 0)
            
    #         return (
    #             anchor_table_padded,
    #             shuffled_table_padded,
    #             anchor_table_mask,
    #             shuffled_table_mask,
    #             pre_idx_padded,
    #             post_idx_padded,
    #             col_name_emb_padded
    #         )
            
    # load table embeddings either from raw csv or pre-processed npy files
    def load_table_embeddings(self, table_file, add_col_emb=False):
        if self.from_csv:
            embedding_dim = self.embedding_dim
            tokenizer = AutoTokenizer.from_pretrained(f"{MODELS_PATH}/{SENTENCE_TRANSFORMER}")
                            
            table_df = pd.read_csv(
                table_file,
                encoding="utf-8",
                low_memory=False,
                nrows=100
            )

            table = self.process_table_df(table_df)
            
            table_emb = torch.zeros((table.shape[0], table.shape[1], embedding_dim)).to(device=get_device(self.model))
            for j, row in enumerate(table):
                if self.numeric_mlp:
                    row_emb = torch.zeros((table.shape[1], embedding_dim)).to(device=get_device(self.model))
                    if len(numeric_indices) > 0:
                        row_emb[numeric_indices] = (
                            self.model.module.num_mlp(
                                torch.tensor(
                                    row[numeric_indices]
                                    .astype(np.float32)
                                    .reshape(-1, 1)
                                ).to(device=get_device(self.model))
                            )
                        )
                    if len(non_numeric_indices) > 0:
                        # bge-small-en
                        # encoded_input = tokenizer(row, padding=True, truncation=True, return_tensors='pt')
                        # encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
                        # row_emb = self.model.module.st(**encoded_input) # for multi-gpu
                        # table_emb[j] = F.normalize(row_emb[0][:, 0], p=2, dim=1)
                        
                            # all-minilm
                        encoded_input = tokenizer(row[non_numeric_indices].astype(str).tolist(), padding=True, truncation=True, return_tensors='pt')
                        encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
                        encoded_output = self.model.module.st(**encoded_input) # for multi-gpu
                        encoded_output = mean_pooling(encoded_output, encoded_input['attention_mask'])
                        row_emb[non_numeric_indices] = F.normalize(encoded_output, p=2, dim=1)
                    table_emb[j] = row_emb
                else:
                    encoded_input = tokenizer(row.astype(str).tolist(), padding=True, truncation=True, return_tensors='pt')
                    encoded_input = {k: v.cuda() for k, v in encoded_input.items()}
                    encoded_output = self.model.module.st(**encoded_input) # for multi-gpu
                    encoded_output = mean_pooling(encoded_output, encoded_input['attention_mask'])
                    table_emb[j] = F.normalize(encoded_output, p=2, dim=1)
                
            column_names = table_df.columns.values.astype(str)
            column_names = column_names.tolist()
            
            # bge-small-en
            # encoded_input_col = tokenizer(column_names, padding=True, truncation=True, return_tensors='pt')
            # encoded_input_col = {k: v.cuda() for k, v in encoded_input_col.items()}
            # output = self.model.module.st(**encoded_input_col)
            # col_name_emb = F.normalize(output[0][:, 0], p=2, dim=1)
            
            # all-minilm
            encoded_input_col = tokenizer(column_names, padding=True, truncation=True, return_tensors='pt')
            encoded_input_col = {k: v.cuda() for k, v in encoded_input_col.items()}
            encoded_output_col = self.model.module.st(**encoded_input_col)
            encoded_output_col = mean_pooling(encoded_output_col, encoded_input_col['attention_mask'])
            col_name_emb = F.normalize(encoded_output_col, p=2, dim=1)
            
            table_emb = table_emb.cpu()
            col_name_emb = col_name_emb.cpu()

            if add_col_emb:
                origin_table_emb = table_emb.copy()
                column_names = table_df.columns.to_list()
                col_name_emb = self.model.encode(column_names) # NOTE: may have bug related to torch/numpy
                for j, row in enumerate(table):
                    table_emb[j] += col_name_emb
        else:
            table_emb = np.load(table_file)
            
            # Row/column truncation + shuffle
            row_truncation = np.random.permutation(range(table_emb.shape[0]))[:self.max_row]
            table_emb = table_emb[row_truncation, :, :]

            column_truncation = np.random.permutation(range(table_emb.shape[1]))[:self.max_col]
            table_emb = table_emb[:, column_truncation, :]

        

        

        if add_col_emb:
            col_name_emb = col_name_emb[column_truncation, :]
            origin_table_emb = origin_table_emb[row_truncation, :, :]
            origin_table_emb = origin_table_emb[:, column_truncation, :]
            return table_emb, origin_table_emb, col_name_emb
        elif self.pred_type == 'contrastive':
            return table_emb, col_name_emb
        else:
            return table_emb 

if __name__ == "__main__":
    generate_special_tokens()
    cls_token, sep_token, shuffle_tokens = get_special_tokens()
    shuffle_tokens_mat = shuffle_tokens[:3].unsqueeze(1).repeat(1, 3, 1)
    print(shuffle_tokens_mat)

# historical copy

# class TableDataset(Dataset):
#     def __init__(
#         self,
#         data_path,
#         pred_type="column_classification",
#         from_csv=False,
#         model=None,
#         train=True,
#         shuffle_num=3,
#     ):
#         self.data_path = data_path
#         self.pred_type = pred_type
        
#         # load the learned speicial tokens from a certain SentenceTransformer
#         self.cls_token, self.sep_token, self.shuffle_tokens = get_special_tokens()
#         self.from_csv = from_csv
#         self.model = model
#         self.shuffle_num = (
#             3  # num of cols that need to be shuffled, won't influence contrastive
#         )
        
#         if self.pred_type == "row_classification":
#             self.table_list = list_same_col_files(self.data_path)
#         elif self.pred_type == "column_classification":
#             self.table_list = list_2npy_files(self.data_path)
#         else:
#             self.table_list = list_npy_csv_files(self.data_path, self.from_csv)
#         self.table_list = np.array(self.table_list)
            
#         # randomly split the dataset into train and valid
#         # the number of valid samples is 10% of the whole dataset
#         valid_num = len(self.table_list) // 10
#         perm_idx = np.random.permutation(len(self.table_list))
#         train_idx = perm_idx[: -valid_num]
#         valid_idx = perm_idx[-valid_num:]
#         if train:
#             self.table_list = self.table_list[train_idx]
#         else:
#             self.table_list = self.table_list[valid_idx]

#     def __len__(self):
#         return len(self.table_list)

#     def __getitem__(self, idx):
#         # load table embeddings either from raw csv or pre-processed npy files
#         def load_table_embeddings(table_file, add_col_emb=False):
#             if self.from_csv:
#                 # embedding_dim = self.model.st.get_sentence_embedding_dimension()
#                 embedding_dim = self.model.module.st.get_sentence_embedding_dimension() # for multi-gpu

#                 table_df = pd.read_csv(
#                     os.path.join(self.data_path, table_file),
#                     encoding="utf-8",
#                     low_memory=False,
#                 )

#                 if len(table_df) > 100:
#                     table_df = table_df.sample(n=100)
#                 table = table_df.to_numpy()
#                 table = table.astype(str)
#                 table_emb = np.zeros((table.shape[0], table.shape[1], embedding_dim))
#                 for j, row in enumerate(table):
#                     # row_emb = self.model.st.encode(row)
#                     row_emb = self.model.module.st.encode(row) # for multi-gpu
#                     table_emb[j] = row_emb

#                 if add_col_emb:
#                     origin_table_emb = table_emb.copy()
#                     column_names = table_df.columns.to_list()
#                     col_name_emb = self.model.encode(column_names)
#                     for j, row in enumerate(table):
#                         table_emb[j] += col_name_emb
#             else:
#                 table_emb = np.load(os.path.join(self.data_path, table_file))

#             # Row/column truncation + shuffle
#             row_truncation = np.random.permutation(range(table_emb.shape[0]))[:100]
#             table_emb = table_emb[row_truncation, :, :]

#             column_truncation = np.random.permutation(range(table_emb.shape[1]))[:50]
#             table_emb = table_emb[:, column_truncation, :]

#             if add_col_emb:
#                 col_name_emb = col_name_emb[column_truncation, :]
#                 origin_table_emb = origin_table_emb[row_truncation, :, :]
#                 origin_table_emb = origin_table_emb[:, column_truncation, :]
#                 return table_emb, origin_table_emb, col_name_emb
#             else:
#                 return table_emb

#         if (
#             self.pred_type == "row_classification"
#             or self.pred_type == "column_classification"
#         ):
#             table_emb = load_table_embeddings(self.table_list[idx][0])
#             shuffled_table_emb = load_table_embeddings(self.table_list[idx][1])
#         elif self.pred_type == "name_prediction":
#             table_emb, origin_table_emb, col_name_emb = load_table_embeddings(
#                 self.table_list[idx], add_col_emb=True
#             )
#         else:
#             table_emb = load_table_embeddings(self.table_list[idx])

#         # Row sampling: No more than 50 rows for anchor table and 5 rows for shuffled table (for reconstruction tasks)
#         if (
#             self.pred_type == "generation"
#             or self.pred_type == "classification"
#             or self.pred_type == "rank_index"
#             or self.pred_type == "rank_comparison"
#         ):
#             anchor_table_row_num_upper = min(table_emb.shape[0] // 2, 50)
#             anchor_table_row_num = np.random.randint(4, anchor_table_row_num_upper)
#             shuffled_table_row_num_upper = min(len(table_emb) - anchor_table_row_num, 5)
#             shuffled_table_row_num = np.random.randint(1, shuffled_table_row_num_upper)

#             # shuffle_num = max(int(table_emb.shape[1] * 0.6), 3)
#             shuffle_num = self.shuffle_num

#             anchor_table = table_emb[:anchor_table_row_num]
#             shuffled_table = table_emb[
#                 anchor_table_row_num : anchor_table_row_num + shuffled_table_row_num
#             ]
#         elif self.pred_type == "name_prediction":
#             anchor_table_row_num_upper = min(table_emb.shape[0] // 2, 50)
#             anchor_table_row_num = np.random.randint(4, anchor_table_row_num_upper)
#             shuffled_table_row_num_upper = min(len(table_emb) - anchor_table_row_num, 5)
#             shuffled_table_row_num = np.random.randint(1, shuffled_table_row_num_upper)

#             shuffle_num = max(int(table_emb.shape[1] * 0.6), 3)
#             # shuffle_num = self.shuffle_num

#             anchor_table = table_emb[:anchor_table_row_num]
#             shuffled_table = origin_table_emb[
#                 anchor_table_row_num : anchor_table_row_num + shuffled_table_row_num
#             ]
#         elif self.pred_type == "column_classification":
#             anchor_table_row_num_upper = min(table_emb.shape[0] // 2, 50)
#             anchor_table_row_num = np.random.randint(4, anchor_table_row_num_upper)
#             shuffled_table_row_num = np.random.randint(4, shuffled_table_emb.shape[0])
#             shuffled_table_row_num = max(int(shuffled_table_emb.shape[1] * 0.6), 3)
#             anchor_table = table_emb[:anchor_table_row_num]
#             shuffled_table = shuffled_table_emb[:shuffled_table_row_num]
#             shuffle_num = self.shuffle_num
#         # Row sampling: shuffled_table only take one random row.
#         elif self.pred_type == "row_classification":
#             anchor_table_row_num_upper = min(table_emb.shape[0] // 2, 49)
#             anchor_table_row_num = np.random.randint(4, anchor_table_row_num_upper)
#             shuffled_table_row_num = np.random.randint(0, shuffled_table_emb.shape[0])
#             shuffle_num = self.shuffle_num
#             anchor_table = table_emb[:anchor_table_row_num]
#             shuffled_table = shuffled_table_emb[
#                 shuffled_table_row_num : shuffled_table_row_num + 1
#             ]
#         # Row sampling: Completely random for contrastive tasks
#         else:
#             anchor_table_row_num_upper = min(table_emb.shape[0] // 2, 50)
#             anchor_table_row_num = np.random.randint(4, anchor_table_row_num_upper)
#             shuffled_table_row_num_upper = min(
#                 len(table_emb) - anchor_table_row_num, 50
#             )
#             shuffled_table_row_num = np.random.randint(1, shuffled_table_row_num_upper)

#             shuffle_num = table_emb.shape[1]

#             anchor_table = table_emb[:anchor_table_row_num]
#             shuffled_table = table_emb[
#                 anchor_table_row_num : anchor_table_row_num + shuffled_table_row_num
#             ]

#         # Shuffle columns
#         # Randomly select columns to shuffle
#         shuffle_idx_pre = np.random.permutation(shuffled_table.shape[1])[:shuffle_num]
#         shuffle_idx_post = shuffle_idx_pre.copy()
#         np.random.shuffle(shuffle_idx_post)
#         shuffled_table[:, shuffle_idx_post, :] = shuffled_table[:, shuffle_idx_pre, :]
#         if self.pred_type == "name_prediction":
#             col_name_emb[shuffle_idx_post, :] = col_name_emb[shuffle_idx_pre, :]

#         # Mark the columns that have been shuffled with shuffle tokens (for reconstruction tasks)
#         if self.pred_type == "generation" or self.pred_type == "classification":
#             shuffle_tokens = self.shuffle_tokens[:shuffle_num]
#             # Insert shuffle tokens
#             shuffled_table_ = np.zeros(
#                 (
#                     shuffled_table.shape[0],
#                     table_emb.shape[1] + shuffle_num,
#                     table_emb.shape[2],
#                 )
#             )
#             for row in range(shuffled_table_.shape[0]):
#                 shuffled_table_[row] = np.insert(
#                     shuffled_table[row], shuffle_idx_post, shuffle_tokens, axis=0
#                 )
#             shuffled_table = shuffled_table_

#             # Target tokens
#             # Find the index of each element in shuffle_idx_pre in the array of shuffle_idx_post
#             target_tokens_idx = np.zeros(shuffle_num, dtype=int)
#             for i, idx in enumerate(shuffle_idx_pre):
#                 target_tokens_idx[i] = np.where(shuffle_idx_post == idx)[0][0]

#             # Get the index of the shuffle tokens after being inserted into the table
#             shuffled_tokens_idx = np.zeros(table_emb.shape[1], dtype=int)
#             shuffled_tokens_idx = np.insert(shuffled_tokens_idx, shuffle_idx_post, 1)

#         if self.pred_type == "generation":
#             anchor_table_template = np.zeros((50, 100, table_emb.shape[2]))
#             shuffled_table_template = np.zeros((5, 100, table_emb.shape[2]))
#             target_template = np.zeros((100, table_emb.shape[2]))

#             anchor_table_padded = anchor_table_template.copy()
#             shuffled_table_padded = shuffled_table_template.copy()

#             anchor_table_padded[:anchor_table_row_num, : table_emb.shape[1], :] = (
#                 anchor_table
#             )
#             shuffled_table_padded[
#                 :shuffled_table_row_num, : shuffled_table.shape[1], :
#             ] = shuffled_table

#             # TODO
#             anchor_table_mask = np.zeros((50, 100))
#             shuffled_table_mask = np.zeros((5, 100))

#             anchor_table_mask[:anchor_table_row_num, : table_emb.shape[1]] = 1
#             shuffled_table_mask[:shuffled_table_row_num, : shuffled_table.shape[1]] = 1

#             # Method 1: Return the target tokens
#             # target_tokens = np.zeros((table_emb.shape[1], table_emb.shape[2]))
#             # target_tokens = np.insert(
#             #     target_tokens,
#             #     shuffle_idx_post,
#             #     shuffle_tokens[target_tokens_idx],
#             #     axis=0,
#             # )

#             # target_padded = target_template.copy()
#             # target_padded[: target_tokens.shape[0], :] = target_tokens
#             # target_mask = target_template.copy()
#             # target_mask[: len(shuffled_tokens_idx)] = np.expand_dims(
#             #     shuffled_tokens_idx, axis=1
#             # )

#             # Method 2: Return a mapping from the original input to the target tokens
#             shuffled_tokens_mapping = np.where(shuffled_tokens_idx == 1)[0]
#             gt_tokens = target_tokens_idx[np.argsort(shuffle_idx_post)]
#             reverse = np.argsort(np.argsort(shuffle_idx_post))
#             shuffled_tokens_mapping = shuffled_tokens_mapping[reverse[gt_tokens]]

#             target_padded = -np.ones(100, dtype=int)
#             target_padded[: len(shuffled_tokens_mapping)] = shuffled_tokens_mapping
#             target_mask = np.zeros(100)
#             target_mask[: len(shuffled_tokens_idx)] = shuffled_tokens_idx

#             # add segment embeddings for shuffled tokens and shuffled columns
#             shuffled_table_padded[
#                 :shuffled_table_row_num, np.where(shuffled_tokens_idx == 1)[0], :
#             ] += self.sep_token
#             shuffled_table_padded[
#                 :shuffled_table_row_num, np.where(shuffled_tokens_idx == 1)[0] + 1, :
#             ] += self.sep_token

#             return (
#                 anchor_table_padded,
#                 shuffled_table_padded,
#                 anchor_table_mask,
#                 shuffled_table_mask,
#                 target_padded,
#                 target_mask,
#             )

#             # Assertion: Check the correctness of the target tokens
#             # target_tokens_ = shuffled_table[0, shuffled_tokens_mapping, :]

#             # restore = shuffled_table.copy()
#             # restore[:, shuffled_tokens_mapping + 1, :] = restore[
#             #     :, np.where(shuffled_tokens_idx == 1)[0] + 1, :
#             # ]
#             # restore = np.delete(restore, np.where(shuffled_tokens_idx == 1)[0], axis=1)
#             # assert (restore == shuffled_table_ori).all()

#         elif self.pred_type == "classification":
#             target = -np.ones(table_emb.shape[1])
#             target = np.insert(target, shuffle_idx_post, target_tokens_idx)

#             anchor_table_template = np.zeros((50, 100, table_emb.shape[2]))
#             shuffled_table_template = np.zeros((5, 100, table_emb.shape[2]))
#             target_template = np.zeros((100, table_emb.shape[2]))

#             anchor_table_padded = anchor_table_template.copy()
#             shuffled_table_padded = shuffled_table_template.copy()

#             anchor_table_padded[:anchor_table_row_num, : table_emb.shape[1], :] = (
#                 anchor_table
#             )
#             shuffled_table_padded[
#                 :shuffled_table_row_num, : shuffled_table.shape[1], :
#             ] = shuffled_table

#             # TODO
#             anchor_table_mask = np.zeros((50, 100))
#             shuffled_table_mask = np.zeros((5, 100))

#             anchor_table_mask[:anchor_table_row_num, : table_emb.shape[1]] = 1
#             shuffled_table_mask[:shuffled_table_row_num, : shuffled_table.shape[1]] = 1

#             target_padded = -np.ones(100)
#             target_padded[: target.shape[0]] = target
#             target_mask = np.zeros(100)
#             target_mask[: len(shuffled_tokens_idx)] = shuffled_tokens_idx

#             shuffled_table_padded[
#                 :shuffled_table_row_num, np.where(shuffled_tokens_idx == 1)[0], :
#             ] += self.sep_token
#             shuffled_table_padded[
#                 :shuffled_table_row_num, np.where(shuffled_tokens_idx == 1)[0] + 1, :
#             ] += self.sep_token

#             return (
#                 anchor_table_padded,
#                 shuffled_table_padded,
#                 anchor_table_mask,
#                 shuffled_table_mask,
#                 target_padded,
#                 target_mask,
#             )
#         elif self.pred_type == "rank_index":
#             anchor_table_padded = np.zeros((50, 50, table_emb.shape[2]))
#             shuffled_table_padded = np.zeros((5, 50, table_emb.shape[2]))
#             anchor_table_padded[:anchor_table_row_num, : table_emb.shape[1], :] = (
#                 anchor_table
#             )
#             shuffled_table_padded[:shuffled_table_row_num, 0, :] += self.cls_token
#             shuffled_table_padded[:shuffled_table_row_num, 1 : shuffle_num + 1, :] = (
#                 shuffled_table[:, shuffle_idx_post, :]
#             )
#             anchor_table_mask = np.zeros((50, 50))
#             shuffled_table_mask = np.zeros((5, 50))

#             anchor_table_mask[:anchor_table_row_num, : table_emb.shape[1]] = 1
#             shuffled_table_mask[:shuffled_table_row_num, : shuffle_num + 1] = 1

#             target_padded = np.argsort(shuffle_idx_post)
#             target_map = {
#                 (0, 1, 2): np.array([0]),
#                 (0, 2, 1): np.array([1]),
#                 (1, 0, 2): np.array([2]),
#                 (1, 2, 0): np.array([3]),
#                 (2, 0, 1): np.array([4]),
#                 (2, 1, 0): np.array([5]),
#             }
#             target_padded = target_map[tuple(target_padded)]
#             target_mask = np.ones(1)
#             return (
#                 anchor_table_padded,
#                 shuffled_table_padded,
#                 anchor_table_mask,
#                 shuffled_table_mask,
#                 target_padded,
#                 target_mask,
#             )
#         elif self.pred_type == "rank_comparison":
#             anchor_table_padded = np.zeros((50, 50, table_emb.shape[2]))
#             shuffled_table_padded = np.zeros((5, 50, table_emb.shape[2]))
#             anchor_table_padded[:anchor_table_row_num, : table_emb.shape[1], :] = (
#                 anchor_table
#             )
#             shuffled_table_padded[:shuffled_table_row_num, :shuffle_num, :] = (
#                 shuffled_table[:, shuffle_idx_post, :]
#             )
#             anchor_table_mask = np.zeros((50, 50))
#             shuffled_table_mask = np.zeros((5, 50))

#             anchor_table_mask[:anchor_table_row_num, : table_emb.shape[1]] = 1
#             shuffled_table_mask[:shuffled_table_row_num, :shuffle_num] = 1

#             target_padded = -np.ones(50)
#             target_padded[: shuffle_num - 1] = (
#                 shuffle_idx_post[:-1] > shuffle_idx_post[1:]
#             )  # 1: > 0: <

#             target_mask = np.zeros(50)
#             target_mask[: shuffle_num - 1] = 1
#             return (
#                 anchor_table_padded,
#                 shuffled_table_padded,
#                 anchor_table_mask,
#                 shuffled_table_mask,
#                 target_padded,
#                 target_mask,
#             )
#         elif self.pred_type == "row_classification":
#             anchor_table_padded = np.zeros((50, 50, table_emb.shape[2]))
#             shuffled_table_padded = np.zeros((1, 50, table_emb.shape[2]))
#             anchor_table_padded[:anchor_table_row_num, 0, :] += self.cls_token
#             anchor_table_padded[
#                 :anchor_table_row_num, 1 : table_emb.shape[1] + 1, :
#             ] = anchor_table
#             shuffled_table_padded[0, : shuffled_table.shape[1], :] = shuffled_table
#             anchor_table_mask = np.zeros((50, 50))
#             shuffled_table_mask = np.zeros((1, 50))

#             anchor_table_mask[:anchor_table_row_num, : table_emb.shape[1] + 1] = 1
#             shuffled_table_mask[0, : shuffled_table.shape[1]] = 1

#             if self.table_list[idx][0] == self.table_list[idx][1]:
#                 target_padded = np.ones(1)
#             else:
#                 target_padded = np.zeros(1)
#             target_mask = np.ones(1)
#             return (
#                 anchor_table_padded,
#                 shuffled_table_padded,
#                 anchor_table_mask,
#                 shuffled_table_mask,
#                 target_padded,
#                 target_mask,
#             )
#         elif self.pred_type == "column_classification":
#             anchor_table_padded = np.zeros((50, 50, table_emb.shape[2]))
#             shuffled_table_padded = np.zeros((5, 50, table_emb.shape[2]))
            
#             # random select some cols from anchor table
#             select_col_num = max(3, table_emb.shape[1] // 2)
#             idx = np.random.permutation(range(table_emb.shape[1]))
#             select_col_idx = idx[:select_col_num]
#             other_col_idx = idx[select_col_num:]
#             anchor_table_emb = anchor_table[:, select_col_idx]

#             # random select some cols from shuffle table
#             shuffled_table_row_num = min(shuffled_table.shape[0], anchor_table.shape[0], 5)
#             shuffled_table_emb = np.concat([anchor_table[:shuffled_table_row_num, other_col_idx], shuffled_table[:shuffled_table_row_num]], axis=1)
#             target = np.zeros(shuffled_table_emb.shape[1])
#             target[:len(other_col_idx)] = 1

#             idx = np.random.permutation(range(shuffled_table_emb.shape[1]))[:50]
#             shuffled_table_emb = shuffled_table_emb[:, idx]
#             target = target[idx]

#             anchor_table_padded[
#                 :anchor_table_row_num, : anchor_table_emb.shape[1], :
#             ] = anchor_table_emb
#             shuffled_table_padded[:shuffled_table_row_num, : shuffled_table_emb.shape[1], :] = shuffled_table_emb
#             anchor_table_mask = np.zeros((50, 50))
#             shuffled_table_mask = np.zeros((5, 50))

#             anchor_table_mask[:anchor_table_row_num, : anchor_table_emb.shape[1]] = 1
#             shuffled_table_mask[:shuffled_table_row_num, : shuffled_table_emb.shape[1]] = 1

#             target_padded = np.zeros(50)
#             target_padded[:len(target)] = target

#             target_mask = np.zeros(50)
#             target_mask[:len(target)] = 1
#             return (
#                 anchor_table_padded,
#                 shuffled_table_padded,
#                 anchor_table_mask,
#                 shuffled_table_mask,
#                 target_padded,
#                 target_mask,
#             )
#         elif self.pred_type == "name_prediction":
#             anchor_table_padded = np.zeros((50, 50, table_emb.shape[2]))
#             shuffled_table_padded = np.zeros((5, 50, table_emb.shape[2]))
#             anchor_table_padded[:anchor_table_row_num, : table_emb.shape[1], :] = (
#                 anchor_table
#             )
#             shuffled_table_padded[:shuffled_table_row_num, :shuffle_num, :] = (
#                 shuffled_table[:, shuffle_idx_post, :]
#             )
#             anchor_table_mask = np.zeros((50, 50))
#             shuffled_table_mask = np.zeros((5, 50))

#             anchor_table_mask[:anchor_table_row_num, : table_emb.shape[1]] = 1
#             shuffled_table_mask[:shuffled_table_row_num, :shuffle_num] = 1

#             target_padded = np.zeros((50, table_emb.shape[2]))
#             target_padded[:shuffle_num, :] = col_name_emb[shuffle_idx_post]

#             target_mask = np.zeros(50)
#             target_mask[:shuffle_num] = 1
#             return (
#                 anchor_table_padded,
#                 shuffled_table_padded,
#                 anchor_table_mask,
#                 shuffled_table_mask,
#                 target_padded,
#                 target_mask,
#             )
#         else:
#             anchor_table_padded = np.zeros((50, 100, table_emb.shape[2]))
#             shuffled_table_padded = np.zeros((50, 100, table_emb.shape[2]))

#             anchor_table_padded[:anchor_table_row_num, : table_emb.shape[1], :] = (
#                 anchor_table
#             )
#             shuffled_table_padded[
#                 :shuffled_table_row_num, : shuffled_table.shape[1], :
#             ] = shuffled_table

#             # TODO
#             anchor_table_mask = np.zeros((50, 100))
#             shuffled_table_mask = np.zeros((50, 100))

#             anchor_table_mask[:anchor_table_row_num, : table_emb.shape[1]] = 1
#             shuffled_table_mask[:shuffled_table_row_num, : shuffled_table.shape[1]] = 1

#             target_pre_padded = -np.ones(100, dtype=int)
#             target_pre_padded[:shuffle_num] = shuffle_idx_pre
#             target_post_padded = -np.ones(100, dtype=int)
#             target_post_padded[:shuffle_num] = shuffle_idx_post

#             return (
#                 anchor_table_padded,
#                 shuffled_table_padded,
#                 anchor_table_mask,
#                 shuffled_table_mask,
#                 target_pre_padded,
#                 target_post_padded,
#             )