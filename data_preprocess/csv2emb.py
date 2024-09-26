import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
from dotenv import load_dotenv

load_dotenv()
MODELS_PATH = os.environ["MODELS_PATH"]
DATASETS_PATH = os.environ["DATASETS_PATH"]

# st_name = "paraphrase-MiniLM-L6-v2"
st_name = "all-MiniLM-L6-v2"

def verify():
    embedded_table_path = '../data/embedded_tables'
    embedded_table_names = os.listdir(embedded_table_path)
    for embedded_table_name in embedded_table_names:
        embedded_table = np.load(os.path.join(embedded_table_path, embedded_table_name))
        print(embedded_table.shape)

if __name__ == '__main__':
    # verify()
    
    # Load sentence transformer
    model = SentenceTransformer(f'{MODELS_PATH}/{st_name}')
    embedding_dim = model.get_sentence_embedding_dimension()
    
    # csv files with meta data
    # Load table data
    # data_path = '/data0/datasets/tabular-data'
    # meta_data_path = os.path.join(data_path, 'single_meta_data.json')
    # with open(meta_data_path) as f:
    #     meta_data = json.load(f)
    # table_meta_list = list(meta_data.items())
    
    # for i, (table_id, table_meta) in enumerate(tqdm(table_meta_list)):
    #     try: 
    #         table_path = os.path.join(data_path, table_meta['table_path'].strip('./'))
    #         table_df = pd.read_csv(table_path, encoding='utf-8', low_memory=False, nrows=200)
    #         if table_df.shape[0] < 10 or table_df.shape[1] < 3:
    #             continue
    #         table = table_df.to_numpy()
    #         table = table.astype(str)
    #         table_emb = np.zeros((table.shape[0], table.shape[1], embedding_dim))
    #         for j, row in enumerate(table):
    #             row_emb = model.encode(row)
    #             table_emb[j] = row_emb
    #         with open(f'../data/embedded_tables/{table_id}.npy', 'wb') as f:
    #             np.save(f, table_emb)
    #     except:
    #         with open('../data/embedded_tables/error.log', 'a') as f:
    #             f.write(f'{table_id}\n')
    
    
    # list of csv files
    # data_path = '/data0/datasets/embedded_tables_3'
    # csv_files = os.listdir(data_path)
    
    # count = 0
    
    # for csv_file in tqdm(csv_files):
    #     try: 
    #         table_path = os.path.join(data_path, csv_file)
    #         table_df = pd.read_csv(table_path, encoding='utf-8', low_memory=False)
    #         if table_df.shape[0] < 1000 or table_df.shape[1] < 5:
    #             continue
    #         table = table_df.to_numpy()
    #         table = table.astype(str)
    #         table_emb = np.zeros((table.shape[0], table.shape[1], embedding_dim))
    #         for j, row in enumerate(table):
    #             row_emb = model.encode(row)
    #             table_emb[j] = row_emb
    #         with open(f'/data0/datasets/embedded_tables_3/table-{count}.npy', 'wb') as f:
    #             np.save(f, table_emb)
    #         count += 1
    #     except:
    #         with open('/data0/datasets/embedded_tables_3/error.log', 'a') as f:
    #             f.write(f'{count}\n')
    
    
    # spider dataset
    database_path = f'{DATASETS_PATH}/spider/database_csv'
    database_files = os.listdir(database_path)
    
    count = 0
    
    for database_file in tqdm(database_files):
        database_file_path = os.path.join(database_path, database_file)
        for csv_file in os.listdir(database_file_path):
            try: 
                table_path = os.path.join(database_file_path, csv_file)
                table_df = pd.read_csv(table_path, encoding='utf-8', low_memory=False, nrows=200)
                if table_df.shape[0] < 10 or table_df.shape[1] < 3:
                    continue
                table = table_df.to_numpy()
                table = table.astype(str)
                table_emb = np.zeros((table.shape[0], table.shape[1], embedding_dim))
                for j, row in enumerate(table):
                    row_emb = model.encode(row)
                    table_emb[j] = row_emb
                dir_path = f'{DATASETS_PATH}/embedded_tables_spider_{st_name}'
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                with open(os.path.join(dir_path, f'table-{count}.npy'), 'wb') as f:
                    np.save(f, table_emb)
                count += 1
            except:
                with open(os.path.join(dir_path, 'error.log'), 'a') as f:
                    f.write(f'{count}\n')