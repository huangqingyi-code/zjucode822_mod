import pandas as pd
import json
import numpy as np

with open('data/Table_samples_50_column_select_v2.json') as f:
    jsonl = json.load(f)

# randomly select 10 tables
idxes = np.random.choice(len(jsonl), 10, replace=False)
for idx in idxes:
    table_info = jsonl[idx]
    table = pd.read_csv(table_info['Data Path'])
    ori_col = table.columns
    amb_col = table_info['Column Name']
    fil_col = ori_col.difference(amb_col)
    table = table[fil_col]
    print(table)