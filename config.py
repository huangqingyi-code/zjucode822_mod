POLLUTE_TABLE_PATH = './dataset/spider_pollute/tables.json'
EVALUATIION_FILE = './test-suite-sql-eval/evaluation.py'
# EVALUATIION_FILE = '/home/llong/gxj/test-suite-sql-eval/evaluation.py'
# SPIDER_DB_PATH = '/data0/datasets/spider/all_database'
SPIDER_DB_PATH = '/home/llong/gxj/code/dataset/sft_multitask/spider/database'
# SENTENCE_TRANSFORMER_PATH = "/data0/pretrained-models/paraphrase-MiniLM-L6-v2"
# SENTENCE_TRANSFORMER_PATH = "/data0/pretrained-models/puff-base-v1"
SENTENCE_TRANSFORMER_PATH = "/data0/pretrained-models/all-MiniLM-L6-v2"
SPIDER_CSV_PATH='/data0/datasets/spider/database_csv'
SPIDER_DEV_ORIGINAL_PATH = 'dataset/new/new_pollute_spider_dev_original.json'
SPIDER_EMB_PATH = '/data0/gxj/db_emb'

EVAL_MAX_NEW_TOKENS = 128

INSERT_EMBS_TOKEN = '<insert_embs>'
INSERT_EMBS_TOKEN_ID = -114
INSERT_SEP_TOKEN='<insert_sep>'

CONTENT_ID_MAX_LEN = 512

MAX_ROW = 50
MAX_COL = 20

# BASE_MODEL_PATH='/data4/sft_output/qwen2-base-0717/checkpoint-2000'
# BASE_MODEL_PATH='/data4/sft_output/qwen2-base-0817/final'
# BASE_MODEL_PATH="/data3/models/Qwen/Qwen2-7B-Instruct"
# BASE_MODEL_PATH="/data4/sft_output/qwen2-base-general-0903/checkpoint-1800"
BASE_MODEL_PATH="/data3/models/Qwen/Qwen2-7B"
