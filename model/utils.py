import os, re
import transformers
from typing import Dict
from config import INSERT_EMBS_TOKEN, INSERT_EMBS_TOKEN_ID
import torch
import pandas as pd


def find_correct_case_file_name(path, name):
    ls = os.listdir(path)
    ls = [x.split(".")[0] for x in ls]
    for gt in ls:
        if gt.lower() == name.lower():
            return gt
    # 找为子串的
    for gt in ls:
        if name.lower() in gt.lower():
            return gt
    raise ValueError(f'path {path}, name "{name}" not found')


# def build_instruction_prompt(instruction: str):
#     return '''
# You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
# ### Instruction:
# {}
# ### Response:
# '''.format(instruction.strip()).lstrip()


# def build_instruction_prompt_with_context(*, question: str, context: str, max_context_length=512, tokenizer):

#     context_list = context.split('INSERT')
#     new_context = context_list[0]
#     context_list = context_list[1:]
#     for i, c in enumerate(context_list):
#         c = 'INSERT' + c
#         if len(tokenizer(new_context + c).input_ids) > max_context_length:
#             break

#         new_context += c
#     context = new_context

#     res = '''You have access to a number of SQL tables. Given a user question about the data, write the SQL query to answer it.
# Notes: Don't assume you have access to any tables other than the ones provided. You MUST only write the SQL query, nothing else, in the format of a single string.
# You MUST only write the SQL query, nothing else, in the format of a single string, like 'SELECT a FROM b WHERE c'. You MUST NOT include any explanation or context in the answer.
# Only the provided tables can be used in the SQL query:
# {tables}
# Question:
# {question}
# '''.format(tables=context, question=question)
#     return build_instruction_prompt(res)


def build_plain_instruction_prompt(instruction: str):
    return """
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{}
### Response:
""".format(
        instruction.strip()
    ).lstrip()


def build_instruction_prompt(question, content):

    decoder_input_text = f"""
You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction: 
You have access to a number of SQL tables. Given a user question about the data, write the SQL query to answer it.
Notes: Don't assume you have access to any tables other than the ones provided. You MUST only write the SQL query, nothing else, in the format of a single string.
You MUST only write the SQL query, nothing else, in the format of a single string, like 'SELECT count(*) FROM head WHERE val > 114514'. You MUST NOT include any explanation or context in the answer.
Only the provided tables can be used in the SQL query.
### Table Information: 
{content}
### Question: 
{question}
### Response:
"""

    return decoder_input_text


def clean_sql_output(output: str):
    # raw_output = output
    # if '```sql' in output:
    #     output = output.split('```sql')[1].split('```')[0].strip()
    # output
    # # assert '```' not in output, raw_output
    # return output
    # 正则表达式模式，匹配以```sql开头```结尾的内容（不包括开头结尾），主要考虑不微调的情况
    pattern = r"```sql\s+(.*?)\s+```"

    # 查找第一个匹配的内容
    match = re.search(pattern, output, re.DOTALL)

    if match:
        # 提取匹配到的内容
        sql_content = match.group(1)

        # 替换\t和\n为单个空格
        processed_text = re.sub(r"[\t\n]", " ", sql_content)
        # 删除首尾空格
        processed_text = processed_text.strip()
        # 多个空格替换为一个空格
        processed_text = re.sub(r"\s+", " ", processed_text)

        return processed_text

    else:
        return output  # 如果没有匹配到内容，返回原本字符串


def generate_create_statements(db_id, table_name) -> str:
    from config import SPIDER_DB_PATH

    db_path = os.path.join(SPIDER_DB_PATH, db_id, f"{db_id}.sqlite")
    import sqlite3

    """
    Fetch all table schemas from the given SQLite database and return a dictionary with table names as keys
    and their respective CREATE TABLE schemas as values.

    :param db_path: Path to the SQLite database file
    :return: Dictionary containing table names and their respective CREATE TABLE schemas
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Fetch all table names and their CREATE TABLE statements from the sqlite_master table
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Initialize a dictionary to store table schemas
    table_schemas = {}

    # Iterate over all tables and store their CREATE TABLE schemas
    ret = None
    for table in tables:
        cur_table_name, create_table_sql = table
        if cur_table_name == table_name:
            ret = create_table_sql
            break

    # Close the connection
    conn.close()

    if ret == None:
        raise ValueError(f"Table {table_name} not found in database {db_id}")

    return ret


def generate_all_create_statements(db_id):
    """
    Fetch all table schemas from the given SQLite database and return a dictionary with table names as keys
    and their respective CREATE TABLE schemas as values.

    :param db_path: Path to the SQLite database file
    :return: Dictionary containing table names and their respective CREATE TABLE schemas
    """
    import sqlite3

    # Connect to the SQLite database
    from config import SPIDER_DB_PATH

    db_path = os.path.join(SPIDER_DB_PATH, db_id, f"{db_id}.sqlite")
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda x: x.decode("utf-8", "ignore")
    cursor = conn.cursor()

    # Fetch all table names and their CREATE TABLE statements from the sqlite_master table
    cursor.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()

    # Initialize a dictionary to store table schemas
    table_schemas = {}

    # Iterate over all tables and store their CREATE TABLE schemas
    for table in tables:
        table_name, create_table_sql = table
        table_schemas[table_name] = create_table_sql

    # Close the connection
    conn.close()

    return table_schemas


def get_column_cnt(db_id, table_name, max_rows=512):
    import sqlite3
    from config import SPIDER_DB_PATH

    db_path = os.path.join(SPIDER_DB_PATH, db_id, f"{db_id}.sqlite")
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda x: x.decode("utf-8", "ignore")
    cursor = conn.cursor()

    # Fetch column names for the table
    cursor.execute(f"PRAGMA table_info({table_name});")
    column_count = len(cursor.fetchall())
    return column_count


def generate_insert_statements(db_id, table_name, max_rows=512):
    """
    Generate INSERT statements for all rows in the specified table.

    :param db_path: Path to the SQLite database file
    :param table_name: Name of the table to generate INSERT statements for
    :return: List of INSERT statements
    """
    # Connect to the SQLite database
    import sqlite3
    from config import SPIDER_DB_PATH

    db_path = os.path.join(SPIDER_DB_PATH, db_id, f"{db_id}.sqlite")
    conn = sqlite3.connect(db_path)
    conn.text_factory = lambda x: x.decode("utf-8", "ignore")
    cursor = conn.cursor()

    # Fetch column names for the table
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [col[1] for col in cursor.fetchall()]

    # Fetch all rows from the table
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()

    # Generate INSERT statements
    insert_statements = []
    import random

    random.shuffle(rows)
    for row in rows:
        row = [
            str(value).replace("'", "''") if isinstance(value, str) else str(value)
            for value in row
        ]
        row = [f"'{value}'" for value in row]
        values = ", ".join(row)
        insert_statement = (
            f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({values});"
        )
        insert_statements.append(insert_statement)

    # Close the connection
    conn.close()

    return insert_statements


def tokenize_insert(prompt: str, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """
    Tokenizes the input prompt by inserting a separator token between each chunk of text.

    Args:
        prompt (str): The input prompt to be tokenized. It contains one or more instances of the INSERT_EMBS_TOKEN.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer object used for tokenization.

    Returns:
        torch.Tensor: The tokenized input prompt as a tensor of input IDs. You need to move to the correct device before using it.

    """
    prompt_chunks = [
        tokenizer(
            e, padding="longest", max_length=tokenizer.model_max_length, truncation=True
        ).input_ids
        for e in prompt.split(INSERT_EMBS_TOKEN)
    ]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):  # tokenizer会在每次encode前面都加一个bos_token_id
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(
        prompt_chunks, [INSERT_EMBS_TOKEN_ID] * (offset + 1)
    ):  # insert separator 返回的是 [chunk1, [sep] * (offset + 1), chunk2, [sep] * (offset + 2), ...]，然后用offset统一减掉一个
        input_ids.extend(x[offset:])
    return torch.tensor(input_ids, dtype=torch.long)


def ray_work(func, data, num_gpus, num_gpus_per_worker, devices):
    import ray

    NUM_GPUS = num_gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    NUM_GPUS_PER_WORKER = num_gpus_per_worker
    NUM_PROCESSES = int(NUM_GPUS // NUM_GPUS_PER_WORKER)
    print(
        f"NUM_GPUS: {NUM_GPUS}, NUM_GPUS_PER_WORKER: {NUM_GPUS_PER_WORKER}, NUM_PROCESSES: {NUM_PROCESSES}"
    )

    ray.shutdown()
    ray.init()
    CHUNK_SIZE = len(data) // NUM_PROCESSES + 1
    get_answers_func = ray.remote(num_gpus=NUM_GPUS_PER_WORKER)(
        func,
    ).remote
    cur_data = [
        data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE] for i in range(NUM_PROCESSES)
    ]
    print(len(cur_data))
    futures = [get_answers_func(tt_data) for tt_data in cur_data]
    ret = ray.get(futures)
    ray.shutdown()
    ret = [r for r in ret if r for r in r]
    return ret


def process_pool_work(func, data, num_workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    CHUNK_SIZE = len(data) // num_workers + 1
    # data_split = [data[i::num_workers] for i in range(num_workers)]
    data_split = [
        data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE] for i in range(num_workers)
    ]

    # data_split = [data[i::num_workers] for i in range(num_workers)]
    ret = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(func, data) for data in data_split]
        for future in as_completed(futures):
            ret.extend(future.result())
    return ret


def process_pool_work_2(func, data, num_workers):
    import multiprocessing

    CHUNK_SIZE = len(data) // num_workers + 1
    # data_split = [data[i::num_workers] for i in range(num_workers)]
    data_split = [
        data[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE] for i in range(num_workers)
    ]

    # Create a multiprocessing Pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(func, data_split)

    # Combine the results
    ret = []
    for result in results:
        ret.extend(result)
    return ret


def build_instruction_qwen(prompt, history, tokenizer):
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for his in history:
        messages.append({"role": "user", "content": his[0]})
        messages.append({"role": "assistant", "content": his[1]})
    messages.append({"role": "user", "content": prompt})
    decoder_input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return decoder_input_text


def dataframe_info_simple(df: pd.DataFrame, df_name: str, comments=None):
    """
    根据 dataframe 获取 dataframe description 信息
    :param df: 输入 dataframe
    :param df_name: dataframe name
    :param comments: 列名的备注信息, dict
    :return: 能反馈 dataframe 的信息
    """
    from config import INSERT_SEP_TOKEN, INSERT_EMBS_TOKEN

    # df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe include the data types, comments, the column values info as follows:\n{desc_info}\n*/"""
    df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe that can be used as follows:\n{desc_info}\n*/"""
    # df_info_template_simple = """/*\n'{df_name}' each column information:\n {desc_info}\n*/"""
    info_df = pd.DataFrame(
        {
            "Column Name": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Contains NaN": df.isnull().any(),
            "Is Unique": df.nunique() == len(df),
        }
    ).reset_index(drop=True)

    # 添加 Example Values 列，使用所有唯一值但最多取三个

    if comments is not None:
        # 将comments转换为一个字典，以便快速查找
        comments_dict = {
            item["content"]: {"comment": item["comment"], "info": item["info"]}
            for item in comments
        }
        # 为每一列添加comment和info信息
        comment_value = info_df["Column Name"].apply(
            lambda x: comments_dict.get(x, {}).get("comment", "")
        )
        info_df.insert(4, "Comment", comment_value)

        # info_df['Comment'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        # info_df['Info'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("info", ""))

    info_df_new = info_df.set_index("Column Name", drop=True).transpose()
    desc_info_dict = info_df_new.to_dict()

    # desc_info_lines = [f"- '{key}': {value}" for key, value in desc_info_dict.items()]
    # desc_info_lines = [f"- '{key}': {value}" for key, value in desc_info_dict.items()]
    desc_info_lines = []
    for key, value in desc_info_dict.items():
        comment = value.get("Comment", "")
        if comment:
            comment_str = "means " + comment + "."
        else:
            comment_str = ""

        data_type = value["Data Type"]

        contains_nan = value["Contains NaN"]
        if contains_nan:
            contains_nan_str = "contains NaN, "
        else:
            contains_nan_str = ""

        is_unique = value["Is Unique"]
        if is_unique:
            unique_str = "is unique, "
        else:
            unique_str = ""
            # unique_str = "is not unique, "

        if ("float" in data_type) or ("int" in data_type):
            unique_str = ""

        dil = f"- '{key}' {data_type}, {unique_str}{contains_nan_str}{comment_str} Example Values: {INSERT_SEP_TOKEN + INSERT_EMBS_TOKEN + INSERT_SEP_TOKEN}"
        desc_info_lines.append(dil)

    desc_info = "\n".join(desc_info_lines)

    desc_info = desc_info.replace(", '...']", ", ...]")

    df_info = df_info_template_simple.format(
        df_name=df_name,
        desc_info=desc_info,
    )

    return df_info
