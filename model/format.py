import pandas as pd
from config import INSERT_SEP_TOKEN, INSERT_EMBS_TOKEN
import random


######################################################## Table info
def get_df_category_desc(query, df):
    return {}

def dataframe_info_raw(df:pd.DataFrame, df_name:str, comments=None):
    """
    根据 dataframe 获取 dataframe description 信息
    :param df: 输入 dataframe
    :param df_name: dataframe name
    :param comments: 列名的备注信息, dict
    :return: 能反馈 dataframe 的信息
    """
    # df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe include the data types, comments, the column values info as follows:\n{desc_info}\n*/"""
    df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe that can be used as follows:\n{desc_info}\n*/"""
    # df_info_template_simple = """/*\n'{df_name}' each column information:\n {desc_info}\n*/"""
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Contains NaN": df.isnull().any(),
        "Is Unique": df.nunique() == len(df)
    }).reset_index(drop=True)

    # 添加 Example Values 列，使用所有唯一值但最多取三个
    example_values = []
    # category_dict = get_df_category_desc(query, df)
    category_dict = {}
    for col in df.columns:
        if col in category_dict.keys():
            col_value = category_dict[col]
        else:
            col_value = df[col].dropna().unique().tolist()
        d_type = str(df[col].dtype)
        
        if len(col_value) > 3:
            if ("float" in d_type):
                col_value = col_value[0:2]
            else:
                col_value = col_value[0:3]
            col_value.append("...")
        
        # 限制值长度
        col_value_limit = [s if not isinstance(s, str) or len(s) <= 80 else s[:80] + "...." for s in col_value]
        # col_value_limit = [s if len(s) <= 50 else s[:50] + "..." for s in col_value]
        example_values.append(col_value_limit)

    info_df['Example Values'] = example_values
    # info_df['Example Values'] = [df[col].dropna().unique()[:3].tolist() for col in df.columns]

    if comments is not None:
        # 将comments转换为一个字典，以便快速查找
        comments_dict = {item["content"]: {"comment": item["comment"], "info": item["info"]} for item in comments}
        # 为每一列添加comment和info信息
        comment_value = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        info_df.insert(4, "Comment", comment_value)

        # info_df['Comment'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        # info_df['Info'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("info", ""))
    
    info_df_new = info_df.set_index('Column Name', drop=True).transpose()
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
        
        examples = value["Example Values"]

        if ("float" in data_type) or ("int" in data_type):
            unique_str = ""

        dil = f"- '{key}' {data_type}, {unique_str}{contains_nan_str}{comment_str} Example Values: {examples}"
        desc_info_lines.append(dil)

    desc_info = "\n".join(desc_info_lines)

    desc_info = desc_info.replace(", '...']", ", ...]")

    df_info = df_info_template_simple.format(
        df_name=df_name,
        desc_info=desc_info,
    )
    
    return df_info

def dataframe_info_simple(df:pd.DataFrame, df_name:str, comments=None):
    """
    根据 dataframe 获取 dataframe description 信息
    :param df: 输入 dataframe
    :param df_name: dataframe name
    :param comments: 列名的备注信息, dict
    :return: 能反馈 dataframe 的信息
    """
    # df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe include the data types, comments, the column values info as follows:\n{desc_info}\n*/"""
    df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe that can be used as follows:\n{desc_info}\n*/"""
    # df_info_template_simple = """/*\n'{df_name}' each column information:\n {desc_info}\n*/"""
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Contains NaN": df.isnull().any(),
        "Is Unique": df.nunique() == len(df)
    }).reset_index(drop=True)

    # 添加 Example Values 列，使用所有唯一值但最多取三个



    if comments is not None:
        # 将comments转换为一个字典，以便快速查找
        comments_dict = {item["content"]: {"comment": item["comment"], "info": item["info"]} for item in comments}
        # 为每一列添加comment和info信息
        comment_value = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        info_df.insert(4, "Comment", comment_value)

        # info_df['Comment'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        # info_df['Info'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("info", ""))
    
    info_df_new = info_df.set_index('Column Name', drop=True).transpose()
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

        dil = f"{INSERT_SEP_TOKEN + INSERT_EMBS_TOKEN + INSERT_SEP_TOKEN} '{key}' {data_type}, {unique_str}{contains_nan_str}{comment_str}"
        desc_info_lines.append(dil)

    desc_info = "\n".join(desc_info_lines)

    desc_info = desc_info.replace(", '...']", ", ...]")

    df_info = df_info_template_simple.format(
        df_name=df_name,
        desc_info=desc_info,
    )
    
    return df_info

def dataframe_info_combined(df:pd.DataFrame, df_name:str, comments=None, selected_values = {}, lower = False):
    """
    根据 dataframe 获取 dataframe description 信息
    :param df: 输入 dataframe
    :param df_name: dataframe name
    :param comments: 列名的备注信息, dict
    :return: 能反馈 dataframe 的信息
    """
    # df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe include the data types, comments, the column values info as follows:\n{desc_info}\n*/"""
    df_info_template_simple = """/*\nDetails about the '{df_name}' dataframe that can be used as follows:\n{desc_info}\n*/"""
    # df_info_template_simple = """/*\n'{df_name}' each column information:\n {desc_info}\n*/"""
    info_df = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Contains NaN": df.isnull().any(),
        "Is Unique": df.nunique() == len(df)
    }).reset_index(drop=True)

    # 添加 Example Values 列，使用所有唯一值但最多取三个
    example_values = []
    for col in df.columns:
        col_value = df[col].dropna().unique().tolist()
        assert(type(col_value) == list)
        if col in selected_values.keys():
            # 删掉对应值
            col_value = [v for v in col_value if v != selected_values[col]]
            
        d_type = str(df[col].dtype)
        
        if len(col_value) > 3:
            if ("float" in d_type):
                col_value = col_value[0:2]
            else:
                col_value = col_value[0:3]
            col_value.append("...")
        
        # 限制值长度
        col_value_limit = [s if not isinstance(s, str) or len(s) <= 80 else s[:80] + "...." for s in col_value]
        # col_value_limit = [s if len(s) <= 50 else s[:50] + "..." for s in col_value]
        example_values.append(col_value_limit)
    info_df['Example Values'] = example_values

    if comments is not None:
        # 将comments转换为一个字典，以便快速查找
        comments_dict = {item["content"]: {"comment": item["comment"], "info": item["info"]} for item in comments}
        # 为每一列添加comment和info信息
        comment_value = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        info_df.insert(4, "Comment", comment_value)

        # info_df['Comment'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("comment", ""))
        # info_df['Info'] = info_df['Column Name'].apply(lambda x: comments_dict.get(x, {}).get("info", ""))
    
    info_df_new = info_df.set_index('Column Name', drop=True).transpose()
    desc_info_dict = info_df_new.to_dict()

    # desc_info_lines = [f"- '{key}': {value}" for key, value in desc_info_dict.items()]
    # desc_info_lines = [f"- '{key}': {value}" for key, value in desc_info_dict.items()]
    desc_info_lines = []
    for key, value in desc_info_dict.items():
        if lower:
            key = key.lower()
            
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
        
        examples = value["Example Values"]
        
        if ("float" in data_type) or ("int" in data_type):
            unique_str = ""

        dil = f"{INSERT_SEP_TOKEN + INSERT_EMBS_TOKEN + INSERT_SEP_TOKEN} '{key}' {data_type}, {unique_str}{contains_nan_str}{comment_str} Example Values: {examples}"
        desc_info_lines.append(dil)

    desc_info = "\n".join(desc_info_lines)

    desc_info = desc_info.replace(", '...']", ", ...]")

    df_info = df_info_template_simple.format(
        df_name=df_name,
        desc_info=desc_info,
    )
    
    return df_info

def dataframe_info_list(info_func, csv_paths: list, df_names: list, comments=None):
    return '\n\n'.join([info_func(pd.read_csv(csv_paths[i]), df_names[i], comments) for i in range(len(csv_paths))])

######################################################## Prompt
def get_text2sql_prompt():
    templates = [
        f"""
You have access to a number of pandas dataframes. Given a user question about the dataframes, write the SQL query to answer it.

## Use the following format:
Question: The user question you need to answer.
SQL Query: Generate the SQL query, wrap it up with ```sql ... ```.

## Here is the information about each dataframe

{{df_info}}

Notes: Only use the tables provided. Write the SQL query alone, in a single string format. Do not include any explanations or context.
Question: {{question}}
SQL Query:

        """,
        f"""
You have several pandas dataframes available. Based on a user query about these dataframes, generate an appropriate SQL query.

## Answer with following format:
SQL Query: Provide the SQL query wrapped in ```sql ... ```.

## DataFrames information:

{{df_info}}

Notes: Do not assume access to any tables other than those given. Your response must be just the SQL query in a single string. Avoid adding any explanations or context.

Question: {{question}}
SQL Query:

        """,
        f"""
Provided with several pandas dataframes, construct an SQL query to answer a user's question about the data.

## Use this format:
Question: The user's question.
SQL Query: Write the SQL query, enclosed in ```sql ... ```.

## Details of each dataframe:

{{df_info}}

Notes: Restrict your SQL query to the provided tables. Submit only the SQL query as a single string without any accompanying explanations or context.

Question: {{question}}
SQL Query:
        """,
        f"""
Given access to multiple pandas dataframes, convert a user question into an SQL query.

## Follow this format:
Question: The user's question.
SQL Query: Generate the SQL query, enclosed in ```sql ... ```.

## Information about the dataframes:

{{df_info}}

Notes: Use only the specified tables for your query. The answer should be a single string containing only the SQL query, without any additional explanations or context.

Question: {{question}}
SQL Query:

""",
        f"""
You can access several pandas dataframes. Based on the user's question, generate the corresponding SQL query.

## Use the following structure:
Question: The user's question.
SQL Query: Formulate the SQL query, wrapped in ```sql ... ```.

## DataFrame details:

{{df_info}}

Notes: Assume no access to tables other than the ones listed. Provide the SQL query alone in a single string format, with no explanations or context included.

Question: {{question}}

SQL Query:
        """
    ]
    choice = random.choice(templates)
    return choice.strip()


def build_text2sql_prompt(*, question:str, csv_paths: list, df_names: list, foreign_keys = None, answer = None):
    
    
    df_info = dataframe_info_list(dataframe_info_combined, csv_paths, df_names)
    if foreign_keys != None and len(foreign_keys) > 0:
        df_info += "\nForeign keys: " + str(foreign_keys) + "\n"
    instruction = get_text2sql_prompt().format(df_info=df_info, question=question)
    if answer is None:
        return instruction
        
    answer = f'''```sql\n{answer}\n```'''
    return instruction, answer
    
def get_nameguess_prompt():
    templates = [
f'''Identify the most appropriate column for each value from the provided pandas dataframes.

## Use this format:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## Information about DataFrames:

{{df_info}}

Notes: Only use the specified columns. Return the column name alone, without any additional context.
Value(s): {{values}}
Column Name(s):
''',
f'''From the given pandas dataframes, find the column that matches each specified value.

## Format to follow:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## Details of the DataFrames:

{{df_info}}

Notes: Restrict your answer to the mentioned columns. Provide the column name only, without additional context or explanation.
Value(s): {{values}}
Column Name(s):
''',
f'''Determine which column each provided value belongs to in the pandas dataframes.

## Expected format:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## DataFrame Description:

{{df_info}}

Notes: Use only the specified columns for your answer. The answer should be the column name as a single string, without any extra information.

## Start!

Value(s): {{values}}
Column Name(s):
''',
f'''Identify the column for each given value from the specified pandas dataframes.

## Format:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## DataFrame Overview:

{{df_info}}

Notes: Only consider the given columns. The response should be the name of the column alone, without further context.
Value(s): {{values}}
Column Name(s):
''',
f'''
For the given pandas dataframes, determine the column that each specified value most likely fits into.

## Required format:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## DataFrame Information:

{{df_info}}

Notes: Limit your response to the provided columns. Return the column name only, as a single word, without any additional explanation.
Value(s): {{values}}
Column Name(s):
''',
f'''
Given multiple pandas dataframes, identify which column a specified value most likely belongs to

## Use this format:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## DataFrame details:

{{df_info}}


Notes: Use only the specified columns. The answer should be the column name as a single string, without any explanations or context.

## Start!

Value(s): {{values}}
Column Name(s):
''',
f'''Identify the most appropriate column for each value from the provided pandas dataframes.

## Use this format:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## Information about DataFrames:

{{df_info}}

Notes: Only use the specified columns. Return the column name alone, without any additional context.

## Start!

Value(s): {{values}}
Column Name(s):
''',
f'''From the given pandas dataframes, find the column that matches each specified value.

## Format to follow:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## Details of the DataFrames:

{{df_info}}

Notes: Restrict your answer to the mentioned columns. Provide the column name only, without additional context or explanation.

Value(s): {{values}}
Column Name(s):
''',
f'''Determine which column each provided value belongs to in the pandas dataframes.

## Expected format:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## DataFrame Description:

{{df_info}}



Notes: Use only the specified columns for your answer. The answer should be the column name as a single string, without any extra information.
Value(s): {{values}}
Column Name(s):
''',
f'''Identify the column for each given value from the specified pandas dataframes.

## Format:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## DataFrame Overview:

{{df_info}}



Notes: Only consider the given columns. The response should be the name of the column alone, without further context.
Value(s): {{values}}
Column Name(s):
''',
f'''
For the given pandas dataframes, determine the column that each specified value most likely fits into.

## Required format:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## DataFrame Information:

{{df_info}}


Notes: Limit your response to the provided columns. Return the column name only, as a single word, without any additional explanation.

Value(s): {{values}}
Column Name(s):
''',
f'''
Given multiple pandas dataframes, identify which column a specified value most likely belongs to.

## Use this format:
Value(s): x1;x2;...;xn
Column Name(s): col1;col2;...;coln

## DataFrame details:

{{df_info}}

Notes: Use only the specified columns. The answer should be the column name as a single string, without any explanations or context.
Value(s): {{values}}
Column Name(s):
'''
]

    return random.choice(templates).strip()
    
def build_nameguess_prompt(*, df, value_str):
    prompt = get_nameguess_prompt()
    df_info = dataframe_info_simple(df, 'df')
    ret = prompt.format(df_info=df_info, values=value_str)
    if ';' not in value_str:
        ret = ret.replace("x1;x2;...;xn", 'x_i')
        ret = ret.replace("col1;col2;...;coln", 'col_name')
    return ret

def get_cellchoose_prompt():
    prompt_sys = [
        """You are an advanced data analysis tool designed to understand and respond to questions about tables and their columns. Your task is to analyze the given table and columns, and then answer which cell value is most likely from the specified column. """,
        """As an advanced data analysis tool, your role is to comprehend and address queries regarding tables and their columns. Analyze the provided table and columns to determine which cell value is most likely to belong to the specified column.""",
        """Designed for advanced data analysis, your task is to interpret and answer questions about tables and columns. Examine the given table and columns to identify which cell value most likely originates from the specified column.""",
        """You are a sophisticated data analysis tool created to analyze and respond to questions concerning tables and their columns. Your task is to review the given table and columns and identify which cell value is most likely from the specified column.""",
        """As an advanced data analysis tool, you are tasked with understanding and answering questions about tables and columns. Analyze the provided table and columns to determine which cell value is most likely from the specified column.""",
        """Your function as an advanced data analysis tool is to interpret and respond to questions about tables and columns. Examine the provided table and columns to find out which cell value is most likely to belong to the specified column.""",
        """Acting as an advanced data analysis tool, your mission is to analyze and respond to inquiries about tables and columns. Evaluate the provided table and columns to identify the cell value most likely from the specified column.""",
        """As a highly advanced data analysis tool, you are to understand and answer questions related to tables and columns. Review the given table and columns to determine which cell value is most likely from the specified column.""",
        """You are a top-tier data analysis tool designed to understand and answer questions regarding tables and their columns. Your task is to analyze the given table and columns and determine which cell value is most likely from the specified column.""",
        """Functioning as an advanced data analysis tool, your task is to interpret and answer questions about tables and columns. Examine the provided table and columns to identify the cell value that is most likely from the specified column."""
    ]

    prompt_user = [
"""Table information:
{table_infos}

The cell values are:
{cell_values}

Question: Among the given cell values, which is most likely from column {specified_column}? Please only answer the cell value that is most likely from the specified column.

Answer:""",
"""Table information:
{table_infos}

The cell values are:
{cell_values}

Question: Out of the given cell values, which one is most probably from column {specified_column}? Only respond with the cell value that fits best with the specified column.

Answer:""",
"""Table information:
{table_infos}

The cell values are:
{cell_values}

Question: Which cell value among the provided ones is most likely from column {specified_column}? Please respond solely with the cell value that most likely belongs to the specified column.

Answer:""",
"""Table information:
{table_infos}

The cell values are:
{cell_values}

Question: Among the provided cell values, which is most likely from column {specified_column}? Provide only the cell value that is most likely associated with the specified column.

Answer:""",
"""Table information:
{table_infos}

The cell values are:
{cell_values}

Question: From the given cell values, which one is most likely from column {specified_column}? Answer only with the cell value that best matches the specified column.

Answer:""",
"""Table information:
{table_infos}

The cell values are:
{cell_values}

Question: Which cell value among the given ones is most likely from column {specified_column}? Answer solely with the most likely cell value from the specified column.

Answer:""",
"""Table information:
{table_infos}

The cell values are:
{cell_values}

Question: Among the given cell values, which is most probably from column {specified_column}? Please respond only with the cell value most likely associated with the specified column.

Answer:""",
"""Table information:
{table_infos}

The cell values are:
{cell_values}

Question: Out of the provided cell values, which one is most likely from column {specified_column}? Only provide the cell value that most likely fits the specified column.

Answer:""",
"""Table information:
{table_infos}

The cell values are:
{cell_values}

Question: From the given cell values, which is most likely from column {specified_column}? Provide only the most likely cell value from the specified column.

Answer:""",
"""Table information:
{table_infos}

The cell values are:
{cell_values}

Question: Which cell value among the provided ones is most probably from column {specified_column}? Respond solely with the cell value that is most likely from the specified column.

Answer:""",
    ]
    return random.choice(prompt_sys) + '\n' + random.choice(prompt_user)

def build_cellchoose_prompt(*, df, specified_column, cell_values):
    table_infos = dataframe_info_simple(df, 'df')
    return get_cellchoose_prompt().format(table_infos=table_infos, specified_column=specified_column, cell_values=cell_values)

def get_recall_prompt():
    prompt_sys_list = [
    "You are a specialized system designed to analyze and understand tabular data structures. Your main function is to evaluate the tables and columns provided and determine which ones are most relevant to the user's query.",
    "You are an intelligent data analysis system tasked with interpreting tables and their columns. Your objective is to select the most pertinent tables and columns that best relate to the user's question.",
    "You are a highly advanced data interpretation tool. Your purpose is to analyze tables and columns, identifying the most relevant tables and columns to the user's query.",
    "You are a sophisticated system for analyzing tabular data. Your role is to carefully examine the given tables and columns and choose those most relevant to the user's question.",
    "You are a powerful data analysis tool focused on understanding tabular data. Your task is to review the tables and columns provided and select the tables and columns most relevant to the user's query.",
    "You are an expert system for analyzing data tables. \n Your mission is to assess the provided tables and columns to determine the most relevant tables and columns based on the user's question.",
    "You are an advanced tabular data analysis engine. Your goal is to examine the given tables and columns and identify the most relevant tables and columns to the user's query.",
    '''You are an advanced data analysis tool designed to understand and respond to queries about tables and their columns. \nYour task is to analyze the given tables and columns, and choose most relevant tables and columns to the user's query.''',
    "Immediately analyze the tabular data and identify the tables and columns most relevant to the user's query.",
    "Quickly evaluate the provided tables and columns, and select only the tables and columns precisely related to the user's question.",
    "Thoroughly examine the tables and columns, and pinpoint the tables and columns most critical to the user's query.",
]
    prompt_user = """table:
{table_infos}

Notion: 
Answer in Format: tables is: ['table_name1','table_name2',...];columns is: ['table_name1.column_name1','table_name2.column_name2',...]

Question: {query}

Answer:"""
    return random.choice(prompt_sys_list) + '\n' + prompt_user

def get_pythongen_prompt():
    prompt_sys_list = [
    '''You have access to multiple pandas dataframes. Your task is to write the Python code to answer the user's query.\n\n## Follow this format:\nQuestion: The user's query.\nThought: Analyze the dataframes and the question to come up with the appropriate solution.\nPython code: Provide the Python code needed to address the user's question, , within ```python ... ```.\n\n## Details about the dataframes:\n\n''',    
    
    '''Given several pandas dataframes, your task is to create the Python code that will solve the user's query.\n\n## Follow this format:\nQuestion: The user's query.\nThought: Assess the dataframes and the query to determine the correct solution.\nPython code: Write the Python code that addresses the user's request, , within ```python ... ```.\n\n## Details about the dataframes:\n\n''',    
    
    '''You are provided with multiple pandas dataframes. Your objective is to develop the Python code that will resolve the user's question.\n\n## Follow this format:\nQuestion: The user's query.\nThought: Review the dataframes and the question to decide on the solution.\nPython code: Write the Python code necessary to answer the user's query, , within ```python ... ```.\n\n## Details about the dataframes:\n\n''',    
    
    '''Several pandas dataframes are available for you. Your task is to generate the Python code that will answer the user's question.\n\n## Follow this format:\nQuestion: The user's query.\nThought: Consider the dataframes and the question to formulate the solution.\nPython code: Produce the Python code that solves the user's problem, , within ```python ... ```.\n\n## Details about the dataframes:\n\n''',    
    
    '''With access to multiple pandas dataframes, your task is to construct the Python code needed to answer the user's query.\n\n## Follow this format:\nQuestion: The user's query.\nThought: Examine the dataframes and the query to determine the appropriate solution.\nPython code: Write the Python code that fulfills the user's request, , within ```python ... ```.\n\n## Details about the dataframes:\n\n''',    
        '''You are working with several pandas dataframes. Your goal is to write the Python code that will correctly address the user's query.\n\n## Follow this format:\nQuestion: The user's query.\nThought: Evaluate the dataframes and the question to identify the correct solution.\nPython code: Generate the Python code needed to solve the user's problem, within ```python ... ```.\n\n## Details about the dataframes:\n\n''', 
    
    '''Multiple pandas dataframes are provided to you. Your task is to craft the Python code that will solve the user's question.\n\n## Follow this format:\nQuestion: The user's query.\nThought: Analyze the dataframes and the user's question to come up with the appropriate solution.\nPython code: Create the Python code needed to answer the user's query, within ```python ... ```.\n\n## Details about the dataframes:\n\n''', 
    
    '''You have been given several pandas dataframes. Your task is to produce the Python code that will answer the user's question.\n\n## Follow this format:\nQuestion: The user's query.\nThought: Examine the dataframes and the question to derive the appropriate solution.\nPython code: Write the Python code that addresses the user's question, within ```python ... ```.\n\n## Details about the dataframes:\n\n''',
]
    return random.choice(prompt_sys_list) + '\n{table_infos}' + '''\n\nQuestion: {query}\nThought:''' 

def build_pythongen_prompt(*, query:str, csv_paths: list, df_names: list):
    df_info = dataframe_info_list(dataframe_info_combined, csv_paths, df_names)
    instruction = get_pythongen_prompt().format(table_infos=df_info, query=query)
    return instruction

def build_instruction(prompt, tokenizer):
    """
    Apply the chat template to the user prompt

    Args:
        prompt (str): The user prompt.
        tokenizer: The tokenizer object.

    Returns:
        str: The instruction text.
    """
    messages = [
        {"role": "user", "content": prompt}
    ]
    decoder_input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return decoder_input_text