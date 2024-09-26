import copy
import random, tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
from transformers import Trainer
from datasets import load_dataset
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch
from torch import nn
import transformers
import tokenizers
import numpy as np
from torch.utils.data import Dataset
from Table_Encoder.model.load_encoder import load_encoder
from model.model_sft import Model
from model.multihead_projector import MultiHeadProjector
from model.utils import build_instruction_qwen

from model.utils import (
    find_correct_case_file_name,
    build_plain_instruction_prompt,
    tokenize_insert,
)
from config import (
    SPIDER_CSV_PATH,
    INSERT_EMBS_TOKEN,
    INSERT_EMBS_TOKEN_ID,
    INSERT_SEP_TOKEN,
    SENTENCE_TRANSFORMER_PATH,
)

IGNORE_INDEX = -100
from PIL import Image

# EOT_TOKEN = "<|EOT|>"


@dataclass
class ModelArguments:
    load_pretrained: bool = field(default=False)
    pretrained_path: str = field(default=None)
    decoder_path: str = field(
        default=None, metadata={"help": "Path of pretrained decoder"}
    )
    encoder_path: str = field(
        default=None, metadata={"help": "Path of pretrained encoder"}
    )
    projector_path: str = field(default=None)
    device: str = field(default="cuda")

    projector_type: str = field(default="mlp2x_gelu")
    encoder_hidden_size: int = field(default=3584)
    decoder_hidden_size: int = field(default=3584)
    projector_num_heads: int = field(default=1)
    torch_dtype: str = field(default="float32")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    data_count: int = field(default=1000000)
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    # seed: int = field(default = 1926)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    gradient_checkpointing: bool = (field(default=False),)
    cache_dir: Optional[str] = field(default="/data4/sft_output/.cache_dir132")
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_projector: bool = field(default=False)
    freeze_encoder: bool = field(default=False)
    freeze_sp: bool = field(default=False)
    freeze_decoder: bool = field(default=True)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _tokenize_fn(
    strings: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    insert_embs=False,
) -> Dict:
    """Tokenize a list of strings."""
    if not insert_embs:
        tokenized_list = [
            tokenizer(
                prompt,
                return_tensors="pt",
                padding="longest",  # 这个padding有用吗？
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            for prompt in strings
        ]

        input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
    else:
        tokenized_list = [tokenize_insert(prompt, tokenizer) for prompt in strings]
        # tokenized_len = [t.shape[0] for t in tokenized_list]
        # print('tokenized_len', tokenized_len)
        input_ids = labels = tokenized_list
        input_ids_lens = labels_lens = [
            tokenized.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    insert_embs: bool = False,
) -> Dict:
    """
    Preprocess the data by tokenizing.
    Parameters:
        - sources: instructions
        - targets: outputs
    Returns:
        - input_ids: tokenized instructions
        - labels: tokenized outputs, padded with IGNORE_INDEX of the same length as the input_ids
    """

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        _tokenize_fn(strings, tokenizer, insert_embs=insert_embs)
        for strings in (examples, sources)
    ]  # 把 sources 和 examples 都tokenize了
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        # print('labels shape', labels.shape)
        # path_csv = [instance['path'] for instance in instances]
        path_csv = [instance["path_csv"] for instance in instances]

        insert_embs = [instance.get("insert_embs", False) for instance in instances]

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            path_csv=path_csv,
            insert_embs=insert_embs,
        )


def train_tokenize_function(examples, tokenizer):
    # conv的字段：instruction, answer, path_csv

    # sources = [
    #     build_instruction_prompt_with_context(question=cur_q, context=cur_c, tokenizer=tokenizer, max_context_length=512)
    #     for cur_q, cur_c in zip(examples['question'], examples['context'])
    # ]

    # sources = [
    #     build_instruction_qwen(cur_q, tokenizer=tokenizer)
    #     for cur_q in examples['instruction']
    # ]
    sources = [
        build_instruction_qwen(cur_q, cur_h, tokenizer=tokenizer)
        for cur_q, cur_h in zip(examples["instruction"], examples["history"])
    ]

    EOT_TOKEN = "<|im_end|>"
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples["answer"]]

    is_insert = True
    data_dict = preprocess(sources, targets, tokenizer, insert_embs=is_insert)
    data_dict["path_csv"] = examples["path_csvs"]

    data_dict["insert_embs"] = [is_insert] * len(data_dict["input_ids"])
    return data_dict


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.torch_dtype = eval("torch." + model_args.torch_dtype)

    if training_args.local_rank == 0:
        print("=" * 100)
        # print(training_args)

    if model_args.load_pretrained:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.pretrained_path
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/data0/pretrained-models/Qwen2-7B",
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True,
        )
        tokenizer.add_special_tokens({"additional_special_tokens": ["<insert_sep>"]})

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    encoder_tokenizer = transformers.AutoTokenizer.from_pretrained(
        SENTENCE_TRANSFORMER_PATH
    )

    if training_args.local_rank == 0:
        print("Load tokenizer from {} over.".format(model_args.decoder_path))
    if model_args.load_pretrained == True:
        print("load pretrained model")
        model = Model.from_pretrained(model_args.pretrained_path).to(
            training_args.device, dtype=torch.bfloat16
        )
    else:
        raise ValueError("not support")

    if training_args.freeze_decoder:
        model.decoder.requires_grad_(False)
        model.decoder.eval()
        print("freeze decoder")
    else:
        model.decoder.requires_grad_(True)
        model.decoder.train()

    if training_args.freeze_encoder:
        model.encoder.eval()
        model.encoder.requires_grad_(False)

    else:
        model.encoder.requires_grad_(True)
        model.encoder.train()

    if training_args.freeze_projector:
        model.projector.requires_grad_(False)
    else:
        model.encoder.qformer.requires_grad_(True)
        model.projector.requires_grad_(True)
        model.encoder.qformer.train()
    for param in model.decoder.parameters():
        assert param.requires_grad != training_args.freeze_decoder

    if training_args.local_rank == 0:
        print("model", model)
        print("qformer query num: ", model.encoder.qformer.query_num)
        # print('device', get_device(model.decoder))
        # print(get_device(model.projector.model))
        # print('dtype', next(model.decoder.parameters()).dtype)

    raw_train_datasets = load_dataset(
        "json",
        data_files=data_args.data_path,
        cache_dir="/data4/sft_output/.cache_dir132",
        # split=f"train[246240:]" # 1032 * 60 + 1024 * 80 + 1024 * 100
        split="train",
    )
    raw_train_datasets1 = load_dataset(
        "json",
        data_files="/data4/code822/tableqa/general.json",
        cache_dir="/data4/sft_output/.cache_dir132",
        split="train",
    )
    raw_train_datasets2 = load_dataset(
        "json",
        data_files="/data4/code822/tableqa/table.json",
        cache_dir="/data4/sft_output/.cache_dir132",
        split="train",
    )
    # raw_train_dataset = raw_train_dataset.shuffle(seed = data_args.seed)

    raw_eval_datasets = load_dataset(
        "json",
        data_files=data_args.eval_data_path,
        cache_dir="/data4/sft_output/.cache_dir132",
        split="train",
    )

    if training_args.local_rank > 0:
        torch.distributed.barrier()
    train_dataset = raw_train_datasets.map(
        train_tokenize_function,
        batched=True,
        # batch_size=3000,
        num_proc=64,
        remove_columns=raw_train_datasets.column_names,
        load_from_cache_file=True,  # not args.overwrite_cache
        desc="Running Encoding train_dataset",
        fn_kwargs={"tokenizer": tokenizer},
    )
    train_dataset1 = raw_train_datasets1.map(
        train_tokenize_function,
        batched=True,
        # batch_size=3000,
        num_proc=64,
        remove_columns=raw_train_datasets1.column_names,
        load_from_cache_file=True,  # not args.overwrite_cache
        desc="Running Encoding train_dataset1",
        fn_kwargs={"tokenizer": tokenizer},
    )
    train_dataset2 = raw_train_datasets2.map(
        train_tokenize_function,
        batched=True,
        # batch_size=3000,
        num_proc=64,
        remove_columns=raw_train_datasets2.column_names,
        load_from_cache_file=True,  # not args.overwrite_cache
        desc="Running Encoding train_dataset2",
        fn_kwargs={"tokenizer": tokenizer},
    )
    eval_dataset = raw_eval_datasets.map(
        train_tokenize_function,
        batched=True,
        # batch_size=3000,
        num_proc=64,
        remove_columns=raw_eval_datasets.column_names,
        # load_from_cache_file=True, # not args.overwrite_cache
        desc="Running Encoding eval_dataset",
        fn_kwargs={"tokenizer": tokenizer},
    )

    if training_args.local_rank == 0 and training_args.world_size > 1:
        torch.distributed.barrier()

    if training_args.local_rank == 0:
        print("Training dataset samples:", len(train_dataset))
        for index in random.sample(range(len(train_dataset)), 3):
            print(
                f"Sample {index} of the training set: {len(train_dataset[index]['input_ids'])}, {train_dataset[index]['labels']}."
            )
            # print(f"Sample {index} of the training set: {tokenizer.decode(list(train_dataset[index]['input_ids']))}.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    data_module = dict(
        train_dataset=train_dataset,
        train_dataset1=train_dataset1,
        train_dataset2=train_dataset2,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    if training_args.local_rank == 0:
        try:
            projector = model.projector.cpu()
            torch.save(
                projector.state_dict(),
                os.path.join(training_args.output_dir, "projector.bin"),
            )
            print("save projector")
        except Exception as e:
            print("fail to save projector", e)

        try:
            encoder = model.encoder.cpu()
            torch.save(
                encoder.state_dict(),
                os.path.join(training_args.output_dir, "encoder.bin"),
            )
            print("save encoder")
        except Exception as e:
            print("fail to save encoder", e)

        try:
            if training_args.freeze_decoder == False:
                decoder = model.decoder.cpu()
                torch.save(
                    decoder.state_dict(),
                    os.path.join(training_args.output_dir, "decoder.bin"),
                )
                print("save decoder")
        except:
            print("fail to save decoder")


if __name__ == "__main__":
    import wandb

    # KEY = 'e994e80ec6814daf51756ffe1706be8d51e71eb5'
    # wandb.login(key=KEY)
    api_key = "904d9b7af10d7a2ea3f3d1be89703a53c20deb47"
    os.environ["WANDB_PROJECT"] = "table-sft"
    wandb.login(key=api_key)
    train()
