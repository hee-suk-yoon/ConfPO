import logging
from typing import Callable, Literal, Optional, Union

from datasets import Dataset, Value
from transformers import AutoTokenizer

from ..trainer.utils import ConstantLengthDataset


FORMAT_MAPPING = {
    "chatml": [{"content": Value(dtype="string", id=None), "role": Value(dtype="string", id=None)}],
    "instruction": {"completion": Value(dtype="string", id=None), "prompt": Value(dtype="string", id=None)},
}

CHAT_TEMPLATE = {
     "llama-2": "{% for message in messages %}{% if (message['role'] == 'user') %}{{'[INST]' + message['content'] + '</s>'  + '[/INST]'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '</s>'}}{% endif %}{% endfor %}",

     "llama-3": "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",

} 


def conversations_formatting_function(tokenizer: AutoTokenizer, messages_field: Literal["messages", "conversations"]):
    r"""
    return a callable function that takes in a "messages" dataset and returns a formatted dataset, based on the tokenizer
    apply chat template to the dataset
    """
    
    def format_dataset(examples):
        if isinstance(examples[messages_field][0], list):
            output_texts = []
            for i in range(len(examples[messages_field])):
                output_texts.append(tokenizer.apply_chat_template(examples[messages_field][i], tokenize=False))
            return output_texts
        else:
            return tokenizer.apply_chat_template(examples[messages_field], tokenize=False)

    return format_dataset


def instructions_formatting_function(tokenizer: AutoTokenizer):
    r"""
    return a callable function that takes in an "instructions" dataset and returns a formatted dataset, based on the tokenizer
    apply chat template to the dataset
    """

    def format_dataset(examples):
        if isinstance(examples["prompt"], list):
            output_texts = []
            for i in range(len(examples["prompt"])):
                converted_sample = [
                    {"role": "user", "content": examples["prompt"][i]},
                    {"role": "assistant", "content": examples["completion"][i]},
                ]
                output_texts.append(tokenizer.apply_chat_template(converted_sample, tokenize=False))
            return output_texts
        else:
            converted_sample = [
                {"role": "user", "content": examples["prompt"]},
                {"role": "assistant", "content": examples["completion"]},
            ]
            return tokenizer.apply_chat_template(converted_sample, tokenize=False)

    return format_dataset


def get_formatting_func_from_dataset(
    dataset: Union[Dataset, ConstantLengthDataset], tokenizer: AutoTokenizer
) -> Optional[Callable]:
    r"""
    Finds the correct formatting function based on the dataset structure. Currently supported datasets are:
    - `ChatML` with [{"role": str, "content": str}]
    - `instruction` with [{"prompt": str, "completion": str}]

    Args:
        dataset (Dataset): User dataset
        tokenizer (AutoTokenizer): Tokenizer used for formatting

    Returns:
        Callable: Formatting function if the dataset format is supported else None
    """
    if isinstance(dataset, Dataset):
        if "messages" in dataset.features:
            if dataset.features["messages"] == FORMAT_MAPPING["chatml"]:
                logging.info("Formatting dataset with chatml format")
                return conversations_formatting_function(tokenizer, "messages")
        if "conversations" in dataset.features:
            if dataset.features["conversations"] == FORMAT_MAPPING["chatml"]:
                logging.info("Formatting dataset with chatml format")
                return conversations_formatting_function(tokenizer, "conversations")
        elif dataset.features == FORMAT_MAPPING["instruction"]:
            logging.info("Formatting dataset with instruction format")
            return instructions_formatting_function(tokenizer)

    return None

# added by esyoon 2024-06-19-16:26:18
def get_hhrlhf_formatting_func_from_dataset(
    dataset: Union[Dataset, ConstantLengthDataset], tokenizer: AutoTokenizer, template_type: str = "llama-2",
) -> Optional[Callable]:

    r"""
    return a callable function that takes in an "instructions" dataset and returns a formatted dataset, based on the tokenizer
    apply chat template to the dataset
    """
    tokenizer.chat_template = CHAT_TEMPLATE[template_type]
    def format_dataset(examples):
        if isinstance(examples["prompt"], list):
            output_texts = []
            for i in range(len(examples["prompt"])):
                import ipdb; ipdb.set_trace()
                converted_sample = process_prompt(examples["prompt"][i])
                # converted_sample = [
                #     {"role": "user", "content": process_prompt(examples["prompt"][i])},
                #     {"role": "assistant", "content": examples["chosen"][i]},
                # ]
                converted_sample.append(
                    {"role": "assistant", "content": examples["chosen"][i]}
                    )
                try:
                    output_texts.append(tokenizer.apply_chat_template(converted_sample, tokenize=False))
                except:
                    continue
            return output_texts
        else:
            converted_sample = [
                {"role": "user", "content": examples["prompt"]},
                {"role": "assistant", "content": examples["chosen"]},
            ]
            return tokenizer.apply_chat_template(converted_sample, tokenize=False)


    def process_prompt(prompt):
        prompt_list = []
        converted_prompt = []
        split_1 = prompt.split('\n\nHuman:')
        for s in split_1:
            prompt_list += s.split('\n\nAssistant:')
        prompt_list = [p for p in prompt_list if not p == '']
        for i, p in enumerate(prompt_list):
            if i % 2 == 0:
                converted_prompt.append(
                    {"role": "user", "content": p}
                )
            else:
                converted_prompt.append(
                    {"role": "assistant", "content": p}
                )
                
        return converted_prompt

    return format_dataset
