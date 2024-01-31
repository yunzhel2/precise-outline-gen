import random
import re
import json
import openai
import backoff
import logging
import os
import pandas as pd
import tiktoken
import argparse
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default='YOUR_API_KEY', type=str)
    parser.add_argument('--model', default='gpt-3.5-turbo',
                        choices=['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'lmsys/vicuna-13b-v1.5', 'lmsys/vicuna-13b-v1.5-16k'],
                        type=str)
    parser.add_argument('--data_load_name', default='wpog',
                        choices=['cdm', 'wpog', 'wpog-l'], type=str)
    parser.add_argument('--running_examples', default=100, type=int)
    parser.add_argument('--result_save_name', default='wpog_intext.jsonl', type=str)
    parser.add_argument('--log_file_name', default='wpog_intext.log', type=str)

    args = parser.parse_args()

    return args


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_text(model, prompt, temperature):
    if 'gpt' in model:

        messages = [{'role': 'user', 'content': prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

        return response['choices'][0]['message']['content']
    else: # open-source llm models

        outputs = model.generate(
            tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device),
            temperature=temperature,
            top_k=50,
            top_p=0.95,
        ).to('cpu')

        response = [tokenizer.decode(output, skip_special_tokens=True).split('ASSISTANT:')[-1].strip()
                    for output in outputs]
        return response


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_multi_round_text(model, messages, temperature):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return response['choices'][0]['message']['content']

# References: https://github.com/openai/openai-cookbook/blob/5783656852d507c335955d14875ebc9902f628ef/examples/How_to_count_tokens_with_tiktoken.ipynb
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def count_message_tokens(content, model, type):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print('Model not found, using cl100k_base encoding.')
        encoding = tiktoken.get_encoding('cl100k_base')

    num_tokens = 0
    if type == 'input':
        messages = [{'role': 'user', 'content': content}]
        tokens_per_message = 4
        tokens_per_name = -1
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == 'name':
                    num_tokens += tokens_per_name
        num_tokens += 3
    elif type == 'output':
        num_tokens = len(encoding.encode(content))

    return num_tokens


def add_allin(example):
    """
    :param example:
    :return: example['all_in'] the generation result in 'all in' mode
    """
    opening = example['openings']
    outlines = example['outline']
    # articles = example['target']
    prompt = f""" Please generate a 500 tokens story with the given first sentence and outlines. The first sentence is {opening}.
    The outline of the story is {outlines}. 
    """
    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ')
                prediction = ''
            else:
                prediction = response
        else:
            logging.warning('Respond content is none.')
            prediction = ''
    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        prediction = ''

    example['all_in'] = prediction
    return example

def add_separate(example):
    """
        :param example:
        :return: example['separate'] the generation result in 'separate' mode
        """
    opening = example['openings']
    outlines = example['outline'].split('\n')
    # articles = example['target']
    messages = []
    input_tokens = 0
    for i in range(len(outlines)):
        if i == 0:
            prompt = f"""  Your final goal is to generate an around 500 tokens story based on given first sentence and outlines.
The first sentence is: {opening} And the outline of this story is: {outlines}.  
Firstly, you need generate only one paragraph of the story corresponding to following plot: {outlines[i]} and opening{opening} . """
        else:
            prompt = f""""Next, you could generate a paragraph of the story corresponding to following plot {outlines[i]}. """

        new_tokens = count_message_tokens(prompt, args.model, 'input')
        input_tokens += new_tokens
        logging.info('input tokens: ' + str(input_tokens))
        messages.append({'role': 'user', 'content': prompt})
        try:
            model_reply = generate_multi_round_text(
                model=args.model,
                messages=messages,
                temperature=temperature
            )

            logging.info('model_reply: ' + str(model_reply))

            if model_reply is not None:
                output_tokens = count_message_tokens(model_reply, args.model, 'output')
                logging.info('output tokens: ' + str(output_tokens))
                if input_tokens + output_tokens > max_tokens:
                    logging.warning('Over total tokens limit ')
                    prediction = ''
                else:
                    prediction = model_reply
                    messages.append({"role": "assistant", "content": model_reply})
            else:
                logging.warning('Respond content is none.')
                prediction = ''

        except Exception as e:
            logging.error('Failed to generate text: ' + e.__str__())
            prediction = ''

    # polish and m
    final_prompt = f"""Now connect all the paragraphs you've written and polish them to a 500 tokens essay to achieve the final goal to generate an around 500 tokens story with given first sentence and outlines. " \
                   First sentence: {opening}  
                   Outline: {outlines}."""
    messages.append({'role': 'user', 'content': final_prompt})

    try:
        model_reply = generate_multi_round_text(
            model=args.model,
            messages=messages,
            temperature=temperature
        )
        logging.info('model_reply: ' + str(model_reply))

        if model_reply is not None:
            output_tokens = count_message_tokens(model_reply, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ')
                prediction = ''
            else:
                prediction = model_reply
        else:
            logging.warning('Respond content is none.')
            prediction = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        prediction = ''

    example['separate'] = prediction
    return example


def add_dual_learning(example):
    """
        :param example:
        :return: example['dual'] the generation result in 'dual' mode
    """

    opening = example['openings']
    outlines = example['outline']

    messages = []
    input_tokens = 0
    for i in range(4):
        if i == 0:
            prompt = f"""  Please generate a 500 tokens story with given first sentence and outlines. The first sentence is: {opening}. And the outline of this story is {outlines}. """
        elif i == 1:
            prompt = f""" Please summarize your text into an outline that has 5 one-sentence points."""
        elif i == 2:
            prompt = f"""Please compare with the following true outline and rethinking how to improve the quality of outline-conditional generation. True outline: {outlines} """
        elif i == 3:
            prompt = f""" Based on the knowledge you just learned, regenerate a 500 tokens story with given first sentence and outlines. The first sentence is: {opening}. And the outline of this story is {outlines}. """

    new_tokens = count_message_tokens(prompt, args.model, 'input')

    input_tokens += new_tokens
    logging.info('input tokens: ' + str(input_tokens))
    messages.append({'role': 'user', 'content': prompt})
    try:
        model_reply = generate_multi_round_text(
            model=args.model,
            messages=messages,
            temperature=temperature
        )

        logging.info('model_reply: ' + str(model_reply))

        if model_reply is not None:
            output_tokens = count_message_tokens(model_reply, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ')
                prediction = ''
            else:
                prediction = model_reply
                messages.append({"role": "assistant", "content": model_reply})
        else:
            logging.warning('Respond content is none.')
            prediction = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        prediction = ''

    example['dual'] = prediction
    return example


def add_combo(example):
    """
        :param example:
        :return: example['combo'] the generation result in 'comb' mode
    """

    opening = example['openings']
    outlines = example['outline']
    separate_result = example['separate']
    messages = []
    input_tokens = 0
    for i in range(3):
        if i == 0:
            prompt = f""" This is your previous generated story based on outline {outlines}: Generated story: {separate_result}. Please summarize it into an outline that has 5 one-sentence points."""
        elif i == 1:
            prompt = f"""Please compare with the following true outline and rethinking how to improve the quality of outline-conditional generation. True outline: {outlines} """
        elif i == 2:
            prompt = f""" Based on the knowledge you just learned, regenerate a 500 tokens story with given first sentence and outlines. The first sentence is: {opening}. And the outline of this story is {outlines}. """

        new_tokens = count_message_tokens(prompt, args.model, 'input')

        input_tokens += new_tokens
        logging.info('input tokens: ' + str(input_tokens))
        messages.append({'role': 'user', 'content': prompt})

        try:
            model_reply = generate_multi_round_text(
                model=args.model,
                messages=messages,
                temperature=temperature
            )

            logging.info('model_reply: ' + str(model_reply))

            if model_reply is not None:
                output_tokens = count_message_tokens(model_reply, args.model, 'output')
                logging.info('output tokens: ' + str(output_tokens))
                if input_tokens + output_tokens > max_tokens:
                    logging.warning('Over total tokens limit ')
                    prediction = ''
                else:
                    prediction = model_reply
                    messages.append({"role": "assistant", "content": model_reply})
            else:
                logging.warning('Respond content is none.')
                prediction = ''

        except Exception as e:
            logging.error('Failed to generate text: ' + e.__str__())
            prediction = ''

    example['combo'] = prediction
    return example


def add_intext_learning(example,df):
    """
       :param example:
       :return: example['intext'] the generation result in 'all in' mode with one random case
       """

    opening = example['openings']
    outlines = example['outline']
    # articles = example['target']
    new_example = df.iloc[random.randint(100, 200)]
    case_opening = new_example['openings']
    case_outlines = new_example['outline']
    case_articles = new_example['targets']

    prompt = f""" Here we give a example of generating a 500 tokens story with the given first sentence and outlines: first sentence is {case_opening}, the outline is {case_outlines}. The generated story is {case_articles}.     
     Please generate a 500 tokens story with the given first sentence and outlines. The first sentence is {opening}.
       The outline of the story is {outlines}. 
       """

    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ')
                prediction = ''
            else:
                prediction = response
        else:
            logging.warning('Respond content is none.')
            prediction = ''
    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        prediction = ''

    example['intext'] = prediction
    return example


def main():
    save_path = Path(__file__).parent / Path('../results') / Path(args.result_save_name)
    load_path = ''
    if args.data_load_name == 'wpogl':
        load_path = Path(__file__).parent / Path('../wp_dataset') / Path('wpog_long.csv')
    elif args.data_load_name == 'wpog':
        load_path = Path(__file__).parent / Path('../wp_dataset') / Path('wpog_test.csv')
    elif args.data_load_name == 'cdm':
        load_path = Path(__file__).parent / Path('cdm') / Path('test.csv')
    data = pd.read_csv(load_path)
    test_index = [i for i in range(args.running_examples)]

    dataset = Dataset.from_pandas(data).select(test_index)
    dataset = dataset.select(list(range(50)))
    logger.info("******Inference with 'intext' *******")
    dataset = dataset.map(lambda row: add_intext_learning(row,data))

    logger.info("******Inference with 'All in' *******")
    dataset = dataset.map(add_allin)

    logger.info("******Inference with 'Outline control' *******")
    dataset = dataset.map(add_separate)

    logger.info("******Inference with 'Dual learning' *******")
    dataset = dataset.map(add_dual_learning)

    logger.info("******Inference with 'Combo' *******")
    dataset = dataset.map(add_combo)

    logger.info("******Completed *******")
    dataset.to_json(save_path, orient='records',lines=True)


if __name__ == '__main__':
    args = parse_arguments()

    log_file_path = Path(__file__).parent / Path('../logs') / Path(args.log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # References: https://platform.openai.com/docs/api-reference/authentication
    openai.api_key = args.api_key
    model_max_tokens = {
        'gpt-3.5-turbo': 4097,
        'gpt-3.5-turbo-16k': 16385,
        'lmsys/vicuna-13b-v1.5': 4097,
        'lmsys/vicuna-13b-v1.5-16k': 16385,
    }

    temperature = 0.0
    max_tokens = model_max_tokens.get(args.model) if model_max_tokens.get(args.model) is not None else 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    if 'gpt' not in args.model:
        tokenizer = AutoTokenizer.from_pretrained(
            args.checkpoint,
            use_fast=True,
            trust_remote_code=True,
            token=args.access_token,
            cache_dir=args.cache_dir
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            cache_dir=args.cache_dir
        )

    main()
