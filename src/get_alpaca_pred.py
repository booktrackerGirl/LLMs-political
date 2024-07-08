#!/usr/bin/env python
from transformers import T5Tokenizer, T5ForConditionalGeneration

import re

import os
import pathlib
import api_keys
import sys
import gc

import numpy as np
# for generatring dispersion chart.

from torch import cuda, bfloat16
import torch
import transformers
import pandas as pd
import argparse
import os
#import bitsandbytes as bnb_4bit_compute_dtype
import time
import itertools
# https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/chains/llm.py
import langchain

from transformers import StoppingCriteria, StoppingCriteriaList


import ctransformers
from ctransformers import AutoModelForCausalLM, AutoTokenizer

from transformers import set_seed
set_seed(1234)

import warnings
warnings.filterwarnings('ignore')


import argparse
parser = argparse.ArgumentParser(description='Process text classification with various models in the dataset.')



def roleplay_alpaca_call(prompt, model):

    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

    listed_labels = ['sexist', 'not sexist']

    system_message = f"""
        You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
        You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 

        Please answer if a text can be classified as any ONE of the following 2 labels:
        1) sexist
        2) not sexist

        Please reply with ONLY ONE of the options and nothing else.

        """

    human_message = f"""
        Label this text based on the past guidelines:
        Text: {prompt}
        ### Response:
        """

    total_prompt = system_message + human_message
    #print(num_tokens_from_string(total_prompt, model))

    response = model(total_prompt)
    #messages = response['choices'][0]['message']['content']
    response = response.lower().strip()

    
    if any((match := n) in response for n in listed_labels):
        response = match
    elif response == 1:
        response = 'sexist'
    elif response == 2:
        response = 'not sexist'
    else:
        response = 'not sexist'

    return response


def content_alpaca_call(prompt, model):

    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

    listed_labels = ['sexist', 'not sexist']

    system_message = f"""
        You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
        You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 

        ### Read on for detailed explanations. Something can be 'sexist' because of its `context` (on the political or non-political incident in question), \
        because of its `content` (what the speaker believes), and because of its `phrasing` (because of the speaker’s choice of words).

        ### Sexism: A text may be sexist if the speaker shows a prescriptive set of behaviors or qualities, \
        that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
        a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
        Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. This may contain texts \
        stating that there are no inequalities between men and women (any more) and/or that are opposing feminism. 

        Please answer if a text can be classified as any ONE of the following 2 labels:
        1) sexist
        2) not sexist

        Please reply with ONLY ONE of the options and nothing else.
        """

    human_message = f"""
    Label this text based on the past guidelines:
    Text: {prompt}
    ### Response:
    """

    # combine the prompts first
    total_prompt = system_message + human_message
    response = model(total_prompt).lower().strip()
    
    if any((match := n) in response for n in listed_labels):
        response = match
    elif response == 1:
        response = 'sexist'
    elif response == 2:
        response = 'not sexist'
    else:
        response = 'not sexist'

    return response


def zeroshot_alpaca_call(prompt, model):

    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

    listed_labels = ['sexist', 'not sexist']

    system_message = f"""
    You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 

    ### Read on for detailed explanations. Something can be 'sexist' because of its `context` (on the political or non-political incident in question), \
    because of its `content` (what the speaker believes), and because of its `phrasing` (because of the speaker’s choice of words).

    ### Sexism: A text may be sexist if the speaker shows a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes.

    ### Texts with mentions of government policies, political parties and their candidates should be looked with caution, to separate disagreement regarding policies and \
    personalities of the mentioned individuals, as it can be either personal or political or both. 'Sexist' texts could be in the form of joke or sarcasm. 

    ### Punctuation, along with the context, should influence how a text is interpreted and categorized with the provided labels.

    Please answer if a text can be classified as any ONE of the following 2 labels:
    1) sexist
    2) not sexist

    Please reply with ONLY ONE of the options and nothing else.
    """

    human_message = f"""
    Label this text based on the past guidelines:
    Text: {prompt}
    ### Response:

    """

    # combine the prompts first
    total_prompt = system_message + human_message

    response = model(total_prompt).lower().strip()

    if any((match := n) in response for n in listed_labels):
        response = match
    elif response == 1:
        response = 'sexist'
    elif response == 2:
        response = 'not sexist'
    else:
        response = 'not sexist'
    
    return response     


def fewshot_alpaca_call(prompt, model):

    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

    listed_labels = ['sexist', 'not sexist']

    system_message = f"""
    You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 
    ### Texts can be 'sexist' because of its `context` (on the political or non-political incident in question), \
    because of its `content` (what the speaker believes), and because of its `phrasing` (because of the speaker’s choice of words).
    ### Sexism: A text may be sexist if the speaker shows a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties that supposedly differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes.
    ### Texts with mentions of government policies, political parties and their candidates should be looked with caution, to separate disagreement regarding policies and \
    personalities of the mentioned individuals, as it can be either personal or political or both. \
    Punctuation, along with the context, should influence how a text is interpreted and categorized with the provided labels.
    ### A text should be labelled as ONE of the following 2 labels:
    1) sexist
    2) not sexist
    Please reply with ONLY ONE of the options and nothing else.
    ### Here are a few examples:
    a. Text: When will people grow up and stop sexualizing women?!
       Response: not sexist
    b. Text: Gender biases will not be tolerated.
       Response: not sexist
    c. Text: Was she flashing her minge?
       Response: sexist
    d. Text: Look the way she dressed! She's definitely asking for it.🙄
       Response: sexist
    
    """

    human_message = f"""
    Label the following text based on the guidelines:
    Text: {prompt}
    ### Response:

    """

    # combine the prompts first
    total_prompt = system_message + human_message

    response = model(total_prompt).lower().strip()

    if any((match := n) in response for n in listed_labels):
        response = match
    elif response == 1:
        response = 'sexist'
    elif response == 2:
        response = 'not sexist'
    else:
        response = 'not sexist'
    
    return response   




def main():

    parser.add_argument('--dataset_name', help = 'give the name of the dataset', required = True, type = str)
    parser.add_argument('--model_name', help = 'give the name of the model', type= str,
            choices = ['TheBloke/claude2-alpaca-13B-GGUF', 'TheBloke/claude2-alpaca-7B-GGUF'],
            default='TheBloke/claude2-alpaca-7B-GGUF')
    parser.add_argument('--model_file', help = 'give the name of the model file name (i.e., GGUF file extension)', type= str,
            choices = ['claude2-alpaca-7b.Q3_K_M.gguf', 'claude2-alpaca-7b.Q4_K_M.gguf', 'claude2-alpaca-7b.Q5_K_M.gguf'], # more for 13b and further
            default='claude2-alpaca-7b.Q4_K_M.gguf')
    #parser.add_argument('--gpu_layers', help='if GPU layers present, add the number here', type = str, required=True, default = 0)
    parser.add_argument('--output_folder', help = 'give the name of the output folder', required = True, type = str)
    parser.add_argument('--output_file', help = 'give the name of the output file name', required = True, type = str)
    args = parser.parse_args()

    # Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
    model = AutoModelForCausalLM.from_pretrained(model_path_or_repo_id=args.model_name, model_file=args.model_file, model_type="llama", gpu_layers=0) #args.gpu_layers

    # Load data
    data = pd.read_csv(args.dataset_name)

    len_of_df = len(data)

    results_chat_1 = []
    flag = False  # Defining the flag variable
    num = 0
    for row in data['text']:
        rate_limit_per_minute = 20
        delay = 60.0 / rate_limit_per_minute
        if num == len_of_df: # In case if the loop does not break
            flag = True
            print(f'Reached the {len_of_df}')
            break
        else:
            try:
                response = roleplay_alpaca_call(row, model)
                print(response)
                results_chat_1.append(response)
                print(f'Finished iteration {num} for 1st one')
                num += 1
                # Calculate the delay based on your rate limit
                time.sleep(delay)
            except Exception as e:
                #print(f'Error type {e.message}')
                print(f'Error type {e}')
                print(f'Since it is taking longer for {num}, we use the default label here')
                num += 1
                results_chat_1.append('not sexist')
                print('Will come back in 5 minutes ...')
                time.sleep(5 * 60) # 5 minutes
        if flag == True:
            print('Reached the whole length of the dataset. Stopping now')
            break

    
    results_chat_2 = []
    flag = False  # Defining the flag variable
    num = 0
    for row in data['text']:
        rate_limit_per_minute = 20
        delay = 60.0 / rate_limit_per_minute
        if num == len_of_df: # In case if the loop does not break
            flag = True
            print(f'Reached the {len_of_df}')
            break
        else:
            try:
                response = content_alpaca_call(row, model)
                print(response)
                results_chat_2.append(response)
                print(f'Finished iteration {num} for 2nd one')
                num += 1
                # Calculate the delay based on your rate limit
                time.sleep(delay)
            except Exception as e:
                #print(f'Error type {e.message}')
                print(f'Error type {e}')
                print(f'Since it is taking longer for {num}, we use the default label here')
                num += 1
                results_chat_2.append('not sexist')
                print('Will come back in 5 minutes ...')
                time.sleep(5 * 60) # 5 minutes
        if flag == True:
            print('Reached the whole length of the dataset. Stopping now')
            break


    results_chat_3 = []
    flag = False  # Defining the flag variable
    num = 0
    for row in data['text']:
        rate_limit_per_minute = 20
        delay = 60.0 / rate_limit_per_minute
        if num == len_of_df: # In case if the loop does not break
            flag = True
            print(f'Reached the {len_of_df}')
            break
        else:
            try:
                response = zeroshot_alpaca_call(row, model)
                print(response)
                results_chat_3.append(response)
                print(f'Finished iteration {num} for 3rd one')
                num += 1
                # Calculate the delay based on your rate limit
                time.sleep(delay)
            except Exception as e:
                #print(f'Error type {e.message}')
                print(f'Error type {e}')
                print(f'Since it is taking longer for {num}, we use the default label here')
                num += 1
                results_chat_3.append('not sexist')
                print('Will come back in 5 minutes ...')
                time.sleep(5 * 60) # 5 minutes
        if flag == True:
            print('Reached the whole length of the dataset. Stopping now')
            break


    results_chat_4 = []
    flag = False  # Defining the flag variable
    num = 0
    for row in data['text']:
        rate_limit_per_minute = 20
        delay = 60.0 / rate_limit_per_minute
        if num == len_of_df: # In case if the loop does not break
            flag = True
            print(f'Reached the {len_of_df}')
            break
        else:
            try:
                response = fewshot_alpaca_call(row, model)
                print(response)
                results_chat_4.append(response)
                print(f'Finished iteration {num} for 4th one')
                num += 1
                # Calculate the delay based on your rate limit
                time.sleep(delay)
            except Exception as e:
                #print(f'Error type {e.message}')
                print(f'Error type {e}')
                print(f'Since it is taking longer for {num}, we use the default label here')
                num += 1
                results_chat_4.append('not sexist')
                print('Will come back in 5 minutes ...')
                time.sleep(5 * 60) # 5 minutes
        if flag == True:
            print('Reached the whole length of the dataset. Stopping now')
            break


    data['roleplay'] = results_chat_1
    data['content'] = results_chat_2
    data['zero-shot'] = results_chat_3
    data['few-shot'] = results_chat_4

    data = data.drop(columns='text') # remove the text

    data.to_csv(f'{args.output_folder}/{args.output_file}.csv')

    del args
    #gc.collect()
    del tokenizer
    #gc.collect()
    del model
    #gc.collect()



if __name__ == '__main__':
    main()