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


def roleplay_vicuna_call(prompt, model): 

  prompt = re.sub(r"\s+", " ", prompt)
  prompt = re.sub("\s\s+" , " ", prompt)

  listed_labels = ['not sexist', 'sexist']

  system_message = f"""
    ### Instruction: You are an expert in understanding slight nuances in the text. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. \
    You will act as a text classification assistant, experienced at understanding the presense of 'sexist' attitudes in textual contents \
    based on `context` (on the political or non-political incident in question),  \
    `content` (what the speaker believes) and `phrasing` (because of the speaker’s choice of words).\
    Classify the input text by assigning them a label from the list of {listed_labels}. \
    The model should only generate ONE of these labels as the output text, and nothing more or different.
    """

  human_message = f"""
    ### Input: {prompt}
    ### Response:
    """

  total_prompt = system_message + human_message
  #print(num_tokens_from_string(total_prompt, model))
 
  response = model(total_prompt)
  #messages = response['choices'][0]['message']['content']
  response = response.lower().strip()

  
  if any((match := n) in response for n in listed_labels):
    response = match
  else:
    response = 'not sexist'


  return response


def content_vicuna_call(prompt, model): 

  prompt = re.sub(r"\s+", " ", prompt)
  prompt = re.sub("\s\s+" , " ", prompt)

  listed_labels = ['not sexist', 'sexist']

  system_message = f"""
    ### Instruction: You are an expert in understanding slight nuances in the text. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. \
    You will act as a text classification assistant, experienced at understanding the presense of 'sexist' attitudes in textual contents \
    based on `context` (on the political or non-political incident in question),  \
    `content` (what the speaker believes) and `phrasing` (because of the speaker’s choice of words).\
    A text may be 'sexist' if the speaker shows a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
    Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. This may contain texts \
    stating that there are no inequalities between men and women (any more) and/or that are opposing feminism. 
    Classify the input text by assigning them a label from the list of {listed_labels}. \
    The model should only generate ONE of these labels as the output text, and nothing more or different.

    """

  human_message = f"""
    ### Input: {prompt}
    ### Response:
    """

  total_prompt = system_message + human_message
  #print(num_tokens_from_string(total_prompt, model))

  response = model(total_prompt)
  #messages = response['choices'][0]['message']['content']
  response = response.lower().strip()

  
  if any((match := n) in response for n in listed_labels):
    response = match
  else:
    response = 'not sexist'


  return response


def zeroshot_vicuna_call(prompt, model): 

  prompt = re.sub(r"\s+", " ", prompt)
  prompt = re.sub("\s\s+" , " ", prompt)

  listed_labels = ['not sexist', 'sexist']

  system_message = f"""

    ### Instruction: You are an expert in understanding slight nuances in the text. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. \
    You will act as a text classification assistant, experienced at understanding the presense of 'sexist'  attitudes in textual contents \
    based on `context` (on the political or non-political incident in question),  \
    `content` (what the speaker believes) and `phrasing`(because of the speaker’s choice of words). \
    A text may be 'sexist' if the speaker shows a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
    Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. \
    Texts may be 'sexist' simply because of how the speaker phrases it – independently from what general beliefs or attitudes the speaker holds. \
    A text is sexist, when it contains attacks, foul language, or derogatory depictions directed towards individuals because of their gender, by means of name-calling, \
    attacks, objectification, inflammatory messages. However, just because a message is aggressive or uses offensive language does not mean that it is 'sexist'. \
    Punctuation, along with the context, should influence how a text is interpreted and categorized with the provided labels. \
    Classify the input text by assigning them a label from the list of {listed_labels}. 
    The model should only generate ONE of these labels as the output text, and nothing more or different.

    """

  human_message = f"""
    ### Input: {prompt}
    ### Response:
    """

  total_prompt = system_message + human_message
  #print(num_tokens_from_string(total_prompt, model))

  response = model(total_prompt)
  #messages = response['choices'][0]['message']['content']
  response = response.lower().strip()

  
  if any((match := n) in response for n in listed_labels):
    response = match
  else:
    response = 'not sexist'

  return response


def fewshot_vicuna_call(prompt, model): 

  prompt = re.sub(r"\s+", " ", prompt)
  prompt = re.sub("\s\s+" , " ", prompt)

  labels = """
        not sexist; sexist
        """

  listed_labels = ['not sexist', 'sexist']

  system_message = f"""

    ### Instruction: You are an expert in understanding slight nuances in the text. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. \
    You will act as a text classification assistant, experienced at understanding the presense of 'sexist' attitudes in textual contents \
    based on context (political or non-political incident in question), content (what the speaker believes) and phrasing (because of the speaker’s choice of words). \
    A text may be 'sexist' if the speaker shows a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
    Texts may be 'sexist' simply because of how the speaker phrases it independently from what general beliefs or attitudes the speaker holds. \
    Sexist text contain attacks, foul language, or derogatory depictions directed towards individuals because of their gender, by means of name-calling, \
    attacks, objectification, inflammatory messages. However, just because a message is aggressive or uses offensive language does not always mean that it is 'sexist'. \
    Punctuation and emojis, aside from context, should influence how a text is interpreted and categorized with the provided labels. \
    Classify the input text by assigning them a label from the list of {listed_labels}. 
    The model should only generate ONE of these labels as the output text, and nothing more or different.
    Here are a few examples:
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
    ### Input: {prompt}
    ### Response:
    """

  total_prompt = system_message + human_message
  #print(num_tokens_from_string(total_prompt, model))

  response = model(total_prompt)
  #messages = response['choices'][0]['message']['content']
  response = response.lower().strip()

  
  if any((match := n) in response for n in listed_labels):
    response = match
  else:
    response = 'not sexist'


  return response



def main():
    parser.add_argument('--dataset_name', help = 'give the name of the dataset', required = True, type = str)
    parser.add_argument('--model_name', help = 'give the name of the model', type= str,
            choices = ['TheBloke/vicuna-13B-v1.5-GGUF', 'TheBloke/vicuna-7B-v1.5-GGUF'],
            default='TheBloke/vicuna-7B-v1.5-GGUF')
    parser.add_argument('--model_file', help = 'give the name of the model file name (i.e., GGUF file extension)', type= str,
            choices = ['vicuna-7b-v1.5.Q5_K_M.gguf', 'vicuna-7b-v1.5.q4_K_M.gguf', 'vicuna-7b-v1.5.Q3_K_M.gguf'], # more for 13b and further
            default='vicuna-7b-v1.5.q4_K_M.gguf')
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
                response = roleplay_vicuna_call(row, model)
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
                response = content_vicuna_call(row, model)
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
                response = zeroshot_vicuna_call(row, model)
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
                response = fewshot_vicuna_call(row, model)
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