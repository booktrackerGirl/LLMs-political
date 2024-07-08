#!/usr/bin/env python
# setting up the environment

import os
import pathlib
from openai import OpenAI
import api_keys
import sys

# using openai api keys as environment variable in the workspace
#os.environ['OPENAI_API_KEY'] = '<api_key>'

#openai_api_key = os.environ['OPENAI_API_KEY']
#openai_api_key = api_key.OPENAI_API_KEY
client = OpenAI(
  api_key=api_keys.OPENAI_API_KEY,  # this is also the default, it can be omitted
)

import numpy as np
# for generatring dispersion chart.
import plotly.graph_objects as go

#from torch import cuda, bfloat16
#import torch
import transformers
import pandas as pd
import io
import argparse
import os
import bitsandbytes
#import bitsandbytes as bnb_4bit_compute_dtype
import sentencepiece
import time
import gc

from functools import partial

#from transformers import Trainer
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.llms import HuggingFacePipeline, LlamaCpp
from langchain import PromptTemplate, FewShotPromptTemplate, LLMChain, HuggingFaceHub
#from trl import SFTTrainer # for training, use this: https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI

from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)


#from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
#    DataCollatorForLanguageModeling, Trainer, TrainingArguments, LlamaTokenizer, logging, EvalPrediction
#from  adapters import AdapterTrainer


from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferWindowMemory
#from langchain.agents import load_tools
from langchain.prompts.example_selector import LengthBasedExampleSelector
#from transformers import StoppingCriteria, StoppingCriteriaList

from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
#from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatAnthropic

# https://sj-langchain.readthedocs.io/en/latest/_modules/langchain/prompts/few_shot_with_templates.html
from langchain.prompts import FewShotPromptWithTemplates, FewShotChatMessagePromptTemplate

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.chat_models import ChatOpenAI

#from skllm import DynamicFewShotGPTClassifier

from langchain.agents import initialize_agent

from collections import defaultdict
#from riveter import Riveter

import re
import gc

from transformers import set_seed
set_seed(1234)



import warnings
warnings.filterwarnings('ignore')


import argparse
parser = argparse.ArgumentParser(description='Process text classification with various models in the dataset.')



def roleplay_gpt_call(prompt, model): 

  prompt = re.sub(r"\s+", " ", prompt)
  prompt = re.sub("\s\s+" , " ", prompt)

  listed_labels = ['not sexist', 'sexist']

  system_message = f"""
    You are an expert in understanding slight nuances in the text, even when presented with texts that lack enough context.. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. \
    You will act as a text classification assistant, experienced at understanding the presense of sexist attitudes in textual contents.
    Classify the input text by assigning them a label from the list of {listed_labels}. 
    The model should generate ONLY ONE of these labels as the output label, and nothing more or different. 
    """

  human_message = f"""
    Label the following text based on the guidelines:
    Text: {prompt}
    """
  
  response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": human_message},
        {"role": "assistant", "content": "Label: "}
    ]
    )
  
  response = response.choices[0].message.content
  
  return response


def content_gpt_call(prompt, model): 

  prompt = re.sub(r"\s+", " ", prompt)
  prompt = re.sub("\s\s+" , " ", prompt)

  listed_labels = ['not sexist', 'sexist']

  system_message = f"""
    You are an expert in understanding slight nuances in the text, even when presented with texts that lack enough context. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. \
    You will act as a text classification assistant, experienced at understanding the presense of 'sexist' attitudes in textual contents \
    based on `context` (on the political or non-political incident in question),  \
    `content` (what the speaker believes) and `phrasing` (because of the speaker’s choice of words).\
    A text may be 'sexist' if the speaker shows a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
    Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. This may contain texts \
    stating that there are no inequalities between men and women (any more) and/or that are opposing feminism. 
    Classify the input text by assigning them a label from the list of {listed_labels}. 
    The model should generate ONLY ONE of these labels as the output label, and nothing more or different.
    """

  human_message = f"""
    Label the following text based on the guidelines:
    Text: {prompt}
    """
  
  response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": human_message},
        {"role": "assistant", "content": "Label: "}
    ]
    )
  
  response = response.choices[0].message.content
  
  return response


def zeroshot_gpt_call(prompt, model): 

  prompt = re.sub(r"\s+", " ", prompt)
  prompt = re.sub("\s\s+" , " ", prompt)

  listed_labels = ['not sexist', 'sexist']

  system_message = f"""
    You are an expert in understanding slight nuances in the text. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. \
    You will act as a text classification assistant, experienced at understanding the presense of 'sexist'  attitudes in textual contents \
    based on `context` (political or non-political incident in question),  \
    `content` (what the speaker believes) and `phrasing` (because of the speaker’s choice of words). \
    A text may be sexist if the speaker shows a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
    Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. \
    Texts may be 'sexist' simply because of how the speaker phrases it independently from what general beliefs or attitudes the speaker holds. \
    Punctuation and emoticons, along with the context, should influence how a text is interpreted and categorized with the provided labels. \
    'Sexist' texts could be in the form of joke or sarcasm. Texts could contain direct or indirect references (e.g., hashtags) to fictional characters from books, movies, TV series and digital comics.
    Classify the input text by assigning them a label from the given list: {listed_labels}. \
    The model should generate ONLY ONE of these labels as the output, and nothing more or different. You CANNOT leave it empty, or generate explanations.
    """

  human_message = f"""
    Label the following text based on the guidelines:
    Text: {prompt}
    """
 
  
  response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": human_message},
        {"role": "assistant", "content": "Label: "}
    ]
    )
  
  response = response.choices[0].message.content
  
  return response



def fewshot_gpt_call(prompt, model): 

  prompt = re.sub(r"\s+", " ", prompt)
  prompt = re.sub("\s\s+" , " ", prompt)

  listed_labels = ['not sexist', 'sexist']

  system_message = f"""
    You are an expert in understanding slight nuances in the text. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. \
    You will act as a text classification assistant, experienced at understanding the presense of 'sexist'  attitudes in textual contents \
    based on `context` (political or non-political incident in question),  \
    `content` (what the speaker believes) and `phrasing` (because of the speaker’s choice of words). \
    A text may be sexist if the speaker shows a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
    Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. \
    Texts may be 'sexist' simply because of how the speaker phrases it independently from what general beliefs or attitudes the speaker holds. \
    Punctuation and emoticons, along with the context, should influence how a text is interpreted and categorized with the provided labels. \
    'Sexist' texts could be in the form of joke or sarcasm. Texts could contain direct or indirect references (e.g., hashtags) to fictional characters from books, movies, TV series and digital comics.
    Classify the input text by assigning them a label from the given list: {listed_labels}. \
    The model should generate ONLY ONE of these labels as the output, and nothing more or different. You CANNOT leave it empty, or generate explanations.
    Here are a few examples:
    a. 'Yes, but she was asking for it!'? When will people grow up and stop sexualizing women?!
       Label: not sexist
    b. The conference will cover various topics, including technology and art. Gender biases will not be tolerated.
       Label: not sexist
    c. Was she flashing her minge?
       Label: sexist
    d. Look the way she dressed! She's definitely asking for it.🙄
       Label: sexist
    e. Text: In the debating chamber her high squeeky voice sounded like a hyena.
       Label: sexist
    f. Text: The government's economic policies disproportionately affect women's access to job opportunities.
       Label: not sexist
    """

  human_message = f"""
    Label the following text based on the guidelines:
    Text: {prompt}
    """
  
  response = client.chat.completions.create(
    model=model,
    messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": human_message},
        {"role": "assistant", "content": "Label: "}
    ]
    )
  
  response = response.choices[0].message.content
  
  return response



def main():
    parser.add_argument('--dataset_name', help = 'give the name of the dataset', required = True, type = str)
    parser.add_argument('--model_name', help = 'give the name of the model', type= str,
            choices = ['gpt-3.5-turbo-1106', 'gpt-3.5-turbo', 'gpt-3.5-turbo-instruct'],
            default='gpt-3.5-turbo')
    parser.add_argument('--output_folder', help = 'give the name of the output folder', required = True, type = str)
    parser.add_argument('--output_file', help = 'give the name of the output file name', required = True, type = str)
    args = parser.parse_args()


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
                response = roleplay_gpt_call(row, model = args.model_name)
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
                response = content_gpt_call(row, model = args.model_name)
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
                response = zeroshot_gpt_call(row, model = args.model_name)
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
                response = fewshot_gpt_call(row, model = args.model_name)
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