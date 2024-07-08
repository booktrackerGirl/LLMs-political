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

from transformers import set_seed
set_seed(1234)

import warnings
warnings.filterwarnings('ignore')

import argparse
parser = argparse.ArgumentParser(description='Process text classification with various models in the dataset.')


def generate(input_text, tokenizer, model):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=10, temperature=0.0, decoder_start_token_id=tokenizer.bos_token_id)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    return output

def roleplay_flan_call(prompt, model, tokenizer):

    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

    listed_labels = ['sexist', 'not sexist']

    system_message = f"""
    You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 

    Please answer if the following text is any of the following 2 labels:
    1) sexist
    2) not sexist

    Please reply with either of the two options and nothing else.
    """

    human_message = f"""
    Text: {prompt}
    Label:
    """

    # combine the prompts first
    total_prompt = system_message + human_message
    response = generate(total_prompt, tokenizer, model)

    if response not in listed_labels:
        response = 'not sexist'
    else:
        response = response
    
    return response   


def content_flan_call(prompt, model, tokenizer):

    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

    listed_labels = ['sexist', 'not sexist']

    system_message = f"""
    You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 
    
    ### Read on for detailed explanations. Something can be sexist because of its `content` (on the political or non-political incident in question).

    ### Instructions for understanding `content`:

    ## Sexist: Texts formulating a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
    Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. This may contain texts \
    stating that there are no inequalities between men and women (any more) and/or that are opposing feminism. They might possess views which indicate \
    women are not competent adults, or women having favourable traits that men stereotypically lack. Also, it could mean \
    viewing women as romantic objects with a genuine desire for psychological closeness. Texts could contain comments or actions that cause or \
    aggravate restrictions on how women communicate. Texts could also have promotion of discriminatory codes of conduct for women in the guise of morality; \
    also applies to statements that feed into such codes and narratives. 'Sexist' attitudes can also be indirect attempting to deny responsibility for an utterance, mediating through irony or by disguising \
    the force of sexism of utterance through humour and innuendo, embedding sexism sexism at the level of presupposition, or prefacing sexist statements with disclaimers or hesitation. \
    The text can contain statements that question the rights and treatment of women in a manner that reflects gender bias or sexism. \
    Sexism could also be institutional, where the sexist attitude drawn upon repeatedly by various institutions become normalised. 

    Please answer if the following text is any of the following 2 labels:
    1) sexist
    2) not sexist

    Please reply with either of the two options and nothing else.
    """

    human_message = f"""
    Label this text based on the past guidelines:
        Text: {prompt}
        Label:

        """

    # combine the prompts first
    total_prompt = system_message + human_message
    response = generate(total_prompt, tokenizer, model)

    if response not in listed_labels:
        response = 'not sexist'
    else:
        response = response

    return response   


def zeroshot_flan_call(prompt, model, tokenizer):

    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

    listed_labels = ['sexist', 'not sexist']

    system_message = f"""
    You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 
    
    ### Read on for detailed explanations. Something can be 'sexist' because of its `context` (on the political or non-political incident in question), \
    because of its `content` (what the speaker believes), and because of its `phrasing` (because of the speaker’s choice of words).

    The `context` is mostly political and it consists of texts that revolve around discussions of policies, government actions, ideologies, elections, etc. \
    These texts would aim to engage with societal issues, power dynamics, and decision-making processes within the realm of public affairs. \
    Pertaining to the practice and theory of influencing other people on a civic or individual level, often concerning government or public affairs. \
    Reference to any of the target's current or former political and/or behavioral activity. This could be an implicit indication \
    in the text, or a direct implication through mentions of their position on a certain topic. A typical political text could have strong language, \
    a harsh tone and slurs; and question the political standing and political opinion of the target (usually indicated by mention of policies or government strategies) \
    or the political position the target holds. Yet it should not undermine the intelligence of the target.

    Sexism due to `content`. A text may be sexist if the speaker shows a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
    Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. This may contain texts \
    stating that there are no inequalities between men and women (any more) and/or that are opposing feminism. They might possess views which indicate \
    women are not competent adults, or women having favourable traits that men stereotypically lack. For example, the speaker may \
    express sexist attitudes towards gender inequality, either endorsing it (e.g. “some jobs are best left to men”), or antagonizing it \
    (e.g. “the pay gap between genders does not exist, feminists should stop complaining”). \
    Also, the speaker may express stereotypes (how genders are traditionally seen and compared to each other) and behavioral expectations (how individuals of a gender 
    should behave according to traditional views). Sexism may also include positive stereotypes (e.g. “women are the best home cooks”), or target men (e.g., “men should not cry”)!

    Sexism due to `phrasing`: Texts may be sexist simply because of how the speaker phrases it–independently from what general beliefs or attitudes the speaker holds. 
    A message is sexist, for example, when it contains attacks, foul language, or derogatory depictions directed towards individuals because of their gender, e.g. by means of name-calling ("you bitch"), 
    attacks ("I’m going to kick her back to the kitchen"), objectification ("She’s stupid but I’d still do her"), inflammatory messages ("burn all women"). 
    However, just because a message is aggressive or uses offensive language does not mean that it is sexist!

    Sexism due to `context`: Texts could be mocking female perspectives from female politicians, minimize their political contributions and undermine their achievements. \
    It can also question their commitments to public office by implicating that they should focus more on their family commitments, and their political performance being compared to their capability in \
    familial setting. They may also publish appearance-centric criticism of the female politicians, unlike their male counterparts. They tone could be ironic and exaggerated, and often in the guise of humour.

    Please answer if the following text is any of the following 2 labels:
    1) sexist
    2) not sexist

    Please reply with either of the two options and nothing else.
    """

    human_message = f"""
    Label this text based on the past guidelines:
    Text: {prompt}
    Label:

    """

    # combine the prompts first
    total_prompt = system_message + human_message
    response = generate(total_prompt, tokenizer, model)

    if response not in listed_labels:
        response = 'not sexist'
    else:
        response = response

    return response   


def fewshot_flan_call(prompt, model, tokenizer):

    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()

    listed_labels = ['sexist', 'not sexist']

    system_message = f"""
    You are an expert in understanding slight linguistic nuances in the text, even when presented with texts that lack enough context. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. 
    
    ### Read on for detailed explanations. Something can be 'sexist' because of its `context` (on the political or non-political incident in question), \
    because of its `content` (what the speaker believes), and because of its `phrasing` (because of the speaker’s choice of words).

    The `context` is mostly political and it consists of texts that revolve around discussions of policies, government actions, ideologies, elections, etc. \
    These texts would aim to engage with societal issues, power dynamics, and decision-making processes within the realm of public affairs. \
    Pertaining to the practice and theory of influencing other people on a civic or individual level, often concerning government or public affairs. \
    Reference to any of the target's current or former political and/or behavioral activity. This could be an implicit indication \
    in the text, or a direct implication through mentions of their position on a certain topic. A typical political text could have strong language, \
    a harsh tone and slurs; and question the political standing and political opinion of the target (usually indicated by mention of policies or government strategies) \
    or the political position the target holds. Yet it should not undermine the intelligence of the target.

    Sexism due to `content`. A text may be sexist if the speaker shows a prescriptive set of behaviors or qualities, \
    that women (and men) are supposed to exhibit in order to conform to traditional gender roles. This could be texts formulating \
    a descriptive set of properties  that  supposedly  differentiates the two genders and expressed through explicit or implicit comparisons and perpetuating gender-based stereotypes. \
    Aside from acknowledging the inequalities, these texts could be endorsing or justifying them in a non-flattering manner. This may contain texts \
    stating that there are no inequalities between men and women (any more) and/or that are opposing feminism. They might possess views which indicate \
    women are not competent adults, or women having favourable traits that men stereotypically lack. For example, the speaker may \
    express sexist attitudes towards gender inequality, either endorsing it (e.g. “some jobs are best left to men”), or antagonizing it \
    (e.g. “the pay gap between genders does not exist, feminists should stop complaining”). \
    Also, the speaker may express stereotypes (how genders are traditionally seen and compared to each other) and behavioral expectations (how individuals of a gender 
    should behave according to traditional views). Sexism may also include positive stereotypes (e.g. “women are the best home cooks”), or target men (e.g., “men should not cry”)!

    Sexism due to `phrasing`: Texts may be sexist simply because of how the speaker phrases it–independently from what general beliefs or attitudes the speaker holds. 
    A message is sexist, for example, when it contains attacks, foul language, or derogatory depictions directed towards individuals because of their gender, e.g. by means of name-calling ("you bitch"), 
    attacks ("I’m going to kick her back to the kitchen"), objectification ("She’s stupid but I’d still do her"), inflammatory messages ("burn all women"). 
    However, just because a message is aggressive or uses offensive language does not mean that it is sexist!

    Sexism due to `context`: Texts could be mocking female perspectives from female politicians, minimize their political contributions and undermine their achievements. \
    It can also question their commitments to public office by implicating that they should focus more on their family commitments, and their political performance being compared to their capability in \
    familial setting. They may also publish appearance-centric criticism of the female politicians, unlike their male counterparts. They tone could be ironic and exaggerated, and often in the guise of humour.

    Please answer if the following text is any of the following 2 labels:
    1) sexist
    2) not sexist

    Please reply with either of the two options and nothing else.

    Here are a few examples to demonstrate how to label texts accordingly:
    1.  Text: 'Yes, but she was asking for it!'? When will people grow up and stop sexualizing women?!
        Label: not sexist
    2.  Text: The conference will cover various topics, including technology and art. Gender biases will not be tolerated.
        Label: not sexist
    3.  Text: If I was her, I would be annoyed at the misogyny.
        Label: not sexist
    4.  Text: Give her a pity party because of some news article. She has been vile attacking people for many years if they do not agree with her politics calling people scum.
        Label: not sexist
    5.  Text: Was she flashing her minge?
        Label: sexist
    6.  Text: In the debating chamber her high squeeky voice sounded like a hyena.
        Label: sexist
    7.  Text: Look the way she dressed! She's definitely asking for it.🙄
        Label: sexist
    8.  Text: Pulling the feminist card is bullshit. She does not get a free card.
        Label: sexist
    9. Text: Men will be men. It is true, and you should accept it.
        Label: sexist
    10. Text: The government's economic policies disproportionately affect women's access to job opportunities.
        Label: not sexist

    """

    human_message = f"""
    Now label the following text based on the past guidelines:
    Text: {prompt}
    Label:

    """

    # combine the prompts first
    total_prompt = system_message + human_message
    response = generate(total_prompt, tokenizer, model)

    if response not in listed_labels:
        response = 'not sexist'
    else:
        response = response

    return response     



def main():

    parser.add_argument('--dataset_name', help = 'give the name of the dataset', required = True, type = str)
    parser.add_argument('--model_name', help = 'give the name of the model', type= str,
            choices = ["google/flan-t5-xl", "google/flan-t5-xxl", "google/flan-t5-large", "google/flan-t5-base"],
            default='google/flan-t5-xl')
    parser.add_argument('--output_folder', help = 'give the name of the output folder', required = True, type = str)
    parser.add_argument('--output_file', help = 'give the name of the output file name', required = True, type = str)
    args = parser.parse_args()

    ## reading the data 
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name, return_dict=False)

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
                response = roleplay_flan_call(row, model, tokenizer)
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
                response = content_flan_call(row, model, tokenizer)
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
                response = zeroshot_flan_call(row, model, tokenizer)
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
                response = fewshot_flan_call(row, model, tokenizer)
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