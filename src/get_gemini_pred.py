#!/usr/bin/env python

import re
import os
import gc
import time
import math
import random
import argparse
import warnings
import numpy as np
import pandas as pd
import api_keys

from google import genai
from google.genai import types

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Process text classification with Gemini Flash in the dataset.')

# -----------------------------
# GEMINI AUTHENTICATION
# -----------------------------
client = genai.Client(api_key=api_keys.GEMINI_API_KEY)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def cal_avg_prob_gemini(logprobs_result):
    try:
        all_step_probs = []

        if logprobs_result is None:
            return []

        if hasattr(logprobs_result, 'top_candidates') and logprobs_result.top_candidates is not None:
            for step in logprobs_result.top_candidates:
                probs = []
                for cand in step.candidates:
                    try:
                        probs.append(math.exp(cand.log_probability))
                    except Exception:
                        continue
                if len(probs) > 0:
                    all_step_probs.append(float(np.mean(probs)))

        elif hasattr(logprobs_result, 'chosen_candidates') and logprobs_result.chosen_candidates is not None:
            for cand in logprobs_result.chosen_candidates:
                try:
                    all_step_probs.append(math.exp(cand.log_probability))
                except Exception:
                    continue

        return all_step_probs
    except Exception:
        return []


def cal_transition_scores_gemini(logprobs_result):
    try:
        scores = []
        if logprobs_result is None:
            return []

        if hasattr(logprobs_result, 'chosen_candidates') and logprobs_result.chosen_candidates is not None:
            for cand in logprobs_result.chosen_candidates:
                try:
                    scores.append(float(cand.log_probability))
                except Exception:
                    continue
        return scores
    except Exception:
        return []


def cal_ppl_gemini(transition_scores):
    try:
        if len(transition_scores) == 0:
            return 1
        return float(np.exp(-np.mean(np.array(transition_scores))))
    except Exception:
        return 1


# -----------------------------
# EXPONENTIAL RETRY GEMINI CALL
# -----------------------------
def gemini_generate(system_prompt, human_text, model_name, token_num, temperature, max_retries=8):
    contents = [
        types.Content(role='user', parts=[types.Part(text=f'''{system_prompt}

{human_text}''')])
    ]

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=token_num,
                    response_logprobs=True,
                    logprobs=20,
                    candidate_count=1
                )
            )

            text_output = response.text.strip() if response.text is not None else 'neither'

            logprobs_result = None
            try:
                if response.candidates is not None and len(response.candidates) > 0:
                    logprobs_result = response.candidates[0].logprobs_result
            except Exception:
                logprobs_result = None

            avg_probs = cal_avg_prob_gemini(logprobs_result)
            transition_scores = cal_transition_scores_gemini(logprobs_result)
            ppl = cal_ppl_gemini(transition_scores)

            return text_output, avg_probs, ppl

        except Exception as e:
            wait_time = min((2 ** attempt) + random.uniform(0,1), 300)
            print(f'Gemini API failed on attempt {attempt+1}: {e}')
            print(f'Waiting {wait_time} seconds before retrying...')
            time.sleep(wait_time)

    return 'neither', [], 1


# -----------------------------
#
# system_prompt stays untouched in wording
# human_text stays untouched in wording
# only transport structure changed from HF chat_template to Gemini contents[]
# -----------------------------
roleplay_prompt = f'''
    You are an expert in understanding slight nuances in the text, even when presented with texts that lack enough context.. \
    You are well-versed with the political discourse/scenario in the United Kingdom since 2018, especially in social media platforms like Twitter. \
    You will act as a text classification assistant, experienced at understanding the presense of sexist attitudes in textual contents.
    
    Classify the input text by assigning them a label from the following list: 
    1) sexist
    2) not sexist

    The model should generate ONLY ONE of these labels as the output label, and nothing more or different. 
    '''

content_prompt = f"""
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

    Classify the input text by assigning them a label from the following list: 
    1) sexist
    2) not sexist

    The model should generate ONLY ONE of these labels as the output label, and nothing more or different. 
    
    """

zeroshot_prompt = f"""
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
   
    Classify the input text by assigning them a label from the following list: 
    1) sexist
    2) not sexist

    The model should generate ONLY ONE of these labels as the output label, and nothing more or different. 

    """ 

fewshot_prompt = f"""
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

    Classify the input text by assigning them a label from the following list: 
    1) sexist
    2) not sexist

    The model should generate ONLY ONE of these labels as the output label, and nothing more or different. 

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




# -----------------------------
# GENERIC LOOP FUNCTION
# -----------------------------
def run_prompt_block(data, prompt_name, system_prompt, args):

    output_results = []
    vocab_probs = []
    ppl_scores = []

    len_of_df = len(data)
    num = 0
    flag = False

    for row in data['text']:
        rate_limit_per_minute = 20
        delay = 60.0 / rate_limit_per_minute

        if num == len_of_df:
            flag = True
            print(f'Reached the {len_of_df}')
            break

        else:
            try:
                prompt = re.sub(r"\s+", " ", str(row))
                prompt = prompt.encode(encoding='ASCII', errors='ignore').decode()

                human_text = f'''
                Label the following text using the aforementioned instructions.
                Text: {prompt}
                Label:
                '''

                results = []
                avgs = []
                ppls = []

                for i in range(args.num_samples):
                    result, avg_probs, ppl = gemini_generate(
                        system_prompt,
                        human_text,
                        args.model,
                        args.token_num,
                        args.temperature
                    )

                    results.append(result)
                    avgs.append(avg_probs)
                    ppls.append(ppl)

                    del result, avg_probs, ppl
                    gc.collect()

                    # API COOLDOWN BETWEEN SAMPLES
                    time.sleep(delay)

                output_results.append(results)
                vocab_probs.append(avgs)
                ppl_scores.append(ppls)

                print(f'Finished iteration {num} for {prompt_name}')
                num += 1

                del results, avgs, ppls
                gc.collect()
                time.sleep(delay)

            except Exception as e:
                print(f'Row-level failure at {num} for {prompt_name}: {e}')
                print('Applying exception-safe completion...')
                output_results.append(['neither'] * args.num_samples)
                vocab_probs.append([[] for _ in range(args.num_samples)])
                ppl_scores.append([1 for _ in range(args.num_samples)])
                num += 1
                time.sleep(300)

        if flag == True:
            print('Reached the whole length of the dataset. Stopping now')
            break

    return output_results, vocab_probs, ppl_scores


# -----------------------------
# MAIN
# -----------------------------
def main():

    parser.add_argument('--dataset', help='give the name of the folder where our datasets are', required=True, type=str)
    parser.add_argument('--model', help='give the name of the model', type=str, default='gemini-2.0-flash')
    parser.add_argument('--output_folder', help='give the name of the output folder', required=True, type=str)
    parser.add_argument('--output_file', help='give the name of the output file name', required=True, type=str)
    parser.add_argument('--token_num', help='give the number of maximum tokens to be generated', required=True, type=int, default=12)
    parser.add_argument('--temperature', help='provide the temperature score', required=True, type=float, default=0.3)
    parser.add_argument('--num_samples', type=int, default=3)
    args = parser.parse_args()

    data = pd.read_csv(f'{args.dataset}.csv', encoding='utf-8')
    print(len(data))

    # BLOCK 1
    r1, v1, p1 = run_prompt_block(data, '1st one', roleplay_prompt, args)
    data['roleplay_results'] = r1
    data['roleplay_vocab_probs'] = v1
    data['roleplay_ppl'] = p1
    data.to_csv(f'{args.output_folder}/{args.output_file}.csv')
    time.sleep(300)
    gc.collect()

    # BLOCK 2
    r2, v2, p2 = run_prompt_block(data, '2nd one', content_prompt, args)
    data['content_results'] = r2
    data['content_vocab_probs'] = v2
    data['content_ppl'] = p2
    data.to_csv(f'{args.output_folder}/{args.output_file}.csv')
    time.sleep(300)
    gc.collect()

    # BLOCK 3
    r3, v3, p3 = run_prompt_block(data, '3rd one', zeroshot_prompt, args)
    data['zeroshot_results'] = r3
    data['zeroshot_vocab_probs'] = v3
    data['zeroshot_ppl'] = p3
    data.to_csv(f'{args.output_folder}/{args.output_file}.csv')
    time.sleep(300)
    gc.collect()

    # BLOCK 4
    r4, v4, p4 = run_prompt_block(data, '4th one', fewshot_prompt, args)
    data['fewshot_results'] = r4
    data['fewshot_vocab_probs'] = v4
    data['fewshot_ppl'] = p4
    data.to_csv(f'{args.output_folder}/{args.output_file}.csv')
    time.sleep(300)
    gc.collect()

    del args


if __name__ == '__main__':
    main()
