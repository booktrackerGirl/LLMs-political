#!/usr/bin/bash
# This script prompts the user to enter a output file name, checks if it is one of the model names, and based on that form the command line.
# Prompt the user to enter a file name.

echo -n "Enter LLM file name: "
read output_file
echo "Hello, you chose $output_file for this run!"

export dataset_name='./data/sexist_political_data.csv'
export output_folder='./data/llm_data'

# check based on the output file name
if [[ "$output_file" == 'flan_t5' ]]; then
    echo "Starting to run Flan-T5 model"
    chmod u+x src/get_flant5_pred.py
    python src/get_flant5_pred.py --dataset_name $dataset_name --model_name 'google/flan-t5-xl' --output_folder $output_folder --output_file $output_file
# for the next ones, the bash comands we are going to set the default values for most of the arguments
elif [[ "$output_file" == 'alpaca' ]]; then
    echo "Starting to run Alpaca model"
    chmod u+x src/get_alpaca_pred.py
    python src/get_alpaca_pred.py --dataset_name $dataset_name --output_folder $output_folder --output_file $output_file
elif [[ "$output_file" == 'mistral' ]]; then
    echo "Starting to run Mistral model"
    chmod u+x src/get_mistral_pred.py
    python src/get_mistral_pred.py --dataset_name $dataset_name --output_folder $output_folder --output_file $output_file
elif [[ "$output_file" == 'vicuna' ]]; then
    echo "Starting to run Vicuna model"
    chmod u+x src/get_vicuna_pred.py
    python src/get_vicuna_pred.py --dataset_name $dataset_name --output_folder $output_folder --output_file $output_file
elif [[ "$output_file" == 'gpt' ]]; then
    echo "Starting to run Chat-GPT model"
    chmod u+x src/get_gpt_pred.py
    python src/get_gpt_pred.py --dataset_name $dataset_name --output_folder $output_folder --output_file $output_file
else
    echo 'You have not chosen any of the right options. Try again!'
fi