#!/usr/bin/bash

python src/semantic_entropy.py --dataset_name './data/sexist_political_data.csv' --model_name 'google/flan-t5-xl' --output_folder './data/entropy_data' --output_file 'flan1'

python src/semantic_entropy.py --dataset_name './data/sexist_political_data.csv' --model_name 'google/flan-t5-xl' --output_folder './data/entropy_data' --output_file 'flan2'

python src/semantic_entropy.py --dataset_name './data/sexist_political_data.csv' --model_name 'google/flan-t5-xl' --output_folder './data/entropy_data' --output_file 'flan3'
