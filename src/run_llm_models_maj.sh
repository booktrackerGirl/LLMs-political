#!/bin/bash

# Create output directory if it doesn't exist
mkdir -p ./data/llm_data_maj/

# Run flan-t5 model
echo "Starting FLAN-T5 predictions..."
python ./src/get_flant5_pred.py --dataset_name ./data/matched_df.csv --output_folder ./data/llm_data_maj --output_file flan_t5
echo "FLAN-T5 complete."
sleep 60

# Run mistral model
echo "Starting Mistral predictions..."
python ./src/get_mistral_pred.py --dataset_name ./data/matched_df.csv --output_folder ./data/llm_data_maj --output_file mistral
echo "Mistral complete."
sleep 60

# Run vicuna model
echo "Starting Vicuna predictions..."
python ./src/get_vicuna_pred.py --dataset_name ./data/matched_df.csv --output_folder ./data/llm_data_maj --output_file vicuna
echo "Vicuna complete."
sleep 60

# Run alpaca model
echo "Starting Alpaca predictions..."
python ./src/get_alpaca_pred.py --dataset ./data/matched_df.csv --output_folder ./data/llm_data_maj --output_file alpaca
echo "Alpaca complete."

