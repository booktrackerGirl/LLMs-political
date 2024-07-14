# LLMs-political

This is the repository for the paper <b>'We are confident in trusting Language Models for detecting online sexism in political discourse, but are they good?'</b>.


- <b>RQ1:</b> Does some external event triggering more influx of conversations centering around female politicians, consequently mean rise in online sexism?
  - Data: [sexist_political_metadata.csv](./data/#sexist_political_metadata.csv)
  - Code: [hypothesis_testing_triggerevents.R](./src/#hypothesis_testing_triggerevents.R)
- <b>RQ2:</b> Can the same online sexism be efficiently captured by the LLMs through prompt engineering? Does it work better with more descriptive prompting?
  - Data: sexist_political_data.csv which contains the text column of the annotated data
  - Predictions:
      - ChatGPT - [get_gpt_pred.py](./src/#get_gpt_pred.py)
      - Flan-T5 - [get_flant5_pred.py](./src/#get_flant5_pred.py)
      - Alpaca - [get_alpaca_pred.py](./src/#get_alpaca_pred.py)
      - Vicuna - [get_vicuna_pred.py](./src/#get_vicuna_pred.py)
      - Mistral - [get_mistral_pred.py](./src/#get_mistral_pred.py)
  - Analysis: [performance_analysis.ipynb](./src/#performance_analysis.ipynb)
- <b>RQ3:</b> Can we compute the confidence of LLMs in their predictions? If so, how effective is it?
  - Data:
      - [hypothesis_data.csv](./data/entropy_data/#hypothesis_data.csv)
      - Files from the folder [entropy_data](./data/#entropy_data)
  - Analysis:
     - [hypothesis_testing_entropy.R](./src/#hypothesis_testing_entropy.R)
     - [entropy_analysis.ipynb](./src/#entropy_analysis.ipynb)
