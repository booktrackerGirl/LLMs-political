import pandas as pd
import re
import string

# Simple stop words list
stop_words = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
])

def normalize_text(text):
    if not isinstance(text, str):
        return ''
    text = text.strip('"')
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '@MENTION', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)
    return text

# Read datasets
df1 = pd.read_csv('data/sexist_political_data.csv')
df1['normalized_text'] = df1['text'].apply(normalize_text)
norm_to_id = dict(zip(df1['normalized_text'], df1['uniqueID']))

df2 = pd.read_csv('data/majority_data_1.csv')
df2['normalized_text'] = df2['text'].apply(normalize_text)

# Calculate stats
current_matches = len(df2[df2['normalized_text'].isin(norm_to_id.keys())])
total_rows_df2 = len(df2)
unique_texts_df2 = df2['normalized_text'].nunique()
additional_without_matching = total_rows_df2 - current_matches

print(f"Total rows in majority_data_1: {total_rows_df2}")
print(f"Unique normalized texts in majority_data_1: {unique_texts_df2}")
print(f"Current matches (with sexist_political_data constraint): {current_matches}")
print(f"Additional rows without matching constraint: {additional_without_matching}")
