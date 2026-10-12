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
    # Strip leading/trailing quotes that might be left from CSV parsing
    text = text.strip('"')
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Replace @mentions with @MENTION
    text = re.sub(r'@\w+', '@MENTION', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Split into words
    words = text.split()
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    # Join back
    text = ' '.join(words)
    return text

# Read the first dataset
df1 = pd.read_csv('data/sexist_political_data.csv')
# Create normalized text
df1['normalized_text'] = df1['text'].apply(normalize_text)
# Create a mapping from normalized_text to uniqueID
norm_to_id = dict(zip(df1['normalized_text'], df1['uniqueID']))

# Read the second dataset
df2 = pd.read_csv('data/majority_data_1.csv')
# Normalize text
df2['normalized_text'] = df2['text'].apply(normalize_text)

# Filter df2 to only rows where normalized_text is in df1
filtered_df = df2[df2['normalized_text'].isin(norm_to_id.keys())].copy()

# Also calculate exact matches for comparison
exact_matches = df2[df2['text'].isin(df1['text'])]
print(f"Exact matches: {len(exact_matches)}")
print(f"Normalized matches: {len(filtered_df)}")

# Add uniqueID column
filtered_df['uniqueID'] = filtered_df['normalized_text'].map(norm_to_id)

# Drop the normalized_text column
filtered_df = filtered_df.drop(columns=['normalized_text'])

# Reorder columns to have uniqueID first
cols = ['uniqueID'] + [col for col in filtered_df.columns if col != 'uniqueID']
filtered_df = filtered_df[cols]

# Save to new CSV
filtered_df.to_csv('data/filtered_majority_with_ids.csv', index=False)

print("New dataset created: data/filtered_majority_with_ids.csv")