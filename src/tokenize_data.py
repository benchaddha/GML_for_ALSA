# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# tokenize_data.py
#
# Identification: /tokenize_data.py
#
# Ben Chaddha
#
# ===----------------------------------------------------------------------===//
from transformers import RobertaTokenizer, RobertaModel
import torch

# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
file_path = '/Users/benjaminchaddha/School/Junior Year/SONIC/GML_for_ALSA-master/data/Sentiment_V2.xlsx'

def get_embeddings(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def tokenize_and_embed(df):
    df['embeddings'] = df['text'].apply(lambda x: get_embeddings(x)[0])
    return df

if __name__ == "__main__":
    import read_data
    df = read_data.read_data('file_path')
    df = tokenize_and_embed(df)
    print(df.head())

