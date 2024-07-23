import pandas as pd
import pickle
import random 

csv_file_path = 'data/Sentiment_V2.csv'
df = pd.read_csv(csv_file_path)

all_datas_0 = [] # Each row in the CSV is transformed into a variable object.
all_datas_2 = [] # Relation-type features are created. If reply.to is -1, it creates a relation with all other messages.
all_datas_3 = [] # Word-type features are created using the text and id columns.

for idx, row in df.iterrows():
    variables = {
        'name': row['id'],
        'isEvidence': True,
        'polarity': 'positive' if row['sentiment'] == 1 else ('negative' if row['sentiment'] == -1 else None),
        'gold_polarity': 'positive' if row['sentiment'] == 1 else ('negative' if row['sentiment'] == -1 else None),
        'prior': random.uniform(0, 1)  # Assign a random prior for simplicity
    }
    all_datas_0.append(variables)

for idx, row in df.iterrows():
    if row['reply.to'] != -1:
        relation = {
            'name1': row['id'],
            'name2': row['reply.to'],  # Assuming 'reply.to' contains related message IDs
            'rel_type': 'asp2asp_sequence_oppo'  # Example relation type, update as necessary
        }
        all_datas_2.append(relation)
    else:
        # Handle the case where the message is a reply to all
        for reply_id in df[df['id'] != row['id']]['id']:  # Relate to all other messages
            relation = {
                'name1': row['id'],
                'name2': reply_id,
                'rel_type': 'asp2asp_sequence_oppo'  # Example relation type, update as necessary
            }
            all_datas_2.append(relation)

# Create word-type features for all_datas[3]
for idx, row in df.iterrows():
    word_feature = {
        'name1': row['text'],  # Assuming 'text' contains the sentence
        'name2': row['id'],
        'rel_type': 'unary_feature'
    }
    all_datas_3.append(word_feature)

# Combine all_datas
all_datas = [all_datas_0, [], all_datas_2, all_datas_3]

# Save to pickle file
with open('data/test01_all_global_parse.pkl', 'wb') as f:
    pickle.dump(all_datas, f)

print("Pickle file created successfully.")

