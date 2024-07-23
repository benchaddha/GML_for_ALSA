# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# save_data.py
#
# Identification: /save_data.py
#
# Ben Chaddha
#
# ===----------------------------------------------------------------------===//
import pandas as pd
import pickle
import os

def save_data(variables, features, easys, variables_path, features_path, easys_path):
    os.makedirs(os.path.dirname(variables_path), exist_ok=True)
    os.makedirs(os.path.dirname(features_path), exist_ok=True)
    os.makedirs(os.path.dirname(easys_path), exist_ok=True)
    
    with open(variables_path, 'wb') as f:
        pickle.dump(variables, f)

    with open(features_path, 'wb') as f:
        pickle.dump(features, f)

    # Save easy instances to a CSV file
    easy_df = pd.DataFrame(easys)
    easy_df.to_csv(easys_path, index=False)

    if __name__ == "__main__":
        import prepare_data

        df = prepare_data.read_data('../data/Sentiment_V2.xlsx')
        df = prepare_data.tokenize_and_embed(df)
        variables, features, easys = prepare_data.prepare_data(df)
        save_data(variables, features, easys, '../data/test01_variables.pkl', '../data/test01_features.pkl', '../data/test01_easys.csv')
        print("Data saved successfully.")