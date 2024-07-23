# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# main.py
#
# Identification: /main.py
#
# Ben Chaddha
#
# ===----------------------------------------------------------------------===//

import os
from src import read_data, tokenize_data, prepare_data, save_data

def main():
    # Step 1: Read the data
    df = read_data.read_data('your_excel_file.xlsx')
    
    # Step 2: Tokenize and get embeddings
    df = tokenize_data.tokenize_and_embed(df)
    
    # Step 3: Prepare data structures
    variables, features, easys = prepare_data.prepare_data(df)
    
    # Step 4: Save the data
    save_data.save_data(variables, features, easys, 'data/test01_variables.pkl', 'data/test01_features.pkl', 'data/test01_easys.csv')
    print("Data preprocessing complete and saved successfully.")

if __name__ == "__main__":
    main()
