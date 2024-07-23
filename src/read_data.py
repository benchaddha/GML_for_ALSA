# ===----------------------------------------------------------------------===//
#
#                         GML for ALSA
#
# read_data.py
#
# Identification: /read_data.py
#
# Ben Chaddha
#
# ===----------------------------------------------------------------------===//

import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch

def read_data(file_path):
    '''
    Read data from the provided Excel file.
    '''
    file_path = '/Users/benjaminchaddha/School/Junior Year/SONIC/GML_for_ALSA-master/data/Sentiment_V2.xlsx'
    # Read Excel File
    df = pd.read_excel(file_path)

    df = df[['text', 'sentiment']]

    return df

if __name__ == "__main__":
    df = read_data('../your_excel_file.xlsx')
    print(df.head())
