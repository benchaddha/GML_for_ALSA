import pandas as pd

df = pd.read_excel('data/Sentiment_V2.xlsx')

df.to_csv ("Sentiment_V2.csv",  
                  index = None, 
                  header=True) 

