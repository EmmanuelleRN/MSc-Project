# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:38:50 2022

@author: Emmanuelle R Nunes
"""
import pandas as pd
import numpy as np

twitter_data = pd.read_csv("../twitter data/training_1600000_processed_noemoticon.csv",
                           encoding = "ISO-8859-1",
                           names=["sentiment", "ids", "date", "flag", "user", "text"])

twitter_df = twitter_data[['sentiment', 'text']]
twitter_df.dropna(inplace=True)
twitter_df.reset_index(inplace=True)

# balanced class
twitter_df.sentiment.value_counts(normalize=True)
twitter_df.to_pickle('twitter data/reviews_raw.pkl.gz')

twitter_df = pd.read_pickle('../twitter data/reviews_raw.pkl.gz')
twitter_df = twitter_df.sample(frac=1).reset_index(drop=True)
for i, df in enumerate(np.array_split(twitter_df, 10)):
    df.to_pickle(f'../twitter data/reviews_raw_{str(i)}.pkl.gz')

stopwords_list = stopwords.words('english') + list(punctuation) + ['`', '’', '…', '\n']

for i in range(10):
    print(i+1, 'of 10 started')
    df = pd.read_pickle(f'../twitter data/reviews_raw_{str(i)}.pkl.gz')
    
    y = df['sentiment'].to_numpy()
    pd.DataFrame(y, columns=['sentiment']).to_pickle(f'../twitter data/processed/y_{str(i)}.pkl.gz')
    
    X = df['text'].to_numpy()
    X = list(map(remove_markdown, X))
    X = list(map(remove_punctuation, X))
    X = list(map(tokenize, X))
    pd.DataFrame([' '.join(text) for text in X]).to_pickle(f'../twitter data/processed/X_preprocessed_{str(i)}.pkl.gz')
    
    X_stopword = []
    for review in X:
        X_stopword.append([word for word in review if word not in stopwords_list])
    pd.DataFrame([' '.join(review) for review in X_stopword]).to_pickle(f'../twitter data/processed/X_stopword_{str(i)}.pkl.gz')

from sklearn.model_selection import train_test_split
for data_name in ['y', 'X_stopword', 'X_preprocessed']:
    print('Starting', data_name)
    df = pd.read_pickle(f'../twitter data/processed/{data_name}_0.pkl.gz')
    
    for i in range(1, 10):
        df = df.append(pd.read_pickle(f'../twitter data/processed/{data_name}_{str(i)}.pkl.gz'))
        
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=88)
    df_train.to_pickle(f'data/processed/{data_name}_train.pkl.gz')
    df_test.to_pickle(f'data/processed/{data_name}_test.pkl.gz')

X_train, y_train = df_train['review'].tolist(), df_train['voted_up'].tolist()
X_test, y_test = df_test['review'].tolist(), df_test['voted_up'].tolist()
len(X_train), len(y_train), len(X_test), len(y_test)