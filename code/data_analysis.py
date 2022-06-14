# -*- coding: utf-8 -*-
"""
Created on Sun May 29 16:25:36 2022

@author: Emmanuelle R Nunes
"""
import sys
sys.path.append('..')
import numpy as np
import pandas as pd
from scripts import preprocessing
from scripts import data_wrangling
from scripts import data_collection
from string import punctuation
from nltk.corpus import stopwords

reviews_df = data_wrangling.aggregate_reviews_to_pandas(appids[0:100])
reviews_df = reviews_df[['app_id', 'review', 'voted_up']]
reviews_df.dropna(inplace=True)
reviews_df.reset_index(inplace=True)
reviews_df

# unbalanced class - 87% positive vs 13% negative
reviews_df.voted_up.value_counts(normalize=True)
reviews_df.to_pickle('data/reviews_raw.pkl.gz')

reviews_df = pd.read_pickle('data/reviews_raw.pkl.gz')
reviews_df = reviews_df.sample(frac=1).reset_index(drop=True)
for i, df in enumerate(np.array_split(reviews_df, 10)):
    df.to_pickle(f'data/reviews_raw_{str(i)}.pkl.gz')

stopwords_list = stopwords.words('english') + list(punctuation) + ['`', '’', '…', '\n']

for i in range(10):
    print(i+1, 'of 10 started')
    df = pd.read_pickle(f'data/reviews_raw_{str(i)}.pkl.gz')
    
    y = df['voted_up'].to_numpy()
    pd.DataFrame(y, columns=['voted_up']).to_pickle(f'data/processed/y_{str(i)}.pkl.gz')
    
    X = df['review'].to_numpy()
    X = list(map(remove_markdown, X))
    X = list(map(remove_punctuation, X))
    X = list(map(tokenize, X))
    pd.DataFrame([' '.join(review) for review in X]).to_pickle(f'data/processed/X_preprocessed_{str(i)}.pkl.gz')
    
    X_stopword = []
    for review in X:
        X_stopword.append([word for word in review if word not in stopwords_list])
    pd.DataFrame([' '.join(review) for review in X_stopword]).to_pickle(f'data/processed/X_stopword_{str(i)}.pkl.gz')


from sklearn.model_selection import train_test_split
for data_name in ['y', 'X_stopword', 'X_preprocessed']:
    print('Starting', data_name)
    df = pd.read_pickle(f'data/processed/{data_name}_0.pkl.gz')
    
    for i in range(1, 10):
        df = df.append(pd.read_pickle(f'data/processed/{data_name}_{str(i)}.pkl.gz'))
        
    df_train, df_test = train_test_split(df, test_size=0.3, random_state=88)
    df_train.to_pickle(f'data/processed/{data_name}_train.pkl.gz')
    df_test.to_pickle(f'data/processed/{data_name}_test.pkl.gz')

X_train, y_train = df_train['review'].tolist(), df_train['voted_up'].tolist()
X_test, y_test = df_test['review'].tolist(), df_test['voted_up'].tolist()
len(X_train), len(y_train), len(X_test), len(y_test)


# feature engineering

# tf-idf

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(max_features=8000, stop_words=stopwords_list)
X_train_tf = pd.DataFrame(tf.fit_transform(X_train).todense(), columns=tf.get_feature_names())
X_test_tf = pd.DataFrame(tf.transform(X_test).todense(), columns=tf.get_feature_names())
X_train_tf.to_feather('data/processed/X_train_tf.feather')
X_test_tf.to_feather('data/processed/X_test_tf.feather')

#TF-IDF with Bigrams
#TF-IDF with Bigrams performed the best after running the models, so I pickled the vectorizer to use again later. When I get the ability to run bigger models and vectorizers, I may come back and try other levels of n-grams.

import pandas as pd
from pickle import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
tf_bigram = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
X_train = pd.read_pickle('data/processed/X_preprocessed_train.pkl.gz')[0]
X_train_bigram = pd.DataFrame(tf_bigram.fit_transform(X_train).todense(), columns=tf_bigram.get_feature_names())

X_train_bigram.to_pickle('data/processed/X_bigram_train.pkl.gz')
dump(tf_bigram, open('../final_model/tfidf_bigram_vectorizer.pk', 'wb'))
tf_bigram = load(open('../final_model/tfidf_bigram_vectorizer.pk', 'rb'))
X_test = pd.read_pickle('data/processed/X_preprocessed_test.pkl.gz')[0]
X_test_bigram = pd.DataFrame(tf_bigram.transform(X_test).todense(), columns=tf_bigram.get_feature_names())
X_test_bigram.to_pickle('data/processed/X_bigram_test.pkl.gz')
X_train_bigram.to_pickle('data/processed/X_bigram_train.pkl.gz')
X_test_bigram.to_pickle('data/processed/X_bigram_test.pkl.gz')
dump(tf_bigram, open('../final_model/tfidf_bigram_vectorizer.pk', 'wb'))

#Document Embeddings
from gensim.sklearn_api import D2VTransformer
from sklearn.preprocessing import MinMaxScaler
d2v = D2VTransformer()
X_train_embed = d2v.fit_transform(X_train_pre)
X_test_embed = d2v.transform(X_test_pre)

scaler = MinMaxScaler((1, 2))
X_train_embed = pd.DataFrame(scaler.fit_transform(X_train_embed))
X_test_embed = pd.DataFrame(scaler.transform(X_test_embed))

X_train_embed.columns = X_train_embed.columns.astype(str)
X_test_embed.columns = X_test_embed.columns.astype(str)
X_train_embed.to_feather('../data/processed/X_train_embed.feather')
X_test_embed.to_feather('../data/processed/X_test_embed.feather')

# baseline model
 
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
def get_model_metrics(X_train, y_train, X_test, y_test, model, model_name, data_name):
    
    model.fit(X_train, y_train)
    y_train_hat = model.predict(X_train)
    y_test_hat = model.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_train_hat)
    pre_train = precision_score(y_train, y_train_hat)
    rec_train = recall_score(y_train, y_train_hat)
    f1_train = f1_score(y_train, y_train_hat, average='macro')
    
    acc_test = accuracy_score(y_test, y_test_hat)
    pre_test = precision_score(y_test, y_test_hat)
    rec_test = recall_score(y_test, y_test_hat)
    f1_test = f1_score(y_test, y_test_hat, average='macro')
    
    metrics = {'Model': model_name,
               'Processing': data_name,
               'Test Accuracy': acc_test,
               'Test Precision': pre_test,
               'Test Recall': rec_test,
               'Test F1': f1_test,
               'Train Accuracy': acc_train,
               'Train Precision': pre_train,
               'Train Recall': rec_train,
               'Train F1': f1_train}
    
    return metrics


datasets = [('TF-IDF with Bigrams', 'bigram'),
            ('Preprocessed', 'preprocessed')]
models = [('Logistic Regression', LogisticRegression(solver='saga')),
          ('Multinomial Naive Bayes', MultinomialNB()),
          ('Random Forest', RandomForestClassifier())]
metrics = []
y_train = pd.read_pickle('data/processed/y_train.pkl.gz')['voted_up'].to_numpy()
y_test = pd.read_pickle('data/processed/y_test.pkl.gz')['voted_up'].to_numpy()

for data_name, file in datasets:
    X_train = pd.read_pickle(f'data/processed/X_{file}_train.pkl.gz').to_numpy()
    X_test = pd.read_pickle(f'data/processed/X_{file}_test.pkl.gz').to_numpy()
    for model_name, model in models:
        print(model_name, data_name)
        metrics.append(get_model_metrics(X_train, y_train, X_test, y_test, model, model_name, data_name))

metrics.append(get_model_metrics(X_train, y_train, X_test, y_test, DummyClassifier()))

metrics_df = pd.DataFrame(metrics)
metrics_df.sort_values(by='Test Accuracy', ascending=False)
metrics_df.to_csv("metrics.csv")

#Gridsearch
#Here I performed a gridsearch on the random forest and logistic regression models using just the bigram data, as it performed the best. Naive Bayes models do not have any hyperparameters to tune, and so there is no grid search to perform on it.

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
y_train = pd.read_pickle('data/processed/y_train.pkl.gz')['voted_up'].to_numpy()
X_train = pd.read_pickle('data/processed/X_bigram_train.pkl.gz').to_numpy()
param_grid_lr = {'C': [0.1, 1, 10],
                 'class_weight': ['balanced', None],
                 'solver': ['saga']}
gs_lr = GridSearchCV(estimator=LogisticRegression(), param_grid=param_grid_lr, scoring='f1_macro', cv=3, verbose=5)
gs_lr.fit(X_train, y_train)
gs_lr.best_params_

param_grid_rf = {'n_estimators': [100, 500],
                 'max_features': ['auto', 150],
                 'class_weight': ['balanced', None]}
gs_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid_rf, scoring='f1_macro', cv=3, verbose=5)
gs_rf.fit(X_train, y_train)
gs_rf.best_params_

y_train = pd.read_pickle('../data/processed/y_train.pkl.gz')['voted_up'].to_numpy()
y_test = pd.read_pickle('../data/processed/y_test.pkl.gz')['voted_up'].to_numpy()
X_train = pd.read_pickle('../data/processed/X_bigram_train.pkl.gz').to_numpy()
X_test = pd.read_pickle('../data/processed/X_bigram_test.pkl.gz').to_numpy()
lr_final = LogisticRegression(C=10, solver='saga')
metrics = get_model_metrics(X_train, y_train, X_test, y_test, lr_final, 'Logistic Regression', 'TF-IDF with Bigrams')
print(metrics)

metrics

lr_final = LogisticRegression(C=10, solver='saga')
nb_final = MultinomialNB()
rf_final = RandomForestClassifier(max_features=150)

final_metrics = []
print('starting Logistic Regression model')
final_metrics.append(get_model_metrics(X_train, y_train, X_test, y_test, lr_final, 'Logistic Regression', 'TF-IDF with Bigrams'))
print('starting Naive Bayes model')
final_metrics.append(get_model_metrics(X_train, y_train, X_test, y_test, nb_final, 'Multinomial Naive Bayes', 'TF-IDF with Bigrams'))
print('starting Random Forest model')
final_metrics.append(get_model_metrics(X_train, y_train, X_test, y_test, rf_final, 'Random Forest', 'TF-IDF with Bigrams'))
print('completed models')

final_metrics_df = pd.DataFrame(final_metrics)
final_metrics_df.sort_values(by='Test Accuracy', ascending=False)

#Save Model
#Even though I will also be creating a neural network model, I still want to save this best logistic regression model. I can try to use it as a backup in case the neural network model is too big to upload to heroku.

import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
y_train = pd.read_pickle('../data/processed/y_train.pkl.gz')['voted_up'].to_numpy()
y_test = pd.read_pickle('../data/processed/y_test.pkl.gz')['voted_up'].to_numpy()
X_train = pd.read_pickle('../data/processed/X_bigram_train.pkl.gz').to_numpy()
X_test = pd.read_pickle('../data/processed/X_bigram_test.pkl.gz').to_numpy()
model = LogisticRegression(C=10, solver='saga')
model.fit(X_train, y_train)
pickle.dump(model, open('../final_model/sklearn-logreg/model.pk', 'wb'))

# do LSTM and RNN models
## next with embedding methods -already trained models with twitter
# then apply models to twitter 
# then move to transformers
## course by nvidia - free for students
### like bert models

## deep learning with python chapter 6.2