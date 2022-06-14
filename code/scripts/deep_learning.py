# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 14:17:08 2022

@author: Emmanuelle R Nunes
"""

#Neural Networks
import pandas as pd
y_train = pd.read_pickle('data/processed/y_train.pkl.gz')['voted_up'].to_numpy()
y_test = pd.read_pickle('data/processed/y_test.pkl.gz')['voted_up'].to_numpy()
X_train = pd.read_pickle('data/processed/X_bigram_train.pkl.gz').to_numpy()
X_test = pd.read_pickle('data/processed/X_bigram_test.pkl.gz').to_numpy()

#Baseline Neural Network
#Here I build a simple sequential convolutional neural network. Although it only has dense and dropout layers, it stillperforms great.

from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras import Sequential
model = Sequential()

# hidden layers
model.add(Dense(500, input_dim = X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))

# output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=3, batch_size=32)
model.evaluate(X_test, y_test)
# acc: 0.9138


from talos import Scan
def dense_network(x_train, y_train, x_val, y_val, params):
    model = Sequential()

    # hidden layers
    model.add(Dense(params['dense'], input_dim=x_train.shape[1], activation=params['activation1']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['dense']*2, activation=params['activation1']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['dense']*0.5, activation=params['activation1']))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['dense']*0.75, activation=params['activation1']))
    # output layer
    model.add(Dense(1, activation=params['activation2']))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    out = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=5,
                    verbose=0)

    return out, model

params = {'dropout': [0.25, 0.5, 0.75], 
          'dense': [10, 50, 100, 500], 
          'activation1': ['relu', 'elu'], 
          'activation2': ['sigmoid', 'tanh']}

results = Scan(X_train, y_train, params=params, model=dense_network, experiment_name='grid')
results.best_model(metric='accuracy')

pd.read_csv('grid/060422143157.csv').sort_values('val_accuracy', ascending=False)

#Final Model
#The top performing model looks very similar to the first model I made, and performs nearly the same as well. I would like to expand my gridsearch when I get a chance, but for now this model is my best, so I wil save it to use on the unlabeled data.

model = Sequential()

# hidden layers
model.add(Dense(10, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10, activation='relu'))

# output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

model.evaluate(X_test, y_test)
model.save('../final_model/model.h5')
model.save_weights('../final_model/model_weights.h5')
# accu: 0.9098

# LSTM

from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalMaxPool1D, LSTM
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing import text, sequence

# opening the file in read mode
my_file = open("idprocessed_on_20220528.txt", "r")

# reading the file
data = my_file.read()

# replacing end splitting the text
# when newline ('\n') is seen.
app_ids = data.split("\n")
print(app_ids)
my_file.close()

y_train = pd.read_pickle('data/processed/y_train.pkl.gz')['voted_up'].to_numpy()
y_test = pd.read_pickle('data/processed/y_test.pkl.gz')['voted_up'].to_numpy()
X_train = pd.read_pickle('data/processed/X_preprocessed_train.pkl.gz').to_numpy()
X_test = pd.read_pickle('data/processed/X_preprocessed_test.pkl.gz').to_numpy()

tokenizer = text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train.flatten().tolist())
tokenized_list_train = tokenizer.texts_to_sequences(X_train.flatten().tolist())
tokenized_list_test = tokenizer.texts_to_sequences(X_test.flatten().tolist())
X_train_tokenized = sequence.pad_sequences(tokenized_list_train, maxlen=256)
X_test_tokenized = sequence.pad_sequences(tokenized_list_test, maxlen=256)
model = Sequential()

# hidden layers
model.add(Embedding(20000, 512))
model.add(LSTM(50, return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))

# output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_tokenized, y_train, epochs=3, batch_size=64)
model.evaluate(X_test_tokenized, y_test)
# acc: 0.9244

#Hyperparameter Tuning
from talos import Scan
def dense_network(x_train, y_train, x_val, y_val, params):
    model = Sequential()

    # hidden layers
    model.add(Embedding(20000, 512))
    model.add(LSTM(params['lstm'], return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(params['dropout1']))
    model.add(Dense(params['dense'], activation=params['activation']))
    model.add(Dropout(params['dropout2']))
    # output layer
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    out = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=params['epoch'],
                    verbose=1)

    return out, model

params = {'dropout1': [0.1, 0.25, 0.5, 0.75], 
          'dropout2': [0.1, 0.25, 0.5, 0.75],
          'dense': [10, 50, 100, 250],
          'activation': ['relu', 'sigmoid', 'tanh', 'elu'],
          'lstm': [10, 50, 100, 250],
          'epoch': [3, 5, 10, 20]}
results = Scan(X_train_tokenized, y_train, params=params, model=dense_network, experiment_name='grid')
results.best_model(metric='accuracy')

df = pd.read_csv('grid/060522111835.csv')
df.sort_values('val_accuracy', ascending=False)
# acc: 0.919704437	dropout1: 0.1	dropout2: 0.1	dense: 10	activation: relu
# lstm: 50	epoch: 3

# Final LSTM model

model = Sequential()

# hidden layers
model.add(Embedding(20000, 512))
model.add(LSTM(50, return_sequences=True))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.1))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.1))

# output layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_tokenized, y_train, epochs=3, batch_size=32, validation_split=0.1)

model.evaluate(X_test_tokenized, y_test)
# [los: 0.20516006648540497, acc: 0.9259893298149109]
model.save('../final_model/LSTM/model')
model.save_weights('../final_model/LSTM/_weights')