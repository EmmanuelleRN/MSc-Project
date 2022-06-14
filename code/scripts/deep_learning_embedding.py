# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 11:40:55 2022

@author: Emmanuelle R Nunes
"""

#Using GloVe word embeddings
#TensorFlow enables you to train word embeddings. However, this process not only requires a lot of data but can also be time and resource-intensive. To tackle these challenges you can use pre-trained word embeddings. Let's illustrate how to do this using GloVe (Global Vectors) word embeddings by Stanford.  These embeddings are obtained from representing words that are similar in the same vector space. This is to say that words that are negative would be clustered close to each other and so will positive ones.
#
#The first step is to obtain the word embedding and append them to a dictionary. After that, you'll need to create an embedding matrix for each word in the training set. Let's start by downloading the GloVe word embeddings.
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tensorflow.keras.preprocessing import text, sequence

import pandas as pd
import zipfile
with zipfile.ZipFile('embedding/glove/glove.6B.zip', 'r') as zip_ref:
    zip_ref.extractall('embedding/glove/glove')
    
y_train = pd.read_pickle('data/processed/y_train.pkl.gz')['voted_up'].to_numpy()
y_test = pd.read_pickle('data/processed/y_test.pkl.gz')['voted_up'].to_numpy()
X_train = pd.read_pickle('data/processed/X_preprocessed_train.pkl.gz').to_numpy()
X_test = pd.read_pickle('data/processed/X_preprocessed_test.pkl.gz').to_numpy()

tokenizer = text.Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train.flatten().tolist())
tokenized_list_train = tokenizer.texts_to_sequences(X_train.flatten().tolist())
tokenized_list_test = tokenizer.texts_to_sequences(X_test.flatten().tolist())
X_train_tokenized = sequence.pad_sequences(tokenized_list_train, maxlen=300)
X_test_tokenized = sequence.pad_sequences(tokenized_list_test, maxlen=300)

word_index = tokenizer.word_index
    
import numpy as np
embeddings_index = {}
f = open('embedding/glove/glove/glove.6B.300d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

#The next step is to create a word embedding matrix for each word in the word index that you obtained earlier. If a word doesn't have an embedding in GloVe it will be presented with a zero matrix.

max_length = 300

embedding_matrix = np.zeros((len(word_index) + 1, max_length))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

#Creating the Keras embedding layer
#The next step is to use the embedding you obtained above as the weights to a Keras embedding layer. You also have to set the trainable parameter of this layer to False so that is not trained. If training happens again the weights will be re-initialized. This will be similar to training a word embedding from scratch. There are also a couple of other things to note:
#
#The Embedding layer takes the first argument as the size of the vocabulary. 
#1 is added because 0 is usually reserved for padding
#The input_length is the length of the input sequences
#The output_dim is the dimension of the dense embedding

from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional

embedding_layer = Embedding(input_dim = len(word_index) + 1,
                            output_dim = max_length,
                            weights = [embedding_matrix],
                            input_length = max_length,
                            trainable = False)

#Creating the TensorFlow model
#The next step is to use the embedding layer in a Keras model. Let's define the model as follows:
#
#The embedding layer as the first layer
#Two Bidirectional LSTM layers to ensure that information flows in both directions
#The fully connected layer, and
#A final layer responsible for the final output

from tensorflow.keras.models import Sequential
model = Sequential([
    embedding_layer,
    Bidirectional(LSTM(150, return_sequences=True)), 
    Bidirectional(LSTM(150)),
    Dense(128, activation='relu'),
   Dense(1, activation='sigmoid')
])

#Training the model
#The next step is to compile and train the model.

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

#As the model is training, you can set an EarlyStopping callback to stop the training process once the mode stops improving. You can also set a TensorBoard callback to quickly see the model's performance later.

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
%load_ext tensorboard

log_folder = 'logs'
callbacks = [
            EarlyStopping(patience = 10),
            TensorBoard(log_dir=log_folder)
            ]
num_epochs = 600
history = model.fit(X_train_tokenized, y_train, epochs=num_epochs, validation_data=(X_test_tokenized, y_test),callbacks=callbacks)

#You can use the evaluate method to quickly check the performance of the model.

loss, accuracy = model.evaluate(X_test_tokenized,y_test)
print('Test accuracy :', accuracy)
# Test accuracy : 0.8708462715148926

# next step:
# all steps with twitter data 
# ask for nvidia videos7   
# workflow