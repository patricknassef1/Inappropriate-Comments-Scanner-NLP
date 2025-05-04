# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 18:22:44 2025

@author: patri
"""

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split



# Data Preprocessing
dataset_train = pd.read_csv("jigsaw-toxic-comment-classification-challenge/train.csv/train.csv")

# If any of the types are 1 it will concider toxic (true or false)
dataset_train['toxic_label'] = dataset_train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].sum(axis=1) > 0


# Converts true or false to integer 0 or 1
dataset_train['toxic_label'] = dataset_train['toxic_label'].astype(int)

# Prepare texts and labels
texts = dataset_train['comment_text'].astype(str).values
labels = dataset_train['toxic_label'].values

# Tokenize: will create a dictionary that will translates words into numbers
# Will track the 16000 most common words, less frequancy is ignored
#if the tokenizer encounter a word that does get recognized it will use the <out of vocabulary> token 
tokenizer = Tokenizer(num_words=16000, oov_token='<OOV>')

# Reads all the comments from text
# Learns which words are common
# Assign unique number to word  ("the" → 1, "you" → 2)
tokenizer.fit_on_texts(texts)


# convert word to the corresponding number
# hi how are you
# 32 21  444 422
sequences = tokenizer.texts_to_sequences(texts)

# NN needs needs fixed length word so we cut the word to 150 words only
padded_sequences = pad_sequences(sequences, maxlen=150)


X_train=padded_sequences
y_train=labels








## RNN

model = Sequential()

# Embedding convert the word into 128 number(code) to capture meaning and relationships between words 
# input_dim = total words in dictionary + 1 (<00v>) 
#padding of the word
model.add(Embedding(input_dim=16001, output_dim=128, input_length=150))

model.add(LSTM(units=128, return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))  # Output: toxic (1) or not toxic (0)

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(
    X_train, 
    y_train, 
    epochs=10, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stopping]
)


#########################################################################################################

model.save('toxic_comment_lstm_model.h5')

import pickle

# Save tokenizer
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)


import numpy as np

# Save padded sequences
np.save('padded_sequences.npy', padded_sequences)


#########################################################################################################

from keras.models import load_model

model = load_model('toxic_comment_lstm_model.h5')



import pickle

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)



new_comment = ["shit"]
#You're the dumbest person I've ever seen.
#This is absolute garbage!
#you're an idiot!
#shit

#Thank you so much for your help!
#Can you please explain that a bit more?
#I totally agree with you.
#Well done, that was very insightful.
sequence = tokenizer.texts_to_sequences(new_comment)
padded = pad_sequences(sequence, maxlen=150)
prediction = model.predict(padded)
if prediction[0][0] > 0.5:
    print("Result: Toxic comment detected!")
else:
    print("Result: Clean comment.")
