from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, SpatialDropout1D, Conv1D
from keras.layers import GlobalMaxPooling1D, Bidirectional, Embedding

import numpy as np
import tensorflow as tf

from constants import TWEET, ES_ODIO, MODEL_TYPES

def getMaxSize(dataset):
  max_size = 0
  for line in dataset:
    line_size = len(line)
    if (line_size > max_size):
      max_size = line_size

  return max_size

class Model():
  # We need the val_dataset in the constructor to find the max size
  def __init__(self, model_type, train_dataset, val_dataset, epochs):
    self.train_dataset = train_dataset[TWEET]
    self.train_labels = np.array(train_dataset[ES_ODIO])
    self.val_dataset = val_dataset[TWEET]
    self.val_labels = np.array(val_dataset[ES_ODIO])
    self.max_size =  max(getMaxSize(train_dataset[TWEET]), getMaxSize(val_dataset[TWEET]))
    self.model = None
    self.epochs= None
    self.train_padded_docs = None
    self.type = model_type
    self.initModel()

  def chooseModel(self, vocab_size, embedding_matrix):
    model = Sequential()
    e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=self.max_size, trainable=False)
    model.add(e)

    if (self.type == MODEL_TYPES["SIMPLE"]):
      model.add(Flatten())

    elif (self.type == MODEL_TYPES["LSTM1"]):
      model.add(LSTM(128, dropout=0.5))

    elif (self.type == MODEL_TYPES["LSTM2"]):
      model.add(SpatialDropout1D(0.25))
      model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
      model.add(Dropout(0.2))

    elif (self.type == MODEL_TYPES["BIDIRECTIONAL"]):
      model.add(Bidirectional(LSTM(128,dropout=0.3)))

    else:
      model.add(Conv1D(128, 5, activation='relu'))
      model.add(GlobalMaxPooling1D())
    
    # else:
    #   model.add(GlobalMaxPooling1D())
    
    # Add last layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

  def train(self, epochs):
    self.model.fit(self.train_padded_docs, self.train_labels, self.epochs, verbose=1)    


  def eval(self):
    # evaluate the model, prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(self.val_dataset)
    encoded_docs = t.texts_to_sequences(self.val_dataset)
    padded_docs = pad_sequences(encoded_docs, maxlen=self.max_size, padding='post')
    loss, accuracy = self.model.evaluate(padded_docs, self.val_labels, verbose=1)
    print('Accuracy: %f' % (accuracy*100))
    return (accuracy*100)
    
  
  def initModel(self):
    # prepare tokenizer
    t = Tokenizer()
    t.fit_on_texts(self.train_dataset)
    vocab_size = len(t.word_index) + 1

    # integer encode the documents
    encoded_docs = t.texts_to_sequences(self.train_dataset)
    self.train_padded_docs = np.array(pad_sequences(encoded_docs, maxlen=self.max_size, padding='post'))

    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('resources/fasttext.es.300.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((vocab_size, 300))
    for word, i in t.word_index.items():
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector 
    
    model = self.chooseModel(vocab_size, embedding_matrix)
    # summarize the model
    print(model.summary())
    self.model = model
