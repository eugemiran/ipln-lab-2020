from numpy import array
from numpy import asarray
from numpy import zeros
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import SpatialDropout1D
from keras.layers import Conv1D
from keras.layers import GlobalMaxPooling1D
from keras.layers import Bidirectional
from keras.layers import Embedding
import numpy as np

import fasttext
import fasttext.util

def readFile(path):
	with open(path, "r") as myfile:
		max_size = 0
		docs = []
		labels = []
		for line in myfile.readlines():
			line_size = len(line)
			if (line_size > max_size):
				max_size = line_size
			docs.append(line[:line_size-2])
			labels.append(int(line[line_size-2]))
		return docs, labels, max_size

#fasttext.util.download_model('es', if_exists='ignore')  # English
#
# ft = fasttext.load_model('cc.es.300.bin')


train_data, train_class, max_size_train = readFile("resources/train.csv")
validation_data, validation_class, max_size_validation = readFile("resources/val.csv")
test_data, test_class, max_size_test = readFile("resources/test.csv")
max_size = max(max_size_train, max_size_validation, max_size_test)

model_optimizer = 'adam'
model_loss_function = 'binary_crossentropy'
model_activation_function = 'sigmoid'

#model_optimizer = 'rmsprop'
#model_loss_function = 'categorical_crossentropy'
#model_activation_function = 'softmax'
#model_activation_function = 'relu'

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(train_data)
vocab_size = len(t.word_index) + 1

# integer encode the documents
encoded_docs = t.texts_to_sequences(train_data)
padded_docs = pad_sequences(encoded_docs, maxlen=max_size, padding='post')

# load the whole embedding into memory
embeddings_index = dict()
f = open('resources/fasttext.es.300.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 300))
for word, i in t.word_index.items():
	embedding_vector = embeddings_index.get(word)
	if embedding_vector is not None:
		embedding_matrix[i] = embedding_vector 
#	else:
#		print("add " + str(word) + " to embeddings")
#		near_neighbor = ft.get_nearest_neighbors(str(word), 1)[0]
#		embedding_matrix[i] = ft.get_word_vector(str(near_neighbor[1]))

# define model
model = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_size, trainable=False)
model.add(e)

#Simple model
#model.add(Flatten())
#model.add(Dense(1, activation='sigmoid'))

#LSTM model
model.add(LSTM(128, dropout=0.5))
model.add(Dense(1, activation='sigmoid'))

#LSTM model version 2
#model.add(SpatialDropout1D(0.25))
#model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
#model.add(Dropout(0.2))
#model.add(Dense(1, activation='sigmoid'))

#Bidirectional model
#model.add(Bidirectional(LSTM(128,dropout=0.3)))
#model.add(Dense(1, activation='sigmoid'))

#Convolutional model
#model.add(Conv1D(128, 5, activation='relu'))
#model.add(GlobalMaxPooling1D())
#model.add(Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())

# fit for simple model

######## Lines that fixed the error
padded_docs = np.array(padded_docs)
train_class = np.array(train_class)
###############################

model.fit(padded_docs, train_class, epochs=50, verbose=1)

# fit for lstm model
model.fit(padded_docs, train_class, epochs=5, batch_size=128, verbose=1)

# evaluate the model, prepare tokenizer
t = Tokenizer()
t.fit_on_texts(validation_data)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(validation_data)
padded_docs = pad_sequences(encoded_docs, maxlen=max_size, padding='post')
######## Lines that fixed the error
validation_class = np.array(validation_class)
#############################################
loss, accuracy = model.evaluate(padded_docs, validation_class, verbose=1)
print('Accuracy: %f' % (accuracy*100))