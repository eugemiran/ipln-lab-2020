from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

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

train_data, train_class, max_size_train = readFile("../../resources/train.csv")
validation_data, validation_class, max_size_validation = readFile("../../resources/val.csv")
test_data, test_class, max_size_test = readFile("../../resources/test.csv")
max_size = max(max_size_train, max_size_validation, max_size_test)

# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(train_data)
vocab_size = len(t.word_index) + 1

# integer encode the documents
encoded_docs = t.texts_to_sequences(train_data)
padded_docs = pad_sequences(encoded_docs, maxlen=max_size, padding='post')

# load the whole embedding into memory
embeddings_index = dict()
f = open('../../resources/fasttext.es.300.txt')
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

# define model
model = Sequential()
e = Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=max_size, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())
# fit the model
model.fit(padded_docs, train_class, epochs=50, verbose=0)

# evaluate the model, prepare tokenizer
t = Tokenizer()
t.fit_on_texts(validation_data)
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(validation_data)
padded_docs = pad_sequences(encoded_docs, maxlen=max_size, padding='post')
loss, accuracy = model.evaluate(padded_docs, validation_class, verbose=0)
print('Accuracy: %f' % (accuracy*100))