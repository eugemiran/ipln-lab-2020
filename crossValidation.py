from sklearn.model_selection import KFold
from constants import ES_ODIO, TWEET, MODEL_TYPES
import data
import model

def crossValidation():
  d = data.Data()
  X = d.val[TWEET]
  Y = d.val[ES_ODIO]

  folds = KFold(n_splits=5)
  types = [MODEL_TYPES["SIMPLE"], MODEL_TYPES["LSTM1"], MODEL_TYPES["LSTM2"], MODEL_TYPES["CONVOLUTIONAL"], MODEL_TYPES["BIDIRECTIONAL"]]
  epochs = [5, 10, 30]
  neural_sizes = []
  # other params go here

  for epoch in epochs:
    for train_index, test_index in folds.split(d.val):
      X_train = d.val.iloc[train_index]
      X_test = d.val.iloc[test_index]
      m = model.Model(model_type="CONVOLUTIONAL", train_dataset=X_train, val_dataset=X_test)
      m.train(epoch)
      m.eval()


  for epoch in epochs:
    for train_index, test_index in folds.split(d.val):
      X_train = d.val.iloc[train_index]
      X_test = d.val.iloc[test_index]
      m = model.Model(model_type="LSTM1", train_dataset=X_train, val_dataset=X_test)
      m.train(epoch)
      m.eval()
  

  for epoch in epochs:
    for train_index, test_index in folds.split(d.val):
      X_train = d.val.iloc[train_index]
      X_test = d.val.iloc[test_index]
      m = model.Model(model_type="SIMPLE", train_dataset=X_train, val_dataset=X_test)
      m.train(epoch)
      m.eval()