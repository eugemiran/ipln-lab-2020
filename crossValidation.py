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

  for model_type in types:
    for train_index, test_index in folds.split(d.val):
      X_train = d.val.iloc[train_index]
      X_test = d.val.iloc[test_index]
      m = model.Model(model_type=model_type, train_dataset=X_train, val_dataset=X_test)
      m.train()
      m.eval()
