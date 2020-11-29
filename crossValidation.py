from sklearn.model_selection import KFold
from constants import ES_ODIO, TWEET, MODEL_TYPES
import data
import model

d = data.Data()
X = d.val[TWEET]
Y = d.val[ES_ODIO]

folds = KFold(n_splits=5)
epochs_simple=[100,200,300,400,500]
epochs=[50,100,150,200,250]
neurons=[1,4,16,32,64,128]
dropout=[0.1,0.3,0.5]
batchs=[64,128,256,512]
types = [MODEL_TYPES["SIMPLE"], MODEL_TYPES["LSTM1"], MODEL_TYPES["LSTM2"], MODEL_TYPES["CONVOLUTIONAL"], MODEL_TYPES["BIDIRECTIONAL"]]
results=[]
cont=0

for model_type in types:
  print('Modelo: %s' % (model_type))
  if model_type==MODEL_TYPES["SIMPLE"]:
    print('Ajuste de epocas.')    
    for e in epochs_simple:
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train,  neurons=128, dropout=0.5,  val_dataset=X_test)
        m.train(e,0)
        acurrency = m.eval()
        cont=cont+acurrency
      cont=cont/l
      print('Promedio de acurrency para esta epoca: %f' % (cont))
      results.append(cont)
      cont=0
    for r in results:
      print('Promedio de cada uno: %f' % (r))
    results=[]
  else:  #MODELOS NO SIMPLES
    for drop in dropout:                                                       
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=drop, val_dataset=X_test)
        m.train(2,128)
        acurrency = m.eval()
        print('Acurrency: %f' % (acurrency))
        cont=cont+acurrency
        print('Sumatorias de acurrency: %f' % (cont))
      cont=cont/5
      print('Promedio de dropout: %f' % (cont))
      #print('COnt: %f' % (cont))
      results.append(cont)
      cont=0
    for r in results:
      print('Promedio de cada uno: %f' % (r))
    results=[]
    print('Ajuste de epocas.')    
    for e in epochs:                                                              
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=0.5, val_dataset=X_test)
        m.train(e,128)
        acurrency = m.eval()
        print('Acurrency: %f' % (acurrency))
        cont=cont+acurrency
        print('Sumatorias de acurrency: %f' % (cont))
      cont=cont/l
      print('Promedio para esta epoca: %f' % (cont))
      #print('COnt: %f' % (cont))
      results.append(cont)
      cont=0
    for r in results:
      print('Promedio de cada uno: %f' % (r))
    results=[]
    print('Ajuste de neuronas.')    
    for n in neurons:                                                         
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=n, dropout=0.5, val_dataset=X_test)
        m.train(25,128)
        acurrency = m.eval()
        print('Acurrency: %f' % (acurrency))
        cont=cont+acurrency
        print('Sumatorias de acurrency: %f' % (cont))
      cont=cont/l
      print('Promedio de neuronas : %f' % (cont))
      #print('COnt: %f' % (cont))
      results.append(cont)
      cont=0
    for r in results:
      print('Promedio de cada uno: %f' % (r))
    results=[]
    print('Ajuste de batchs.')    
    for b in batchs:                                                       
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=0.5, val_dataset=X_test)
        m.train(25,b)
        acurrency = m.eval()
        print('Acurrency: %f' % (acurrency))
        cont=cont+acurrency
        print('Sumatorias de acurrency: %f' % (cont))
      cont=cont/l
      print('Promedio de batches : %f' % (cont))
      results.append(cont)
      cont=0
    for r in results:
      print('Promedio de cada uno: %f' % (r))
    results=[]
    print('Ajuste de dropout.')    
    
