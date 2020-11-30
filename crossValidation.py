from sklearn.model_selection import KFold
from constants import ES_ODIO, TWEET, MODEL_TYPES
import data
import model

d = data.Data()
X = d.val[TWEET]
Y = d.val[ES_ODIO]

folds = KFold(n_splits=5)
epochs_simple=[50,100,200,300,400,500]
epochs=[25,50,75,100,125,150]
neurons=[1,4,16,32,64,128]
dropout=[0.1,0.3,0.5,0.7]
batchs=[16,32,64,128,256]
types = [MODEL_TYPES["SIMPLE"], MODEL_TYPES["LSTM1"], MODEL_TYPES["LSTM2"], MODEL_TYPES["CONVOLUTIONAL"], MODEL_TYPES["BIDIRECTIONAL"]]
results=[]
results_accuracy=[]
cont=0
cont_accuracy=0
res=open('res.txt','w')

for model_type in types:
  if model_type==MODEL_TYPES["SIMPLE"]:
    res.write('Modelo Simple: \n')
    res.write('\n')
    res.write('Ajuste epocas: \n')
    res.write('\n')
    for e in epochs_simple:
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train,  neurons=128, dropout=0.5,  val_dataset=X_test)
        m.train(e,0)
        accuracy,f1_score = m.eval()
        cont=cont+f1_score
        cont_accuracy=cont_accuracy+accuracy
      cont=cont/l
      results.append((e,cont))

      cont_accuracy=cont_accuracy/l
      results_accuracy.append((e,cont_accuracy))

      cont=0
      cont_accuracy=0
    for r in results:
      res.write('Promedio de F1-score para cada valor del hiperparametro epocas: %s \n' % (r,)) 
    res.write('\n')
    for ra in results_accuracy:
      res.write('Promedio de accuracy para cada valor del hiperparametro epocas: %s \n' % (ra,))
    results=[]  
    results_accuracy=[]
  else:  #MODELOS NO SIMPLES
    res.write('\n')
    res.write('Modelo %s \n' % (model_type))
    res.write('\n')
    res.write('Ajuste dropout: \n')
    res.write('\n')
    for drop in dropout:                                                       
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=drop, val_dataset=X_test)
        m.train(20,128)
        accuracy,f1_score = m.eval()
        cont=cont+f1_score
        cont_accuracy=cont_accuracy+accuracy
      cont=cont/l
      results.append((drop,cont))

      cont_accuracy=cont_accuracy/l
      results_accuracy.append((drop,cont_accuracy))

      cont=0
      cont_accuracy=0
    for r in results:
      res.write('Promedio de F1-score para cada valor del hiperparametro dropout: %s \n' % (r,))
    res.write('\n')
    for ra in results_accuracy:
      res.write('Promedio de accuracy para cada valor del hiperparametro dropout: %s \n' % (ra,))
    results=[]
    results_accuracy=[]
    res.write('\n')
    res.write('Ajuste de epocas \n')
    res.write('\n')
    for e in epochs:                                                              
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=0.5, val_dataset=X_test)
        m.train(e,128)
        accuracy,f1_score = m.eval()
        cont=cont+f1_score
        cont_accuracy=cont_accuracy+accuracy
      cont=cont/l
      results.append((e,cont))
      
      cont_accuracy=cont_accuracy/l
      results_accuracy.append((e,cont_accuracy))

      cont=0
      cont_accuracy=0
    for r in results:
      res.write('Promedio de F1-score para cada valor del hiperparametro epocas: %s \n' % (r,))
    res.write('\n')
    for ra in results_accuracy:
      res.write('Promedio de accuracy para cada valor del hiperparametro epocas: %s \n' % (ra,))
    results=[]
    results_accuracy=[]
    res.write('\n')
    res.write('Ajuste de neuronas \n')
    res.write('\n')
    for n in neurons:                                                         
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=n, dropout=0.5, val_dataset=X_test)
        m.train(20,128)
        accuracy,f1_score = m.eval()
        cont=cont+f1_score
        cont_accuracy=cont_accuracy+accuracy
      cont=cont/l
      results.append((n,cont))
      
      cont_accuracy=cont_accuracy/l
      results_accuracy.append((n,cont_accuracy))

      cont=0
      cont_accuracy=0
    for r in results:
      res.write('Promedio de F1-score para cada valor del hiperparametro neuronas: %s \n' % (r,))
    res.write('\n')
    for ra in results_accuracy:
      res.write('Promedio de accuracy para cada valor del hiperparametro neuronas: %s \n' % (ra,))
    results=[]
    results_accuracy=[]
    res.write('\n')
    res.write('Ajuste de batchs \n') 
    res.write('\n')
    for b in batchs:                                                       
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=0.5, val_dataset=X_test)
        m.train(20,b)
        accuracy,f1_score = m.eval()
        cont=cont+f1_score
        cont_accuracy=cont_accuracy+accuracy
      cont=cont/l
      results.append((b,cont))
      
      cont_accuracy=cont_accuracy/l
      results_accuracy.append((b,cont_accuracy))

      cont=0
      cont_accuracy=0
    for r in results:
      res.write('Promedio de F1-score para cada valor del hiperparametro batchs: %s \n' % (r,))
    res.write('\n')
    for ra in results_accuracy:
      res.write('Promedio de accuracy para cada valor del hiperparametro batchs: %s \n' % (ra,))
    results=[]  
    results_accuracy=[] 
res.close()
