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
res=open('res.txt','w')

for model_type in types:
  if model_type==MODEL_TYPES["SIMPLE"]:
    res.write('Modelo Simple \n')
    for e in epochs_simple:
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train,  neurons=128, dropout=0.5,  val_dataset=X_test)
        m.train(e,0)
        acurrency,f1_score = m.eval()
        cont=cont+f1_score
      cont=cont/l
      res.write('Promedio de f1 para esta epoca: %f para el modelo %s \n' % (cont,model_type))
      results.append(cont)
      cont=0
    for r in results:
      res.write('Promedio de cada uno: %f para el modelo %s \n' % (r,model_type))
    results=[]  
  else:  #MODELOS NO SIMPLES
    res.close()
    res=open('res.txt','w')
    res.write('Modelos No Simples \n')
    for drop in dropout:                                                       
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=drop, val_dataset=X_test)
        m.train(2,128)
        acurrency,f1_score = m.eval()
        res.write('f1: %f model %s' % (f1_score,model_type))
        cont=cont+f1_score
        res.write('Sumatorias de f1: %f model %s \n' % (cont,model_type))
      cont=cont/5
      res.write('Promedio de dropout: %f model %s \n' % (cont,model_type))
      results.append(cont)
      cont=0
    for r in results:
      res.write('Promedio de cada uno: %f  model %s \n' % (r,model_type))
    results=[]
    res.write('Ajuste de epocas \n')
    res.close()
    res=open('res.txt','w')
    for e in epochs:                                                              
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=0.5, val_dataset=X_test)
        m.train(e,128)
        acurrency,f1_score = m.eval()
        res.write('f1: %f model %s \n' % (f1_score,model_type))
        cont=cont+f1_score
        res.write('Sumatorias de f1: %f model %s \n' % (cont,model_type))
      cont=cont/l
      res.write('Promedio para esta epoca: %f model %s \n' % (cont,model_type))
      results.append(cont)
      cont=0
    for r in results:
      res.write('Promedio de cada uno: %f model  %s \n' % (r,model_type))
    results=[]
    res.write('Ajuste de neuronas \n')
    res.close()
    res=open('res.txt','w')    
    for n in neurons:                                                         
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=n, dropout=0.5, val_dataset=X_test)
        m.train(25,128)
        acurrency,f1_score = m.eval()
        res.write('f1: %f model  %s \n' % (f1_score,model_type))
        cont=cont+f1_score
        res.write('Sumatorias de f1: %f model %s  \n' % (cont,model_type))
      cont=cont/l
      res.write('Promedio de neuronas : %f  \n' % (cont))
      results.append(cont)
      cont=0
    for r in results:
      res.write('Promedio de cada uno: %f model %s \n' % (r,model_type))
    results=[]
    res.write('Ajuste de batchs \n') 
    res.close()
    res=open('res.txt','w')  
    for b in batchs:                                                       
      l=len(list(folds.split(d.val)))
      for train_index, test_index in folds.split(d.val):
        X_train = d.val.iloc[train_index]
        X_test = d.val.iloc[test_index]
        m = model.Model(model_type=model_type, train_dataset=X_train, neurons=128, dropout=0.5, val_dataset=X_test)
        m.train(25,b)
        acurrency,f1_score = m.eval()
        res.write('f1: %f model %s \n' % (f1_score,model_type))
        cont=cont+f1_score
        res.write('Sumatorias de f1: %f model %s \n' % (cont,model_type))
      cont=cont/l
      res.write('Promedio de batches : %f model %s \n' % (cont,model_type))
      results.append(cont)
      cont=0
    for r in results:
      res.write('Promedio de cada uno: %f model %s \n' % (r,model_type))
    results=[]
    res.write('Ajuste de dropout \n')    
res.close()
