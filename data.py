import pandas as pd
import re
from constants import TWEET, ES_ODIO, COL_NAMES

def toLowerCase(text):
  return text.lower()

def removeUrl(text):
  return re.sub(r'http\S+', 'URL',text) 

def removePunctuation(text):
  return re.sub('[\W_]+', ' ',text)

def removeRepetitions(text):
  def repl(matchobj):
    c=matchobj.group(0)
    return c[0]
  return re.sub(r'(\w)\1{2,}',repl ,text)         

def removeUsers(text):
  return re.sub("\\@.*?\\ ", "USER ", text)

def removeHashtags(text):
  return re.sub(r"#", "", text)

def removeLaughter(text):
  return re.sub(r"((j|J)aja[\w]*)|((j|J)ajs[\w]*)|((j|J)eje[\w]*)|(JAJA[\w]*)", "jaja", text)

def removeLaughter2(text):
  def repl(matchobj):
    c=matchobj.group(0)
    laugth=re.sub(r"((j|J)aja[\w]*)|((j|J)ajs[\w]*)|((j|J)eje[\w]*)","",c)
    return laugth
  return re.sub(r"([\w]*(j|J)aja[\w]*)|([\w]*(j|J)ajs[\w]*)|([\w]*(j|J)eje[\w]*)",repl, text)+" "+"jaja"

def removen(text):
  return re.sub(r'\\n', "",text)

class Data():
  def __init__(self):
    test = pd.read_csv("./resources/test.csv", names=COL_NAMES, sep="\t")
    train = pd.read_csv("./resources/train.csv", names=COL_NAMES, sep="\t")
    val = pd.read_csv("./resources/val.csv", names=COL_NAMES, sep="\t")

    for i , row in test.iterrows():
      test.at[i,'Tweet'] = self.preprocess(row[TWEET])

    for i , row in train.iterrows():
      train.at[i,'Tweet'] = self.preprocess(row[TWEET])
    
    for i , row in val.iterrows():
      val.at[i,'Tweet'] = self.preprocess(row[TWEET])

    self.test = test
    self.train = train
    self.val = val

  def preprocess(self, text):
    text = removen(text)
    text = removeUrl(text)
    text = removeHashtags(text)
    text = toLowerCase(text)
    text = removeUsers(text)
    text = removeRepetitions(text)
    text = removePunctuation(text)
    text = removeLaughter(text)
    return text

def main():
  pass

if __name__ == "__main__":
  main()