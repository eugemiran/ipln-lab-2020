import pandas as pd
import re

TWEET = "Tweet"
ES_ODIO = "EsOdio"
COL_NAMES = [TWEET, ES_ODIO]

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

def removeEmojis(text):
  return text

def removeHashtags(text):
  #return re.sub(r'\#[\w\_]+','HASHTAG' ,text)
  return re.sub(r"#", "", text)


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
    text = removeUrl(text)
    text = removeHashtags(text)
    text = removeEmojis(text)
    text = toLowerCase(text)
    text = removeUsers(text)
    text = removeRepetitions(text)
    text = removePunctuation(text)
    return text

def main():
    x=Data()
    print(x.train.iloc[0].Tweet)
    print(x.test.iloc[0].Tweet)
    print(x.test.iloc[59].Tweet)

if __name__ == "__main__":
    main()