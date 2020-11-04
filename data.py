import pandas as pd
import re

TWEET = "Tweet"
ES_ODIO = "EsOdio"
COL_NAMES = [TWEET, ES_ODIO]

def toLowerCase(text):
  return text.lower()

def removeUrl(text):
  return text

def removePunctuation(text):
  return re.sub('[\W_]+', ' ',text)

def removeRepetitions(text):
  return text

def removeUsers(text):
  return re.sub("\\@.*?\\ ", "USER ", text)

def removeEmojis(text):
  return text

def removeHashtags(text):
  return re.sub(r"#", "", text)


class Data():
  def __init__(self):
    test = pd.read_csv("./resources/test.csv", names=COL_NAMES, sep="\t")
    train = pd.read_csv("./resources/train.csv", names=COL_NAMES, sep="\t")
    val = pd.read_csv("./resources/val.csv", names=COL_NAMES, sep="\t")

    for _, row in test.iterrows():
      row[TWEET] = self.preprocess(row[TWEET])

    for _, row in train.iterrows():
      row[TWEET] = self.preprocess(row[TWEET])
    
    for _, row in val.iterrows():
      row[TWEET] = self.preprocess(row[TWEET])

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