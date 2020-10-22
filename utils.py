#import fasttext
#import fasttext.util
#import io
#from gensim.models.wrappers import FastText
import numpy as np
import pandas as pd
import csv

def test():
    #print('waiting for load model')
    #ft = fasttext.load_model('cc.es.300.bin')
    #print('load model success')
    #print(ft.get_dimension())
    #data = load_vectors('resources/fasttext.es.300.txt')
    #print(data)
    #model = fasttext.train_supervised('data.train.txt')
    # Skipgram model
    #model = fasttext.train_unsupervised('resources/test.csv', model='skipgram')
    #print(model.words)
    #model = fasttext.train_supervised('resources/train.csv')
    #print(model.words)
    #print(model.label)
    #model = FastText.load_fasttext_format('resources/fasttext.es.300.txt')
    readLines("resources/train.csv")

def loadVectors(path):
    fin = io.open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data


def readCSV(path, separator):
    df = pd.read_csv(path, sep=separator, engine='python')
    return df

def readLines(path):
    with open(path, "r") as myfile:
        for line in myfile.readlines():
            clasification = line[len(line)-2]