#!/bin/env python3

import sys
import argparse
import os
import fasttext
from scipy.spatial import distance

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--traindata", help = "tsv file with 3-column training data, for training only 2-nd column is needed")
    parser.add_argument("-m", "--modeldir", help = "Directory where to save the trained model")
    args = parser.parse_args()
    print(args)
    
    model_params = {
    'input': 'temp/corpus.txt',
    'lr': 0.1,
    'epoch': 50,
    'dim': 300,
    'minCount': 1,
    'ws': 5,
    'bucket': 200000,
    'thread': 4,
    }
    
    try:
        rowlist = open(args.traindata,encoding="utf-8").read().splitlines()
        preprocessed_sentences = []

        for line in rowlist:
            sentences=line.rstrip().split('\t')[1].split(' | ')
            if len(sentences)==2:
                preprocessed_sentences.append(sentences[0].lower())
                preprocessed_sentences.append(sentences[1].lower())

        if not os.path.isdir('temp/'):
            os.mkdir('temp/')
        with open('temp/corpus.txt','w',encoding='utf-8',) as f:
            f.write('\n'.join(preprocessed_sentences))
            
        model = fasttext.train_unsupervised(**model_params)

        if not os.path.exists(args.modeldir):
            os.makedirs(args.modeldir)
            
        model.save_model(f"{args.modeldir}/fasttext_model.bin")
        
    except Exception as err:
        print(f"Unexpected {err=}")
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
