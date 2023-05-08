#!/bin/env python3

import sys
import argparse
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import defaultdict
from typing import Tuple, List

def KnownWordsInTestTrain(word_freq, dataset, train_size):
    train_data: List[Tuple[str, str]] = []
    testdev_data: List[Tuple[str, str]] = []
    word_in_testdev_freq = defaultdict(lambda: 0)
    numsentencesin_test_dev = int((1-train_size) * len(dataset))
    shuffle_index_list = shuffle(dataset, random_state=1)
    for item in shuffle_index_list:
        if len(testdev_data) == numsentencesin_test_dev:
            train_data.append(item)
        else:
            tokens = item[0].split(' ')
            word_in_curr_freq = defaultdict(lambda: 0)
            for token in tokens:
                word_in_curr_freq[token] = word_in_curr_freq[token] + 1
            goodsent = True
            for token in tokens:
                if (word_in_curr_freq[token] + word_in_testdev_freq[token]) / word_freq[token] > 0.2:
                    goodsent = False
                    break
            if goodsent == True:
                testdev_data.append(item)
                for token in tokens:
                    word_in_testdev_freq[token] = word_in_testdev_freq[token] + 1
            else:
                train_data.append(item)
            
        
    half = len(testdev_data)//2
    return train_data, testdev_data[0:half], testdev_data[half:]

def saveinfile(data, datasettag, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for sentence, sentence_type in data:
            f.write(f'{sentence_type}\t{sentence}\t{datasettag[sentence]}\n')
    return
    
def main():

    parser = argparse.ArgumentParser(description="""Split examples in train/test/dev parts with proportion.""")
    parser.add_argument("-i", "--infile", type=str, required=True, help="""TAB separated 3-column file.""")
    parser.add_argument("-r", "--randomsplit", default=False, action="store_true", help = "Split method: random stratified or words in test/dev must be also in train. If parameter omitted then words in test/dev must be also in train.")  
    args = parser.parse_args()
    dataset: List[Tuple[str, str]] = []
    datasettag={}
    word_freq = defaultdict(lambda: 0)
    
    try:
        with open(args.infile, "r", encoding="utf-8") as f:
            for line in f:
                cols=line.split('\t')
                if len(cols) == 3:
                    sent = cols[1].rstrip()
                    dataset.append((sent, cols[0].rstrip()))
                    datasettag[sent] = cols[2].rstrip()
                    if not args.randomsplit:
                        tokens=sent.split(' ')
                        for token in tokens:
                            word_freq[token] = word_freq[token] + 1
         
        if args.randomsplit:
            classes = [item[1] for item in dataset]
            train_data, tmp_data = train_test_split(dataset, train_size=0.8, random_state=1, stratify=classes)

            classes = [item[1] for item in tmp_data]
            test_data, dev_data = train_test_split(tmp_data, train_size=0.5, random_state=1, stratify=classes)
        else:
            train_data, test_data, dev_data = KnownWordsInTestTrain(word_freq, dataset, train_size=0.9)
        
        namenoext = args.infile.replace(".tsv","")
        saveinfile(test_data, datasettag, f"{namenoext}_test.tsv")
        saveinfile(train_data, datasettag, f"{namenoext}_train.tsv")
        saveinfile(dev_data, datasettag, f"{namenoext}_dev.tsv")
    except Exception as err:
        print(f"File: {args.infile}")
        print(f"Unexpected {err=}")
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
