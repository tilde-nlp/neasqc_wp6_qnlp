#!/bin/env python3

import sys
import argparse
import csv
from sklearn.model_selection import train_test_split

def saveinfile(data, datasettag, filename):
    with open(filename, "w", encoding="utf-8") as f:
        for sentence, sentence_type in data:
            f.write(f'{sentence_type}\t{sentence}\t{datasettag[sentence]}\n')
    return
    
def main():

    parser = argparse.ArgumentParser(description="""Split examples in train/test/dev parts with proportion 80/10/10.""")
    parser.add_argument("-i", "--infile", type=str, required=True, help="""TAB separated 3-column file.""")
    args = parser.parse_args()

    dataset: List[Tuple[str, str]] = []
    datasettag={}

    try:
        with open(args.infile, "r", encoding="utf-8") as f:
            for line in f:
                cols=line.split('\t')
                if len(cols) == 3:
                    sent = cols[1].rstrip()
                    dataset.append((sent, cols[0].rstrip()))
                    datasettag[sent] = cols[2].rstrip()
                
        classes = [item[1] for item in dataset]
        train_data, tmp_data = train_test_split(dataset, train_size=0.8, random_state=1, stratify=classes)

        classes = [item[1] for item in tmp_data]
        test_data, dev_data = train_test_split(tmp_data, train_size=0.5, random_state=1, stratify=classes)

        namenoext = args.infile.replace(".tsv","")
        saveinfile(test_data, datasettag, f"{namenoext}_test.tsv")
        saveinfile(train_data, datasettag, f"{namenoext}_train.tsv")
        saveinfile(dev_data, datasettag, f"{namenoext}_dev.tsv")
    except Exception as err:
        print(f"File: {args.infile}")
        print(f"Unexpected {err=}")
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
