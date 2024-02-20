#!/bin/env python3

import sys
import argparse
import csv
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from collections import defaultdict
from typing import Tuple, List
import pandas as pd


def main():

    parser = argparse.ArgumentParser(description="""Split examples in train/test/dev parts with proportion.""")
    parser.add_argument("-i", "--infile", type=str, required=True, help="""TAB separated 3-column file.""")
    args = parser.parse_args()
    
    if True:
        dataset = pd.read_csv(args.infile, sep='\t',header=None, names=['Class','Txt','Tag'], dtype=str)

        minval=dataset.groupby('Class').size().min()
        numintrain=(minval*8)//10
        numintest=minval//10
        print(minval)
        print(numintrain)
        print(numintest)
        traindataset=dataset.groupby('Class').nth[0:numintrain]
        lastidx=numintrain+numintest
        testdataset=dataset.groupby('Class').nth[numintrain:lastidx]
        devdataset=dataset.groupby('Class').nth[lastidx:lastidx+numintest]
        namenoext = args.infile.replace(".tsv","")
        
        traindataset.to_csv(f"{namenoext}_train.tsv", sep='\t', mode='w',encoding='utf-8',index=False,header=False,columns=['Class','Txt','Tag'])
        testdataset.to_csv(f"{namenoext}_test.tsv", sep='\t', mode='w',encoding='utf-8',index=False,header=False,columns=['Class','Txt','Tag'])
        devdataset.to_csv(f"{namenoext}_dev.tsv", sep='\t', mode='w',encoding='utf-8',index=False,header=False,columns=['Class','Txt','Tag'])
        
    #except Exception as err:
    #    print(f"File: {args.infile}")
    #    print(f"Unexpected {err=}")
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
