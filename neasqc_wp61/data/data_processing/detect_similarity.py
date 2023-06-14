#!/bin/env python3

import sys
import argparse
import csv
from sentence_transformers import SentenceTransformer
import fasttext

import itertools
from scipy.spatial import distance


def main():

    parser = argparse.ArgumentParser(description="""Detects if two sentences are similar.""")
    parser.add_argument("-i", "--infile", type=str, required=True, help="""TAB separated 3-column file.""")
    parser.add_argument("-m", "--modeldir", help = "Directory of pre-tained Fasttext vectorization model")
    args = parser.parse_args()
    
    rowlist = open(args.infile,encoding="utf-8").read().splitlines()
    
    if (args.modeldir == '-'):
    
        LabseEncoder = SentenceTransformer('sentence-transformers/LaBSE')
        eucl_score_threshold=0.8 # pair is not similar if an euclidean distance score >= eucl_score_threshold
        
        for line in rowlist:
            sentences=line.rstrip().split('\t')[1].split(' | ')
            if len(sentences)==2:
                eucl_score = distance.euclidean( LabseEncoder.encode(sentences[0]),LabseEncoder.encode(sentences[1]))
        
                if eucl_score >= eucl_score_threshold:
                    print("False")
                else:
                    print("True")
            else:
                print("")
                
    else:
        fasttext.FastText.eprint = lambda x: None
        model= fasttext.load_model(f"{args.modeldir}/fasttext_model.bin")
        eucl_score_threshold=0.5
        
        for line in rowlist:
            sentences=line.rstrip().split('\t')[1].split(' | ')
            if len(sentences)==2:
                eucl_score = distance.euclidean(model.get_sentence_vector(sentences[0].lower()),model.get_sentence_vector(sentences[1].lower()))
        
                if eucl_score >= eucl_score_threshold:
                    print("False")
                else:
                    print("True")
            else:
                print("")        
        
if __name__ == "__main__":
    sys.exit(int(main() or 0))
