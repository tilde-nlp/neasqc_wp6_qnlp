#!/bin/env python3

import sys
import argparse
import numpy as np
sys.path.append("./models/classical/")
from NNClassifier import (loadData, NNClassifier, prepareXWords, prepareXSentence)

 
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help = "Json data file for classifier testing (with embeddings)")
    parser.add_argument("-m", "--modeldir", help = "Directory of pre-tained classifier model")
    parser.add_argument("-e", "--etype", help = "Embedding type: sentence or word")
    parser.add_argument("-g", "--gpu", help = "Number of GPU to use (from '0' to available GPUs), '-1' if use CPU (default is '-1')")
    args = parser.parse_args()

    try:
        testdata = loadData(args.infile)
        classdict = loadData(args.modeldir+'/dict.json')      
        if args.etype == "word":
            classifier = NNClassifier(model='CNN',vectorSpaceSize=300, gpu=int(args.gpu))
            classifier.load(args.modeldir)
            X=prepareXWords(testdata)
        elif args.etype == "sentence":
            classifier = NNClassifier(gpu=int(args.gpu))
            classifier.load(args.modeldir)
            X=prepareXSentence(testdata)
        else:
            print("Invalid embedding type. it must be 'word' or 'sentence'.")
            sys.exit(0)

        prediction = classifier.predict(X)
        newvals = [classdict[str(item)] for item in prediction]
        print("\n".join(newvals))
        
    except Exception as err:
        print(f"Unexpected {err=}")
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
