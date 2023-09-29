#!/bin/env python3

import sys
import argparse
import json
import time
sys.path.append("./models/classical/")
from NNClassifier import (loadData, NNClassifier, prepareXYWords, prepareXYSentence, prepareClassValueDict, prepareXYWordsNoEmbedd, prepareTokenDict)
 
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--traindata", help = "Json data file for classifier training (with embeddings)")
    parser.add_argument("-d", "--devdata", help = "Json data file for classifier validation (with embeddings)")
    parser.add_argument("-f", "--field", help = "Classify by field")
    parser.add_argument("-e", "--etype", help = "Embedding type: 'sentence', 'word' or '-'")
    parser.add_argument("-m", "--modeldir", help = "Directory where to save the trained model")
    parser.add_argument("-g", "--gpu", help = "Number of GPU to use (from '0' to available GPUs), '-1' if use CPU (default is '-1')")
    args = parser.parse_args()
    print(args)
    try:
        traindata = loadData(args.traindata)
        devdata = loadData(args.devdata)
        idxdict = prepareClassValueDict(traindata, args.field)
        batch_size=32
        start_time= time.time()
        if len(traindata) > 10000:
            batch_size=4096
        if args.etype == "word":
            maxLen = 6
            trainX, trainY = prepareXYWords(traindata, maxLen, args.field, idxdict)
            vecsize = len(trainX[0][0])
            print(F"Vec size: {vecsize}")
            classifier = NNClassifier(model='CNN',vectorSpaceSize=vecsize, gpu=int(args.gpu), batch_size=batch_size)
            devX, devY = prepareXYWords(devdata, maxLen, args.field)
        elif args.etype == "sentence":
            classifier = NNClassifier(gpu=int(args.gpu),batch_size=batch_size)
            trainX, trainY = prepareXYSentence(traindata, args.field, idxdict)
            devX, devY = prepareXYSentence(devdata, args.field)
        elif args.etype == "-":
            maxLen = 6
            tokdict = prepareTokenDict(traindata)
            trainX, trainY = prepareXYWordsNoEmbedd(traindata, tokdict, maxLen, args.field, idxdict)
            vecsize = len(trainX[0])
            print(F"Vec size: {vecsize}")
            classifier = NNClassifier(model='LSTM',vectorSpaceSize=vecsize,gpu=int(args.gpu),batch_size=batch_size)
            devX, devY = prepareXYWordsNoEmbedd(devdata, tokdict, maxLen, args.field)
        else:
            print("Invalid embedding type. it must be 'word' or 'sentence'.")
            sys.exit(0)
	
        history = classifier.train(trainX, trainY, devX, devY)
        end_time= time.time()
                        
        nn_train_acc = history.history["accuracy"][-1]
        print(f"Model train accuracy: {nn_train_acc}")
        print(f"Saving model to {args.modeldir}")
        classifier.save(args.modeldir)
        inv_map = {v: k for k, v in idxdict.items()}
        with open(args.modeldir+'/dict.json', 'w') as map_file:
            map_file.write(json.dumps(inv_map))
        if args.etype == "-":
            with open(args.modeldir+'/tokdict.json', 'w') as tok_file:
               tok_file.write(json.dumps(tokdict))
        
        ds = {
            "input_args": {"runs": 1,
            "iterations": len(history.history["val_accuracy"])},
            "best_val_acc": max(history.history["val_accuracy"]),
            "best_run": 0,
            "time": [end_time-start_time],
            "train_acc": [history.history["accuracy"]],
            "train_loss": [history.history["loss"]],
            "val_acc": [history.history["val_accuracy"]],
            "val_loss": [history.history["val_loss"]],
            }

        with open(args.modeldir+'/results.json', "w", encoding="utf-8") as f:
            json.dump(ds, f, ensure_ascii=False, indent=2)
            
        classifier.plot_model(args.modeldir+'/model_plot.png')
            
    except Exception as err:
        print(f"Unexpected {err=}")
    
if __name__ == "__main__":
    sys.exit(int(main() or 0))
