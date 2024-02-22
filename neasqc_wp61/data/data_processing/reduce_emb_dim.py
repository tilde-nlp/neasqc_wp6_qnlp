#!/bin/env python3

import sys
import argparse
import json
import pandas as pd
import numpy as np
from dim_reduction import (PCA, ICA, TSVD, UMAP, TSNE)
 
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data", help = "Json data file with embeddings")
    parser.add_argument("-o", "--dataout", help = "Json data file with reduced embeddings")
    parser.add_argument("-n", "--dimout", help = "Desired output dimension of the vectors")
    parser.add_argument("-a", "--algorithm", help = "Dimensionality reduction algorithm - 'PCA', 'ICA', 'TSVD', 'UMAP' or 'TSNE'")
    args = parser.parse_args()
    print(args)
       
    if args.algorithm not in ['PCA', 'ICA', 'TSVD', 'UMAP', 'TSNE']:
        print(f"{args.algorithm} not supported.")
        return 

    try:
        df = pd.read_json(args.data)
  
        vectors=df["sentence_vectorized"].tolist()
        flat_list=[x[0] for x in vectors]
        df["sentence_embedding"]=flat_list
        reduceddf=df[['class', 'sentence', 'tree', 'sentence_embedding']]
        reduceddf['class'] = reduceddf['class'].astype(str)
        if args.algorithm=='PCA':
            reducer=PCA(reduceddf,int(args.dimout))
        elif args.algorithm=='ICA':
            reducer=ICA(reduceddf,int(args.dimout))
        elif args.algorithm=='TSVD':
            reducer=TSVD(reduceddf,int(args.dimout))
        elif args.algorithm=='UMAP':
            reducer=UMAP(reduceddf,int(args.dimout))
        elif args.algorithm=='TSNE':
            reducer=TSNE(reduceddf,int(args.dimout))

        reducer.reduce_dimension()

        reducer.dataset=reducer.dataset.rename(columns={"reduced_sentence_embedding": "sentence_vectorized"})

        vectors=reducer.dataset["sentence_vectorized"].tolist()
        list_of_list=[[x] for x in vectors]
        reducer.dataset["sentence_vectorized"]=list_of_list

        dflistofdict=reducer.dataset[['class', 'sentence', 'tree', 'sentence_vectorized']].apply(lambda x: x.to_dict(), axis=1).to_list()

        with open(args.dataout,'w') as fout:
            json.dump(dflistofdict,fout, indent=2)
            
        print("Done!")
        
    except Exception as err:
        print(f"Unexpected {err=}")
        
if __name__ == "__main__":
    sys.exit(int(main() or 0))
