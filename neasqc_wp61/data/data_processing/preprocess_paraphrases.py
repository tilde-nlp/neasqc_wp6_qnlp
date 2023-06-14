#!/bin/env python3

import sys
import argparse
import pandas as pd

def main():

    parser = argparse.ArgumentParser(description="""Change file structure by assigning same ID to similar sentences.""")
    parser.add_argument("-i", "--infile", type=str, required=True, help="""comma separated 4-column file to filter.""")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="""2-column file.""")
    args = parser.parse_args()
  
    df = pd.read_csv(args.infile, usecols = ['text','paraphrases'],converters={"paraphrases": lambda x: x.strip("[]").replace('"','\'').split(", '")})
    df['text'] = df['text'].apply(lambda x: x.replace("\n",""))
    df = df[df['text'].notna()]
    txtcol = df.loc[:,['text']]
    txtcol=txtcol.rename(columns={"text": "paraphrases"})
    df=df.drop(columns=['text'])
    df = df.explode('paraphrases')
    df['paraphrases'] = df['paraphrases'].apply(lambda x: x.strip("'\""))
    result = pd.concat([df,txtcol])
    result = result.sort_index(ascending=True)
    result.to_csv(args.outfile,header=False,sep='\t')

if __name__ == "__main__":
    sys.exit(int(main() or 0))
