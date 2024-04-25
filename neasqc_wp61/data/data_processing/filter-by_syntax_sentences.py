#!/bin/env python3

import sys
import argparse
import csv
import re

def main():

    parser = argparse.ArgumentParser(description="""Filter by syntax tags in the third column.""")
    parser.add_argument("-i", "--infile", type=str, required=True, help="""TAB separated 3-column file to filter.""")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="""Filtered 3-column file.""")
    args = parser.parse_args()

    
    with open(args.infile, "r", encoding="utf8") as ifile, open(args.outfile, "w", encoding="utf8") as ofile:
        tsv_reader = csv.DictReader(ifile, fieldnames=['Class','Txt','Tag'], delimiter="\t", quotechar='"')
        for item in tsv_reader:
            if re.search("^s\[.+\]$",item['Tag']):
                print("{0}\t{1}\t{2}".format(item['Class'],item['Txt'],item['Tag']),file=ofile)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
