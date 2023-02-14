#!/bin/env python3

import sys
import argparse
import csv

def main():

    parser = argparse.ArgumentParser(description="""Filter by syntax tags in the third column.""")
    parser.add_argument("-i", "--infile", type=str, required=True, help="""TAB separated 3-column file to filter.""")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="""Filtered 2-column file.""")
    parser.add_argument("-f", "--filterfile", type=str, required=True, help="""File containing list of filtering conditions""")
    args = parser.parse_args()

    filterlist = open(args.filterfile).read().splitlines()
    
    with open(args.infile, "r", encoding="utf8") as ifile, open(args.outfile, "w", encoding="utf8") as ofile:
        tsv_reader = csv.DictReader(ifile, fieldnames=['Class','Txt','Tag'], delimiter="\t", quotechar='"')
        for item in tsv_reader:
            if item['Tag'] in filterlist:
                print("{0}\t{1}".format(item['Class'],item['Txt']),file=ofile)

if __name__ == "__main__":
    sys.exit(int(main() or 0))
