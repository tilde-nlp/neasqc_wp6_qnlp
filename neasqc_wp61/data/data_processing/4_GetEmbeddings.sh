#!/bin/bash

echo 'This script gets embedding vectors for every sample in train and dev datasets using specified embedding model.'

infile='-'
embtype='-'
embname='-'
column='0'
 
while getopts i:c:m:t flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        c) column=${OPTARG};;
        m) embname=${OPTARG};;
        t) embtype=${OPTARG};;
    esac
done

if [[ "$embtype" == "fasttext" ]] 
then
	replace="_{embtype}.json"
else
	replace="_{embname}.json"
fi

outfile=${infile//.tsv/$replace}
outfile=${infile//.txt/$replace}

if [[ "$infile" == "-" ]] || [[ "$embtype" == "-" ]] || [[ "$embname" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <input file>      input file with text examples
  -c <column>          '3' - if 3-column input file containing class, text and parse tree columns, '0' - if the whole line is a text example
  -m <embedding name>  Name of the embedding model
  -t <embedding type>  Type of the embedding model - 'fasttext', 'transformer' or 'bert'
"
	echo "$__usage"
else
	python ./data_vectorisation/Embeddings.py -i "${infile}" -o "${outfile}" -c "${column}" -m "${embname}" -t "${embtype}"
fi
