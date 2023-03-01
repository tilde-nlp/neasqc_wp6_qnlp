#!/bin/bash

echo 'This script gets embedding vectors for every sample in train and dev datasets using specified embedding model.'

infile='-'
embtype='-'
embname='-'
column='0'
gpu='-1'
 
while getopts i:c:m:t:g: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        c) column=${OPTARG};;
        m) embname=${OPTARG};;
        t) embtype=${OPTARG};;
        g) gpu=${OPTARG};;
    esac
done

if [[ "$embtype" == "fasttext" ]] 
then
	replace="_${embtype}.json"
else
	replace="_${embname}.json"
fi

outfile=${infile//.tsv/$replace}

if [[ "$infile" == "-" ]] || [[ "$embtype" == "-" ]] || [[ "$embname" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <input file>      input file with text examples
  -c <column>          '3' - if 3-column input file containing class, text and parse tree columns, '0' - if the whole line is a text example
  -m <embedding name>  Name of the embedding model
  -t <embedding type>  Type of the embedding model - 'fasttext', 'transformer' or 'bert'
  -g <use gpu>		   Number of GPU to use (from 0 to available GPUs), -1 if use CPU (dfault is -1)
"
	echo "$__usage"
else
	python ./data/data_processing/data_vectorisation/Embeddings.py -i "${infile}" -o "${outfile}" -c "${column}" -m "${embname}" -t "${embtype}" -g "${gpu}"
fi
