#!/bin/bash

echo 'This script trains FastText embedding model.'

tfile='-'
modeldir='-'
 
while getopts t:m: flag
do
    case "${flag}" in
        t) tfile=${OPTARG};;
        m) modeldir=${OPTARG};;
    esac
done

if [[ "$tfile" == "-" ]] || [[ "$modeldir" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -t <train data file> tsv file acquired using script 3_SplitTrainTestDev.sh
  -m <model directory> Directory where to save trained model
"
	echo "$__usage"
else
	python ./data/data_processing/train_embeddings.py -t "${tfile}" -m "${modeldir}"
fi
