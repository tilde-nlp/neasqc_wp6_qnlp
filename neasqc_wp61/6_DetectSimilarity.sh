#!/bin/bash

echo 'This script that detects if two sentences are similar.'

infile='-'
outfile='-'
modeldir='-'
 
while getopts i:o:m: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        o) outfile=${OPTARG};;
        m) modeldir=${OPTARG};;
    esac
done

if [[ "$infile" == "-" ]] || [[ "$outfile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <input file> 	   tsv file (acquired using script 3_SplitTrainTestDev.sh)
  -o <output file>     Result file with predicted True/False values
  -m <model directory> Directory of pre-tained Fasttext vectorization model
"
	echo "$__usage"
else
	python ./data/data_processing/detect_similarity.py -i "${infile}" -m "${modeldir}" > $outfile
fi
