#!/bin/bash

echo 'This script classifies examples using pre-tained neural network classifier model. Results with predicted classes are printed to stdout.'

infile='-'
etype='-'
modeldir='-'
outfile='-'
 
while getopts i:o:e:m: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        o) outfile=${OPTARG};;
        e) etype=${OPTARG};;
        m) modeldir=${OPTARG};;
    esac
done

if [[ "$infile" == "-" ]] || [[ "$outfile" == "-" ]] || [[ "$etype" == "-" ]] || [[ "$modeldir" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <input file> 	   Json data file for classifier testing (with embeddings acquired using script 4_GetEmbeddings.sh)
  -o <output file>     Result file with predicted classes
  -e <embedding type>  Embedding type: 'sentence' or 'word'
  -m <model directory> Directory of pre-tained classifier model
"
	echo "$__usage"
else
	python use_classifier.py -i "${infile}" -e "${etype}" -m "${modeldir}" > $outfile
fi
