#!/bin/bash

echo 'This script classifies examples using pre-tained neural network classifier model.'

infile='-'
etype='-'
modeldir='-'
outfile='-'
gpu='-1'
 
while getopts i:o:e:m:g: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        o) outfile=${OPTARG};;
        e) etype=${OPTARG};;
        m) modeldir=${OPTARG};;
        g) gpu=${OPTARG};;
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
  -g <use gpu>		   Number of GPU to use (from 0 to available GPUs), -1 if use CPU (dfault is -1)
"
	echo "$__usage"
else
	python ./data/data_processing/use_classifier.py -i "${infile}" -e "${etype}" -m "${modeldir}" -g "${gpu}" > $outfile
fi
