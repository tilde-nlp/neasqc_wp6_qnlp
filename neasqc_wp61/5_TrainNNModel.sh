#!/bin/bash

echo 'This script trains neural network classifier model.'

tfile='-'
dfile='-'
field='class'
etype='-'
modeldir='-'
 
while getopts t:d:f:e:m: flag
do
    case "${flag}" in
        t) tfile=${OPTARG};;
        d) dfile=${OPTARG};;
        f) field=${OPTARG};;
        e) etype=${OPTARG};;
        m) modeldir=${OPTARG};;
    esac
done

if [[ "$tfile" == "-" ]] || [[ "$dfile" == "-" ]] || [[ "$etype" == "-" ]] || [[ "$modeldir" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -t <train data file> Json data file for classifier training (with embeddings)
  -d <dev data file>   Json data file for classifier validation (with embeddings)
  -f <field>           Classify by field
  -e <embedding type>  Embedding type: 'sentence' or 'word'
  -m <model directory> Directory where to save trained model
"
	echo "$__usage"
else
	python train_classifier.py -t "${tfile}" -d "${dfile}" -f "${field}" -e "${etype}" -m "${modeldir}"
fi
