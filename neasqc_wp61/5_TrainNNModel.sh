#!/bin/bash

echo 'This script trains neural network classifier model.'

tfile='-'
dfile='-'
field='class'
etype='-'
modeldir='-'
gpu='-1'
 
while getopts t:d:f:e:m:g: flag
do
    case "${flag}" in
        t) tfile=${OPTARG};;
        d) dfile=${OPTARG};;
        f) field=${OPTARG};;
        e) etype=${OPTARG};;
        m) modeldir=${OPTARG};;
        g) gpu=${OPTARG};;
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
  -g <use gpu>		   Number of GPU to use (from 0 to available GPUs), -1 if use CPU (dfault is -1)
"
	echo "$__usage"
else
	python ./data/data_processing/train_classifier.py -t "${tfile}" -d "${dfile}" -f "${field}" -e "${etype}" -m "${modeldir}"  -g "${gpu}"
fi
