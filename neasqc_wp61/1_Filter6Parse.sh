#!/bin/bash

echo 'This script tokenizes and parses dataset using Bobcat parser. Text examples containing more that 20 tokens are skipped.'

infile='-'
delimiter=','
classfield='-'
txtfield='-'
gpu='-1'

while getopts i:d:c:t:g: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        d) delimiter=${OPTARG};;
        c) classfield=${OPTARG};;
        t) txtfield=${OPTARG};;
        g) gpu=${OPTARG};;
    esac
done

replace="_alltrees.tsv"
outfile=${infile/.csv/$replace}
outfile=${infile/.txt/$replace}
	
if [[ "$infile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <dataset>            Dataset file (with path)
  -d <delimiter>          Field delimiter symbol
  -c <class fiels>        Name of the class field (only if the first line in the file contains field names)
  -t <text field>         Name of the text field (only if the first line in the __usagefile contains field names)
  -g <use gpu>		   Number of GPU to use (from 0 to available GPUs), -1 if use CPU (dfault is -1)
"
	echo "$__usage"
else
	if [[ "$classfield" == "-" ]]
	then
		python ./data/data_processing/filter-with-spacy-preprocessing.py -i "${infile}" -o "${outfile}" -d "${delimiter}" -g "${gpu}"
	else
		python ./data/data_processing/filter-with-spacy-preprocessing.py -i "${infile}" -o "${outfile}" -d "${delimiter}" -c "${classfield}" -t "${txtfield}" -g "${gpu}"
	fi
fi
