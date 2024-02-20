#!/bin/bash

echo 'This script filters lines that have a certain syntactical structure.'

infile='-'
filterfile='-'
classes='-'

while getopts i:f:c: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        f) filterfile=${OPTARG};;
        c) classes=${OPTARG};;
    esac
done

replace="_filtered.tsv"
replacetmp="_tmp.tsv"
outfiletmp=${infile//_alltrees.tsv/$replacetmp}
outfile=${infile//_alltrees.tsv/$replace}
	
if [[ "$infile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <input file>               3-column file containing class, text and syntactical tree of the text
  -f <syntactical trees file>   File containing syntactical trees for filtering
  -c <classes to ignore>		Optionally classes to ignore in dataset separated by ',' 
"
	echo "$__usage"
elif [[ "$filterfile" == "-" ]]
then
	python ./data/data_processing/filter-by_syntax_sentences.py -i "${infile}" -o "${outfiletmp}"
	if [[ "$classes" == "-" ]]
	then
		python ./data/data_processing/balance_classes.py -i "${outfiletmp}" -o "${outfile}"
	else
		python ./data/data_processing/balance_classes.py -i "${outfiletmp}" -o "${outfile}" -c 	"${classes}"
	fi
else
	python ./data/data_processing/filter-by_syntax.py -i "${infile}" -o "${outfile}" -f "${filterfile}"
	if [[ "$classes" == "-" ]]
	then
		python ./data/data_processing/balance_classes.py -i "${outfiletmp}" -o "${outfile}"
	else
		python ./data/data_processing/balance_classes.py -i "${outfiletmp}" -o "${outfile}" -c 	"${classes}"
	fi
	wc -l $outfile | awk '{ print $1, " lines added to the filtered file."}'
fi
