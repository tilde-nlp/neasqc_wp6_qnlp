#!/bin/bash

echo 'This script filters lines that have a certain syntactical structure.'

infile='-'
filterfile='-'

while getopts i:f: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        f) filterfile=${OPTARG};;
    esac
done

replace="_filtered.tsv"
outfile=${infile//_alltrees.tsv/$replace}
	
if [[ "$infile" == "-" ]] || [[ "$filterfile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <input file>              3-column file containing class, text and syntactical tree of the text
  -f <syntactical trees file>  File containing syntactical trees for filtering
"
	echo "$__usage"
else
	python ./data/data_processing/filter-by_syntax.py -i "${infile}" -o "${outfile}" -f "${filterfile}"
	wc -l $outfile | awk '{ print $1, " lines added to the filtered file."}'
fi
