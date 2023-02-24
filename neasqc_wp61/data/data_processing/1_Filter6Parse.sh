#!/bin/bash

echo 'This script tokenizes and parses dataset using Bobcat parser. Text examples containing more that 6 tokens are skipped.'

infile='-'
delimiter=','
classfield='-'
txtfield='-'

while getopts i:d:c:t flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
        d) delimiter=${OPTARG};;
        c) classfield=${OPTARG};;
        t) txtfield=${OPTARG};;
    esac
done

replace="_alltrees.tsv"
outfile=${infile//.csv/$replace}
	
if [[ "$infile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <dataset>            Dataset file (with path)
  -d <delimiter>          Field delimiter symbol
  -c <class fiels>        Name of the class field (only if the first line in the file contains field names)
  -t <text field>         Name of the text field (only if the first line in the __usagefile contains field names)
"
	echo "$__usage"
else
	if [[ "$classfield" == "-" ]]
	then
		python ./filter-with-spacy-preprocessing.py -i "${infile}" -o "${outfile}"
	else
		python ./filter-with-spacy-preprocessing.py -i "${infile}" -o "${outfile}" -d "${delimiter}" -c "${classfield}" -t "${txtfield}"
	fi
fi
