#!/bin/bash

echo 'This script splits examples in train/test/dev parts with proportion 80/10/10.'

infile='-'

while getopts i: flag
do
    case "${flag}" in
        i) infile=${OPTARG};;
    esac
done

if [[ "$infile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -i <input file>              3-column file containing class, text and syntactical tree of the text
"
	echo "$__usage"
else
	python ./data/data_processing/train_test_dev_split.py -i "${infile}"
fi
