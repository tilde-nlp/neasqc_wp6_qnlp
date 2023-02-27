#!/bin/bash

echo 'This script compares expected results with results acquired using classifier.'

efile='-'
cfile='-'
 
while getopts e:c: flag
do
    case "${flag}" in
        e) efile=${OPTARG};;
        c) cfile=${OPTARG};;
    esac
done

if [[ "$efile" == "-" ]] || [[ "$cfile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -e <expected results file> Expected results file containing class in the first column and optionaly other columns
  -c <classifier results file>   Results acquired using classifier.
"
	echo "$__usage"
elif [ "$(wc -l < $efile)" -eq "$(wc -l < $cfile)" ]
then
python -c '
import sys, csv, numpy
file1, file2 = sys.argv[1:]
with open(file1) as f1:
    expectedClass = [line.rstrip().split()[0] for line in f1]
with open(file2) as f2:
    predictedClass = [line.rstrip() for line in f2]
print("Test accuracy: " + str(sum(x == y for x, y in zip(expectedClass, predictedClass))/len(predictedClass)))
' $efile $cfile
else
	echo 'Different number of lines in files! Can not compare.'
fi
