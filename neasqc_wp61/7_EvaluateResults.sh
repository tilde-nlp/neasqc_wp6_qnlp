#!/bin/bash

echo 'This script compares expected results with results acquired using classifier.'

efile='-'
cfile='-'
ofile='-'
 
while getopts e:c:o: flag
do
    case "${flag}" in
        e) efile=${OPTARG};;
        c) cfile=${OPTARG};;
        o) ofile=${OPTARG};;
    esac
done

if [[ "$efile" == "-" ]] || [[ "$cfile" == "-" ]] || [[ "$ofile" == "-" ]]
then
__usage="
Usage: $(basename $0) [OPTIONS]

Options:
  -e <expected results file>     Expected results file containing class in the first column and optionaly other columns
  -c <classifier results file>   Results acquired using classifier.
  -o <accuracy file>             File for calculated test accuracy.
"
	echo "$__usage"
else
python3 -c '
import sys, csv, numpy
file1, file2, file3 = sys.argv[1:]
with open(file1) as f1:
    expectedClass = [line.rstrip().split()[0] for line in f1]
accuracies=[]
with open(file3, "w", encoding="utf-8") as f3:
	for i in range(30):
		ioutfile=file2.replace(".txt",f"{i}.txt")
		print(ioutfile)
		with open(ioutfile, "r", encoding="utf-8") as f2:
			predictedClass = [line.rstrip().split()[0] for line in f2]
		accuracies.append(sum(x == y for x, y in zip(expectedClass, predictedClass))/len(predictedClass))
		print(f"Test accuracy for run {i}: " + str(accuracies[-1]),file=f3)
	print(f"Average test accuracy: " + str(sum(accuracies) / len(accuracies)),file=f3)
' $efile $cfile $ofile
fi
