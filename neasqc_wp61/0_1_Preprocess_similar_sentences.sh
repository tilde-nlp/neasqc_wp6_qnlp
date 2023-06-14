#!/bin/bash


echo 'Text examples are converted to the two-column format - the label (the same label for the similar sentences) and the sentence.'

cat ./data/datasets/Europarl.en-fr.fr | awk '{seen[$0]++} seen[$0]>1' | uniq > ./data/datasets/notuniq_fr.txt
python ./data/data_processing/SentencesWithSameTransl.py ./data/datasets/Europarl.en-fr.en ./data/datasets/Europarl.en-fr.fr ./data/datasets/notuniq_fr.txt | sort | uniq > ./data/datasets/sorted_Europarl_fr.txt
rm ./data/datasets/notuniq_fr.txt

cat ./data/datasets/Tatoeba.en-it.it | awk '{seen[$0]++} seen[$0]>1' | uniq > ./data/datasets/notuniq_it.txt
python ./data/data_processing/SentencesWithSameTransl.py ./data/datasets/Tatoeba.en-it.en ./data/datasets/Tatoeba.en-it.it ./data/datasets/notuniq_it.txt | sort | uniq > ./data/datasets/sorted_tatoeba_it.txt
rm ./data/datasets/notuniq_it.txt
#python ./data/data_processing/filter-with-spacy-preprocessing.py -i ../datasets/sorted_tatoeba_it.txt -o ../datasets/spacysorted_tatoeba_it.txt -g -1 -d $'\t'
#cut -d $'\t' -f3 ../datasets/spacysorted_tatoeba_it.txt| sort | uniq -c | sort -n -r > ../datasets/topsyntactic_trees.txt


python ./data/data_processing/preprocess_paraphrases.py -i ./data/datasets/chatgpt_paraphrases.csv -o ./data/datasets/chatgpt_paraphrases.txt
#python ./FinalFilter.py sorted_tatoeba_it.txt > final_tatoeba_it.txt