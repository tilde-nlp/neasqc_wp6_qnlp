#!/bin/bash

echo 'All 3 datasets are merged. Similar sentences are paired. Unsimilar sentence pairs are created from the random sentences.'

cat ./data/datasets/chatgpt_paraphrases_alltrees.tsv | sed 's/^/T/' > ./data/datasets/similarity_filtered.tsv
 
cat ./data/datasets/sorted_tatoeba_it_alltrees.tsv | sed 's/^/T/' >> ./data/datasets/similarity_filtered.tsv

cat ./data/datasets/sorted_europarl_fr_alltrees.tsv | sed 's/^/E/' >> ./data/datasets/similarity_filtered.tsv

python ./data/data_processing/PairSentences.py ./data/datasets/similarity_filtered.tsv > ./data/datasets/paired_similarity_filtered.tsv