import sys
import os
import json
#import numpy as np
#import nltk
#from nltk.tokenize import word_tokenize
#nltk.download('punkt')
import csv
import truecase
from lambeq.ccg2discocat import DepCCGParser
from spacy.lang.en import English
nlp = English()
depccg_parser = DepCCGParser()

with open('Reviews.csv', newline='') as csvfile:
    news_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    processed_summaries = set()
    for row in news_reader:
        score = row['Score']
        summary = row['Summary'].replace("\n"," ").replace("\t"," ").replace("\r"," ")
        tc_summary = truecase.get_true_case("a "+summary)[2::]
        doc = nlp(tc_summary)
        tokens = [token.text for token in doc]
        if len(tokens)>6:
            continue
        tok_summary = ' '.join(tokens)
        sent_type = ''
        if tok_summary in processed_summaries:
            continue
        processed_summaries.add(tok_summary)
        try:
            sent_type = 's'
            sent_type = depccg_parser.sentence2tree(tok_summary).to_json()['type']
        except:
            sent_type = ''
        if sent_type == 's':
            print("{0}\t{1}".format(score, tok_summary))
