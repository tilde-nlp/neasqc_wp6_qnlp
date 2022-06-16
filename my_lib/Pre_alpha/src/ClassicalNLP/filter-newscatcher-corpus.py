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

with open('labelled_newscatcher_dataset.csv', newline='') as csvfile:
    news_reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
    for row in news_reader:
        topic = row['topic']
        title = row['title'].replace("\n"," ").replace("\t"," ").replace("\r"," ")
        tc_title = truecase.get_true_case("a "+title)[2::]
        doc = nlp(tc_title)
        tokens = [token.text for token in doc]
        if len(tokens)>6:
            continue
        tok_title = ' '.join(tokens)
        sent_type = ''
        try:
            sent_type = 's'
            sent_type = depccg_parser.sentence2tree(tok_title).to_json()['type']
        except:
            sent_type = ''
        if sent_type == 's':
            print("{0}\t{1}".format(topic, tok_title))
