import sys
import os
import json
#import numpy as np
#import nltk
#from nltk.tokenize import word_tokenize
#nltk.download('punkt')
import csv
import re
from lambeq import BobcatParser
bobcat_parser = BobcatParser()

from spacy.lang.en import English
nlp = English()

with open('../datasets/Reviews.csv', encoding="utf8", newline='') as csvfile, open('../datasets/withtags_Reviews.tsv', "w", encoding="utf8") as tsvfile:
    news_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    processed_summaries = set()
    for row in news_reader:
        score = row['Score']
        summary = row['Summary'].replace("\n"," ").replace("\t"," ").replace("\r"," ")
        doc = nlp(summary)
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
            result = bobcat_parser.sentence2tree(tok_summary).to_json()
            str1=re.sub('\'(rule|text)\':\s[\"\'][^\s]+[\"\']','',str(result))
            str1=re.sub('\'(type|children)\':\s+','',str1)
            str1=re.sub('[\{\},\']','',str1)
            str1=re.sub('\s+([\[\]])','\\1',str1)
            sent_type = str1
        except:
            sent_type = ''
        print("{0}\t{1}\t{2}".format(score, tok_summary,sent_type),file=tsvfile)
