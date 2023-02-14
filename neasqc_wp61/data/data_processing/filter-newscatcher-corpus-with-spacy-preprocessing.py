import sys
import os
import json
import csv
import subprocess
from lambeq import BobcatParser
bobcat_parser = BobcatParser()
from lambeq import SpacyTokeniser
tokeniser = SpacyTokeniser()
import re

with open('../datasets/labelled_newscatcher_dataset.csv', encoding="utf8", newline='') as csvfile, open('../datasets/withtags_labelled_newscatcher_dataset.tsv', "w", encoding="utf8") as tsvfile:
    news_reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
    processed_titles = set()
    norm_process_params = ["perl", "./scripts/normalize-punctuation.perl","-b","-l", "en"]
    norm_process = subprocess.Popen(norm_process_params, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    
    for row in news_reader:
        topic = row['topic']
        title = row['title'].replace("\n"," ").replace("\t"," ").replace("\r"," ")
        norm_process.stdin.write(title.encode('utf-8'))
        norm_process.stdin.write('\n'.encode('utf-8'))
        norm_process.stdin.flush()
        norm_title = norm_process.stdout.readline().decode("utf-8").rstrip()
        tok_title = " ".join(tokeniser.tokenise_sentences([norm_title])[0])
        if len(tok_title.split())>6:
            continue
        sent_type = ''
        if tok_title in processed_titles:
            continue
        processed_titles.add(tok_title)
        try:
            sent_type = 's'
            result = bobcat_parser.sentence2tree(tok_title).to_json()
            str1=re.sub('\'(rule|text)\':\s[\"\'][^\s]+[\"\']','',str(result))
            str1=re.sub('\'(type|children)\':\s+','',str1)
            str1=re.sub('[\{\},\']','',str1)
            str1=re.sub('\s+([\[\]])','\\1',str1)
            sent_type = str1
        except:
            sent_type = ''
        print("{0}\t{1}\t{2}".format(topic, tok_title,sent_type),file=tsvfile)
			
    norm_process.kill()
