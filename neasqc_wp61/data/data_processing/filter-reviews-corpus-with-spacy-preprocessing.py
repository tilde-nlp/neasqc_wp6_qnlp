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

with open('../datasets/Reviews.csv', encoding="utf8", newline='') as csvfile, open('../datasets/withtags_Reviews.tsv', "w", encoding="utf8") as tsvfile:
    news_reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    processed_summaries = set()
    norm_process_params = ["perl", "./scripts/normalize-punctuation.perl","-b","-l", "en"]
    norm_process = subprocess.Popen(norm_process_params, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
	
    for row in news_reader:
        score = row['Score']
        summary = row['Summary'].replace("\n"," ").replace("\t"," ").replace("\r"," ")
        norm_process.stdin.write(summary.encode('utf-8'))
        norm_process.stdin.write('\n'.encode('utf-8'))
        norm_process.stdin.flush()
        norm_summary = norm_process.stdout.readline().decode("utf-8").rstrip()
        tok_summary = " ".join(tokeniser.tokenise_sentences([norm_summary])[0])
        if len(tok_summary.split())>6:
            continue
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
			
    norm_process.kill()
