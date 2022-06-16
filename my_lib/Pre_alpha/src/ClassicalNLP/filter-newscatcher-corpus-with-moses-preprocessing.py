import sys
import os
import json
import csv
import subprocess
from lambeq.ccg2discocat import DepCCGParser
depccg_parser = DepCCGParser()

with open('labelled_newscatcher_dataset.csv', newline='') as csvfile:
    news_reader = csv.DictReader(csvfile, delimiter=';', quotechar='"')
    processed_titles = set()
    tok_process_params = ["perl","./scripts/tokenizer.perl","-no-escape","-b","-l", "en"]
    tok_process = subprocess.Popen(tok_process_params, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    norm_process_params = ["perl", "./scripts/normalize-punctuation.perl","-b","-l", "en"]
    norm_process = subprocess.Popen(norm_process_params, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    tc_process_params = ["perl","./scripts/truecase.perl","-b","--model","news.2021.en.shuffled.deduped.tc.model"]
    tc_process = subprocess.Popen(tc_process_params, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
    
    for row in news_reader:
        topic = row['topic']
        title = row['title'].replace("\n"," ").replace("\t"," ").replace("\r"," ")
        norm_process.stdin.write(title.encode('utf-8'))
        norm_process.stdin.write('\n'.encode('utf-8'))
        norm_process.stdin.flush()
        norm_title = norm_process.stdout.readline().decode("utf-8").rstrip()
        tok_process.stdin.write(norm_title.encode('utf-8'))
        tok_process.stdin.write('\n'.encode('utf-8'))
        tok_process.stdin.flush()
        tok_title = tok_process.stdout.readline().decode("utf-8").rstrip()
        if len(tok_title.split())>6:
            continue
        tc_process.stdin.write(tok_title.encode('utf-8'))
        tc_process.stdin.write('\n'.encode('utf-8'))
        tc_process.stdin.flush()
        tc_title = tc_process.stdout.readline().decode("utf-8").rstrip()
        sent_type = ''
		
        if tc_title in processed_titles:
            continue
        processed_titles.add(tc_title)
		
        try:
            sent_type = 's'
            sent_type = depccg_parser.sentence2tree(tc_title).to_json()['type']
        except:
            sent_type = ''
        if sent_type == 's':
            print("{0}\t{1}".format(topic, tc_title))
			
    tc_process.kill()
    norm_process.kill()
    tok_process.kill()
