import sys
import os
import json
import csv
import subprocess
from lambeq import BobcatParser
from lambeq import SpacyTokeniser
tokeniser = SpacyTokeniser()
import re
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help = "Comma separated CSV data file")
    parser.add_argument("-o", "--outfile", help = "Result file")
    parser.add_argument("-d", "--delimiter", default=',', help = "Field delimiter symbol")
    parser.add_argument("-c", "--classfield", required=False, help = "Name of class field")
    parser.add_argument("-t", "--txtfield", required=False, help = "Name of text field")
    parser.add_argument("-f", "--firstsentence", action='store_true', required=False, help = "Take the first sentence if the text is longer that 6 tokens")   
    parser.add_argument("-g", "--gpu", help = "Number of GPU to use (from '0' to available GPUs), '-1' if use CPU (default is '-1')")
    args = parser.parse_args()
    
    if args.classfield != None: 
        classfield = args.classfield
        txtfield = args.txtfield
    else:
        classfield = 'f1'
        txtfield = 'f2'
 
    bobcat_parser = BobcatParser(device=int(args.gpu))
    print(args)
    print(classfield+' and '+txtfield)
    with open(args.infile, encoding="utf8", newline='') as csvfile, open(args.outfile, "w", encoding="utf8") as tsvfile:
        if args.classfield != None: 
            news_reader = csv.DictReader(csvfile, delimiter=args.delimiter, quotechar='"')
        else:
            news_reader = csv.DictReader(csvfile, delimiter=args.delimiter, fieldnames = [classfield, txtfield], quotechar='"')        
        processed_summaries = set()
        norm_process_params = ["perl", "./data/data_processing/scripts/normalize-punctuation.perl","-b","-l", "en"]
        norm_process = subprocess.Popen(norm_process_params, stdin=subprocess.PIPE, stdout=subprocess.PIPE, close_fds=True)
	
        for row in news_reader:
            score = row[classfield]
            if score == '0':
                continue
            if row[txtfield] == None:
                continue
            summary = row[txtfield].replace("\n"," ").replace("\t"," ").replace("\r"," ")
            summary = re.sub('@[^\s]+ ','',summary)
            norm_process.stdin.write(summary.encode('utf-8'))
            norm_process.stdin.write('\n'.encode('utf-8'))
            norm_process.stdin.flush()
            norm_summary = norm_process.stdout.readline().decode("utf-8").rstrip()
            tok_summary = " ".join(tokeniser.tokenise_sentences([norm_summary])[0])

            if len(tok_summary.split())>6:
                if args.firstsentence==True:
                    tok_summary = re.sub('^([^\.!?]+).+','\\1',tok_summary)
                    if len(tok_summary.split())>6:
                        continue
                else:
                    continue
            if tok_summary in processed_summaries:
                continue
            sent_type = ''
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

if __name__ == "__main__":
    sys.exit(int(main() or 0))