import sys

sp_set = set(map(str.strip, open(sys.argv[3])))
n=0

es_lines = {}
with open(sys.argv[1]) as textfile1, open(sys.argv[2]) as textfile2 : 
    for x, y in zip(textfile1, textfile2):
        x = x.strip()
        y = y.strip()
        if y in sp_set:
            if y not in es_lines:
                n = n + 1
                es_lines[y] = n     
            print(f"{es_lines[y]}\t{x}")