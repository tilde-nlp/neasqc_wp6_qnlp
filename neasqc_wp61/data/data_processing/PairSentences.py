import sys
import re

def StripSentence(snt):
    snt = snt.lower()
    snt = re.sub(r"[,.:;!?\-\d\"\[\]]","",snt)
    snt = re.sub(r"\b(i'm|i've|hasn't|doesn't|didn't|she's|isn't|aren't|they're|what'll|he's|we've|couldn't|can't|you're|we're|should've|it's|don't|that's|we'll|i'll|it'll|won't|ain't|where's|what's|shouldn't)\b",'',snt)
    snt = re.sub(r"\b('m|'ve|n't|'s|'re|'ll)\b",'',snt)
    snt = re.sub(r"\b(they|i|she|he|you|his|our|what|who|where|their|her|him|we|in|at|of|someone|somebody|anyone|anybody|any|me|the|or|and|an|that|this|that|it|a)\b",'',snt)
    snt = re.sub(r"\b(cannot|done|could|might|has|wanna|want|got|going|done|did|are|should|have|must|could|to|do|not|had|does|is|was|will|where|am|were|would|no|be|have|is)\b",'',snt)
    snt = re.sub(r"'s\b",'',snt)
    snt = re.sub(r" +",' ',snt)
    return snt
    
def main():
    id = 0
    similarsent = set()
    comparesent = set()
    printed = set()
    notprinted = set()
    trees={}
    numTrue=0
    
    with open(sys.argv[1]) as textfile1: 
        for x in  textfile1:
            items = x.strip().split('\t')
            if len(items) == 3:
                items[1] = re.sub(r"[\(\)]","",items[1])
                if len(items[1])>5 and items[1].isascii():
                    stripped=StripSentence(items[1])
                    trees[items[1]]=items[2]
                    if items[0] ==id:
                        if stripped not in comparesent:
                            similarsent.add(items[1])
                            comparesent.add(stripped)
                    else:
                        if len(similarsent)>1 and ';'.join(similarsent) not in printed:
                            printed.add(';'.join(similarsent))
                            s1 = similarsent.pop()
                            s2 = similarsent.pop()
                            print(f"True\t{s1} | {s2}\t{trees[s1]} | {trees[s2]}")
                            numTrue = numTrue + 1
                            notprinted.update(similarsent)
                        elif len(similarsent)==1:
                            notprinted.update(similarsent)
                            
                        similarsent.clear()
                        comparesent.clear()
                        similarsent.add(items[1])
                        comparesent.add(stripped)
                        id = items[0]
                        
    for i in range(min(numTrue, int(len(notprinted)/2))):
        s1 = notprinted.pop()
        s2 = notprinted.pop()
        print(f"False\t{s1} | {s2}\t{trees[s1]} | {trees[s2]}")
        
if __name__ == "__main__":
    main()