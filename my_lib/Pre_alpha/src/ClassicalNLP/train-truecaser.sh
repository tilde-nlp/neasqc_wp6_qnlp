
toolsdir=./scripts

corpus=news.2021.en.shuffled.deduped

# Download an English monolingual corpus
wget https://data.statmt.org/news-crawl/en/$corpus.gz

# Extract monolingual data
gunzip $corpus.gz

# Normalise data
perl $toolsdir/normalize-punctuation.perl -l en < $corpus > $corpus.norm

# Tokenise data
perl $toolsdir/tokenizer.perl -no-escape -l en < $corpus.norm > /tmp/$corpus.tok

# Train a truecasing model
perl $toolsdir/train-truecaser.perl --model $corpus.tc.model --corpus /tmp/$corpus.tok
