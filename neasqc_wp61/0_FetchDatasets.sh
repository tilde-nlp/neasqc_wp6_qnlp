#!/bin/bash

echo 'If you do not have Kaggle API installed read instruction on https://www.endtoend.ai/tutorial/how-to-download-kaggle-datasets-on-ubuntu/'

$HOME/.local/bin/kaggle datasets download snap/amazon-fine-food-reviews/versions/1
python -c "from zipfile import PyZipFile; PyZipFile('amazon-fine-food-reviews.zip', mode='r').extract('Reviews.csv', path='./data/datasets/')"
rm amazon-fine-food-reviews.zip

$HOME/.local/bin/kaggle datasets download kotartemiy/topic-labeled-news-dataset
python -c "from zipfile import PyZipFile; PyZipFile('topic-labeled-news-dataset.zip', mode='r').extract('labelled_newscatcher_dataset.csv', path='./data/datasets/')"
rm topic-labeled-news-dataset.zip

$HOME/.local/bin/kaggle datasets download shuyangli94/food-com-recipes-and-user-interactions
python -c "from zipfile import PyZipFile; PyZipFile('food-com-recipes-and-user-interactions.zip', mode='r').extract('RAW_interactions.csv', path='./data/datasets/')"
rm food-com-recipes-and-user-interactions.zip

$HOME/.local/bin/kaggle datasets download kritanjalijain/amazon-reviews
python -c "from zipfile import PyZipFile; PyZipFile('amazon-reviews.zip', mode='r').extract('train.csv', path='./data/datasets/')"
rm amazon-reviews.zip

echo 'Datasets for sentence similarity detection'

$HOME/.local/bin/kaggle datasets download vladimirvorobevv/chatgpt-paraphrases
python -c "from zipfile import PyZipFile; PyZipFile('chatgpt-paraphrases.zip', mode='r').extract('chatgpt-paraphrases.csv', path='./data/datasets/')"
rm chatgpt-paraphrases.zip

wget -O en-fr.zip https://opus.nlpl.eu/download.php?f=Europarl/v8/moses/en-fr.txt.zip
python -c "from zipfile import PyZipFile; PyZipFile('en-fr.zip', mode='r').extract('Europarl.en-fr.en', path='./data/datasets/'); PyZipFile('en-fr.zip', mode='r').extract('Europarl.en-fr.fr', path='./data/datasets/')"
rm en-fr.zip

wget -O en-it.zip https://opus.nlpl.eu/download.php?f=Tatoeba/v2023-04-12/moses/en-it.txt.zip
python -c "from zipfile import PyZipFile; PyZipFile('en-it.zip', mode='r').extract('Tatoeba.en-it.en', path='./data/datasets/'); PyZipFile('en-it.zip', mode='r').extract('Tatoeba.en-it.it', path='./data/datasets/')"
rm en-it.zip