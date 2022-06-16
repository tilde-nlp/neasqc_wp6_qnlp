# Classical NLP

A module implementing classical processing of the datasets including text vectorization and classifier training.

## Vectorization

The folder *DataVectorizer* contains source for building services that vectorize text using pretrained embeddings. The services are run as Docker containers.

## Neural Network Classifiers

The folder *Classifier* contains source code of a class implementing neural network classifiers. Currently, a shallow feedforward neural network and a convolutional network are implemented.

## Data preparation and classifier training examples
This folder contains the workflow for training classifiers with two publicly available datasets from the *kaggle* - 'Amazon Fine Food Reviews' and 'Topic Labeled New Dataset'.

### Workflow
1. Download the datasets:
	- *Reviews.csv* from the <https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset>
	- *labelled_newscatcher_dataset.csv* from the <https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset>
	
2. The data must be filtered as there are some constraints on sentence length and syntactical structure in the NEASQC project.
	- using script `train-truecaser.sh` train the truecaser model on a large English data corpus available at <https://data.statmt.org/>
	- use scripts `filter-reviews-corpus-with-moses-preprocessing.py` and `filter-newscatcher-corpus-with-moses-preprocessing.py` to filter data using punctuation normalization and truecasing.
	
	or
	
	- use scripts `filter-reviews-corpus-with-moses-preprocessing.py` and `filter-newscatcher-corpus-with-moses-preprocessing.py` to filter data without using punctuation normalization and truecasing.
	
This step can be skipped as allready filtered data `reviews.tsv` and `labelled_newscatcher_dataset.tsv` are included in this folder.

3. Using Jupyter notebook `Prepare_datasets_4classifier.ipynb` split data in train/test parts and obtain embeddings using cased and uncased BERT sentence-level embeddings and fastText word-level embeddings.<br>
Prior to running this notebook three Docker containers should be started serving BERT and fastText embeddings.<br>
Pre-trained embedding models are specified in *Dockerfile* of the containers. Code of containers are located in the floder *DataVectorizer*.

4. Train classifier using script `./Classifier/NNClassifier.py`. The script has the following parameters:
	- *-i* or *--input* \<json data file for classifier training with embeddings\>
	- *-f* or *--field* \<field name containing labels for classification\>
	- *-t* or *--type* \<embedding type: 'sentence' or 'word'\>

Examples:

`python ./Classifier/NNClassifier.py -i reviews_FASTTEXT.json -f "class" -t "word"`

`python ./Classifier/NNClassifier.py -i reviews_BERT_CASED.json -f "class" -t "sentence"`

### Results

|                                                                        | BERT sentence emb.: *cased_L-12_H-768_A-12*<br>NN model type: Shallow feedforward neural network | BERT sentence emb.: *uncased_L-12_H-768_A-12*<br>NN model type: Shallow feedforward neural network | fastText word emb.: *wiki.en.bin*<br>NN model type: Convolutional network   with max sentence length 5                           |
|------------------------------------------------------------------------|-----------------------------------------------------|-----------------------------------------------------|-------------------------------------------------------------------|
| review.tsv<br>(train: 53329, test: 5926)<br>8 classes                     | Train accuracy:   0.6702<br>Test accuracy: 0.6627 | Train accuracy: 0.6767<br>Test accuracy: 0.6591 | Train accuracy: 0.7717<br>Test accuracy: 0.6628 |
| labelled_newscatcher_dataset.tsv<br>(train: 2317, test: 258)<br>5 classes | Train accuracy:   0.8934<br>Test accuracy: 0.5233 | Train accuracy: 0.9413<br>Test accuracy: 0.6240 | Train accuracy: 0.9965<br>Test accuracy: 0.6008 |