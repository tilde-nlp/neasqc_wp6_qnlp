# Classical NLP

A module implementing classical processing of the datasets including text vectorization and classifier training.

## Vectorization

The folder *data_vectorisation* contains source for building services that vectorize text using pretrained embeddings. The services are run as Docker containers.

## Neural Network Classifiers

The folder *../../models/classical* contains source code of a class implementing neural network classifiers. Currently, a shallow feedforward neural network and a convolutional network are implemented.

## Data preparation and classifier training examples
This folder contains the workflow for training classifiers with two publicly available datasets from the *kaggle* - 'Amazon Fine Food Reviews' and 'Topic Labeled New Dataset'.

### Workflow
1. Download the datasets:
	- *Reviews.csv* from the <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/versions/1>
	- *labelled_newscatcher_dataset.csv* from the <https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset>
	
2. The data must be filtered as there are some constraints on sentence length and syntactical structure in the NEASQC project.
	- use scripts `filter-reviews-corpus-with-spacy-preprocessing.py` and `filter-newscatcher-corpus-with-spacy-preprocessing.py` to filter data using punctuation normalization.
	
	or
	
	- use scripts `filter-reviews-corpus.py` and `filter-newscatcher-corpus.py` to filter data without using punctuation normalization.
	
	These scripts determine syntactical structure of text. Text lines longer than 6 tokens are omitted. As a result 3-column files are created containing class, text and syntactic tags of the text.
	
	- use script `filter-by_syntax.py` to select the text lines with certain syntactical structure. The script has the following parameters:
	
		- *-i* or *--infile* \<TAB separated 3-column file to filter\>
		- *-o* or *--outfile* \<Filtered 2-column file\>
		- *-f* or *--filterfile* \<File containing list of preferable syntactical tags\> 
		
This step can be skipped as allready filtered data `reviews.tsv` and `labelled_newscatcher_dataset.tsv` are included in *../datasets* folder.

3. Using Jupyter notebook `Prepare_datasets_4classifier.ipynb` split data in train/test parts and obtain embeddings using transformer sentence-level embeddings and fastText word-level embeddings.<br>
Prior to running this notebook the following Python libraries must be installed:

`pip install -U sentence-transformers`

4. Train classifier using script `../../models/classical/NNClassifier.py`. The script has the following parameters:
	- *-i* or *--input* \<json data file for classifier training with embeddings\>
	- *-f* or *--field* \<field name containing labels for classification\>
	- *-t* or *--type* \<embedding type: 'sentence' or 'word'\>

Examples:

`python ../../models/classical/NNClassifier.py -i ../datasets/reviews_FASTTEXT.json -f "class" -t "word"`

`python ../../models/classical/NNClassifier.py -i ../datasets/reviews_BERT_CASED.json -f "class" -t "sentence"`

### New Results

|                                                                                                                    | review.tsv<br>(train: 53329, test: 5926)<br>8 classes | labelled_newscatcher_dataset.tsv<br>(train: 2317, test: 258)<br>5 classes |
|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------|
| fastText word emb.: *cc.en.300.bin*<br>NN model type: Convolutional network (max sentence length 6)                | Train accuracy: 0.7811<br>Test accuracy: 0.6718       | Train accuracy: 0.9914<br>Test accuracy: 0.7054                           |
| Transformer sentence emb.: *paraphrase-xlm-r-multilingual-v1*<br>NN model type: Shallow feedforward neural network | Train accuracy: 0.6784<br>Test accuracy: 0.6674       | Train accuracy: 0.9033<br>Test accuracy: 0.6318                           |
| Transformer sentence emb.: *LaBSE*<br>NN model type: Shallow feedforward neural network                            | Train accuracy: 0.6736<br>Test accuracy: 0.6687       | Train accuracy: 0.7613<br>Test accuracy:  0.6783                          |
| Transformer sentence emb.: *all-mpnet-base-v2*<br>NN model type: Shallow feedforward neural network                | Train accuracy: 0.6934<br>Test accuracy: 0.6846       | Train accuracy: 0.8261<br>Test accuracy: 0.6977                           |
| Transformer sentence emb.: *all-distilroberta-v1*<br>NN model type: Shallow feedforward neural network             | Train accuracy: 0.6835<br>Test accuracy: 0.6698       | Train accuracy: 0.8187<br>Test accuracy: 0.7364                           |
| BERT word emb.: *bert-base-uncased*<br>NN model type: Shallow feedforward neural network                           | Train accuracy: 0.6733<br>Test accuracy: 0.6497       | Train accuracy: 0.90007<br>Test accuracy: 0.6357                          |
| BERT word emb.: *bert-base-cased*<br>NN model type: Shallow feedforward neural network                             | Train accuracy: 0.6609<br>Test accuracy: 0.6429       | Train accuracy: 0.8377<br>Test accuracy: 0.5814                           |
