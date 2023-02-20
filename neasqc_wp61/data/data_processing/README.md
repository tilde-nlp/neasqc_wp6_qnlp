# Classical NLP

A module implementing classical processing of the datasets including text vectorization and classifier training.

## Vectorization

The folder *data_vectorisation* contains source for building services that vectorize text using pretrained embeddings. The services are run as Docker containers.

## Neural Network Classifiers

The folder *../../models/classical* contains source code of a class implementing neural network classifiers. Currently, a shallow feedforward neural network and a convolutional network are implemented.

## Data preparation and classifier training examples
This folder contains the workflow for training classifiers with publicly available datasets from the *kaggle* - 'Amazon Fine Food Reviews', 'Topic Labeled New Dataset', 'Food.com Recipes and Interactions', and 'Amazon Reviews'.

### Workflow
1. Download the datasets:
	- *Reviews.csv* from the <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/versions/1>
	- *labelled_newscatcher_dataset.csv* from the <https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset>
	- *RAW_interactions.csv* from the <https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_interactions.csv>
	- *train.csv* <https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=train.csv> (rename it *amazonreview_train.csv*)
	
2. The data must be filtered as there are some constraints on sentence length and syntactical structure in the NEASQC project.

It can be done either using Jupyter notebook `Prepare_datasets_4classifier.ipynb` in *../../doc/tutorials* folder or using separate scripts for each step.

	- use scripts `filter-with-spacy-preprocessing.py` to filter data using punctuation normalization. The script has the following parameters:
	
		- *-i* or *--infile* \<Comma separated CSV data file\>
		- *-o* or *--outfile* \<TAB separated 3-column file\>  (syntactic tags in the 3rd column)
		- *-c* or *--classfield* \<Name of the field containing class\>  (this parameter should be skipped if the first line of the file does not contain column names)	
		- *-t* or *--txtfield* \<Name of the field containing text\>  (this parameter should be skipped if the first line of the file does not contain column names)	
	
	These scripts determine syntactical structure of text. Text lines longer than 6 tokens are omitted. As a result 3-column files are created containing class, text and syntactic tags of the text.

Files with syntactic tags `withtags_Reviews.tsv`, `withtags_labelled_newscatcher_dataset.tsv`, `withtags_RAW_interactions.tsv` and `withtags_amazonreview_train.tsv` are included in *../datasets* folder.
	
	- use script `filter-by_syntax.py` to select the text lines with certain syntactical structure. The script has the following parameters:
	
		- *-i* or *--infile* \<TAB separated 3-column file to filter\>
		- *-o* or *--outfile* \<Filtered 2-column file\>
		- *-f* or *--filterfile* \<File containing list of preferable syntactical tags\> 
		
Allready filtered data `reviews.tsv`, `labelled_newscatcher_dataset.tsv`, `RAW_interactions.tsv` and `withtags_amazonreview_train.tsv` are included in *../datasets* folder. These files contain text examples that parse as sentences.

3. Split data in train/test parts and obtain embeddings using sentence transformer and BERT sentence-level embeddings and fastText word-level embeddings.

This step is performed in the Jupyter notebook `Prepare_datasets_4classifier.ipynb` after data filtering.

4. Train classifier using script `../../models/classical/NNClassifier.py`. The script has the following parameters:
	- *-i* or *--input* \<json data file for classifier training with embeddings\>
	- *-f* or *--field* \<field name containing labels for classification\>
	- *-t* or *--type* \<embedding type: 'sentence' or 'word'\>

Examples:

`python ../../models/classical/NNClassifier.py -i ../datasets/reviews_FASTTEXT.json -f "class" -t "word"`

`python ../../models/classical/NNClassifier.py -i ../datasets/reviews_BERT_CASED.json -f "class" -t "sentence"`

### New Results

|                                                                                                                    | review.tsv<br>(train: 53329, test: 5926)<br>8 classes | labelled_newscatcher_dataset.tsv<br>(train: 2317, test: 258)<br>5 classes | RAW_interactions.tsv<br>(train: 3319, test: 369)<br>5 classes |
|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|---------------------------------------------------------------------------|---------------------------------------------------------------|
| fastText word emb.: *cc.en.300.bin*<br>NN model type: Convolutional network (max sentence length 6)                | Train accuracy: 0.7811<br>Test accuracy: 0.6718       | Train accuracy: 0.9914<br>Test accuracy: 0.7054                           | Train accuracy: 0.9259<br>Test accuracy: 0.813                |
| Transformer sentence emb.: *paraphrase-xlm-r-multilingual-v1*<br>NN model type: Shallow feedforward neural network | Train accuracy: 0.6784<br>Test accuracy: 0.6674       | Train accuracy: 0.9033<br>Test accuracy: 0.6318                           |                                                               |
| Transformer sentence emb.: *LaBSE*<br>NN model type: Shallow feedforward neural network                            | Train accuracy: 0.6736<br>Test accuracy: 0.6687       | Train accuracy: 0.7613<br>Test accuracy:  0.6783                          |                                                               |
| Transformer sentence emb.: *all-mpnet-base-v2*<br>NN model type: Shallow feedforward neural network                | Train accuracy: 0.6934<br>Test accuracy: 0.6846       | Train accuracy: 0.8261<br>Test accuracy: 0.6977                           | Train accuracy: 0.8741<br>Test accuracy: 0.8184               |
| Transformer sentence emb.: *all-distilroberta-v1*<br>NN model type: Shallow feedforward neural network             | Train accuracy: 0.6835<br>Test accuracy: 0.6698       | Train accuracy: 0.8187<br>Test accuracy: 0.7364                           | Train accuracy: 0.8729<br>Test accuracy: 0.8265               |
| BERT word emb.: *bert-base-uncased*<br>NN model type: Shallow feedforward neural network                           | Train accuracy: 0.6733<br>Test accuracy: 0.6497       | Train accuracy: 0.90007<br>Test accuracy: 0.6357                          | Train accuracy: 0.884<br>Test accuracy: 0.7832                |
| BERT word emb.: *bert-base-cased*<br>NN model type: Shallow feedforward neural network                             | Train accuracy: 0.6609<br>Test accuracy: 0.6429       | Train accuracy: 0.8377<br>Test accuracy: 0.5814                           | Train accuracy: 0.8843<br>Test accuracy: 0.8347               |