# Classical NLP

A module implementing classical processing of the datasets including text vectorization and classifier training.

## Neural Network Classifiers

The folder *../../models/classical* contains source code of a class implementing neural network classifiers. Currently, a shallow feedforward neural network and a convolutional network are implemented.

## Data preparation and classifier training examples
This folder contains the workflow for training classifiers with publicly available datasets from the *kaggle* - 'Amazon Fine Food Reviews', 'Topic Labeled New Dataset', 'Food.com Recipes and Interactions', and 'Amazon Reviews'.

### Workflow

The Workflow has 8 steps. Data preparation steps (Step 0 to Step 4) are common to the classical and the quantum algorithms.

To run each step use the corresponding bash script that is located in the directory *../data/data_preparation*. 

#### Step 0 - data download

We use the following datasets from the kaggle.com. 
	- *Reviews.csv* from the <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/versions/1>
	- *labelled_newscatcher_dataset.csv* from the <https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset>
	- *RAW_interactions.csv* from the <https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions?select=RAW_interactions.csv>
	- *train.csv* <https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=train.csv>

To retrive the datasets from the kaggle.com you first have to
	- install kaggle API using command `pip install --user kaggle`
	- create Kaggle account, create API token, make a directory .kaggle at root ~, and place kaggle.json in that directory.  
	
See instructions in <https://www.endtoend.ai/tutorial/how-to-download-kaggle-datasets-on-ubuntu/>

After run the script `0_FetchDatasets.sh` to download the datasets.

#### Step 1 - Tokenizing and parsing datasets using Bobcat parser. Text examples containing more that 6 tokens are skipped.

To perform this step run the script *1_Filter6Parse.sh* passing the following parameters:

	- *-i <dataset>*            Dataset file (with path)
	- *-d <delimiter>*          Field delimiter symbol
	- *-c <class fiels>*        Name of the class field (only if the first line in the file contains field names)
	- *-t <text field>*         Name of the text field (only if the first line in the file contains field names)

Examples:

`1_Filter6Parse.sh -i ../data/datasets/Reviews.csv -d ',' -c 'Score' -t 'Summary'`

`1_Filter6Parse.sh -i ../data/datasets/train.csv -d ','`

`1_Filter6Parse.sh -i ../data/datasets/labelled_newscatcher_dataset.csv -d ';' -c 'topic' -t 'title'`

Files with syntactic tags `Reviews_alltrees.tsv`, `labelled_newscatcher_dataset_alltrees.tsv`, `RAW_interactions_alltrees.tsv` are included in *../datasets* folder.

#### Step 2 - Selecting data with a certain syntactical structure.

To perform this step run the script *2_FilterSyntacticTrees.sh* passing the following parameters:

	- *-i <input file>*               TAB separated 3-column file to filter
	- *-f <syntactical trees file>*   File containing list of preferable syntactical tags

Example:

`2_FilterSyntacticTrees.sh -i ../data/datasets/Reviews_alltrees.tsv -f ../data/datasets/validtrees.txt`
	
#### Step 3 - Split data in train/dev/test parts with proportions 80/10/10.

To perform this step run the script *3_SplitTrainTestDev.sh* passing the following parameters:

	- *-i <input file>*         TAB separated 3-column filtered file

3 files fill be created containing suffix '_train', '_test' and '_dev' in the name.

Example:

`3_SplitTrainTestDev.sh -i ../data/datasets/labelled_newscatcher_dataset_filtered.tsv`

#### Step 4 - Acquire embedding vectors using chosen pre-trained embedding model.

To perform this step run the script *4_GetEmbeddings.sh* passing the following parameters:

	- *-i <input file>*      input file with text examples
	- *-c <column>*          '3' - if 3-column input file containing class, text and parse tree columns, '0' - if the whole line is a text example
	- *-m <embedding name>*  Name of the embedding model
	- *-t <embedding type>*  Type of the embedding model - 'fasttext', 'transformer' or 'bert'	

Run this step for all 3 parts of the dataset - train dev and test.

Example:

`4_GetEmbeddings.sh -i ../data/datasets/labelled_newscatcher_dataset_filtered_train.tsv -c '3' -m 'all-distilroberta-v1' -t 'transformer'`

`4_GetEmbeddings.sh -i ../data/datasets/labelled_newscatcher_dataset_filtered_test.tsv -c '3' -m 'all-distilroberta-v1' -t 'transformer'`

`4_GetEmbeddings.sh -i ../data/datasets/labelled_newscatcher_dataset_filtered_dev.tsv -c '3' -m 'all-distilroberta-v1' -t 'transformer'`


#### Step 5 - Training model.

Examples:

`./5_TrainNNModel.sh -t ../datasets/reviews_filtered_train_all-distilroberta-v1.json -d ../datasets/reviews_filtered_dev_all-distilroberta-v1.json -f 'class' -e 'sentence' -m ../../models/classical/reviews_distilroberta`

`./5_TrainNNModel.sh -t ../datasets/reviews_filtered_train_bert-base-uncased.json -d ../datasets/reviews_filtered_dev_bert-base-uncased.json -f 'class' -e 'sentence' -m ../../models/classical/reviews_bert-uncased`

`./5_TrainNNModel.sh -t ../datasets/amazonreview_train_filtered_train_fasttext.json -d ../datasets/amazonreview_train_filtered_dev_fasttext.json -f 'class' -e 'word' -m ../../models/classical/amazonreview_test_fasttext`


#### Step 6 - Using classifier.



#### Step 7 - Evaluating results.

To perform this step run the script *7_EvaluateResults.sh* passing the following parameters:

	- *-e <expected results file>* Expected results file containing class in the first column and optionaly other columns
	- *-c <expected results file>* Results acquired using classifier.

Example:

`7_EvaluateResults.sh -e ../data/datasets/labelled_newscatcher_dataset_filtered_test.tsv -c results.txt`

### Results

#### Results for data with noun phrases

Test examples has the following syntactical structure:

`n[(n/n)   n[(n/n)   n]]`

`n[(n/n)   n[(n/n)   n[(n/n)   n]]]`

`n[n[n[(n/n)   n]] (n\\n)[((n\\n)/n)   n[n[(n/n)   n]]]]`

`n[(n/n)[((n/n)/(n/n))   (n/n)] n]`

`n[n[n[(n/n)   n]] (n\\n)[((n\\n)/n)   n[(n/n)   n]]]`

|                                                                                                                    | review.tsv<br>(train: 27001, test: 3001)<br>5 classes | labelled_newscatcher_dataset.tsv<br>(train: 160, test: 18)<br>8 classes | amazonreview_train.tsv<br>(train: 183918, test: 20436)<br>2 classes |
|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------|
| fastText word emb.: *cc.en.300.bin*<br>NN model type: Convolutional network (max sentence length 6)                | Train accuracy: 0.8677<br>Test accuracy: 0.7223       | Train accuracy: 0.6111<br>Test accuracy: 0.9929                         | Train accuracy: 0.9106<br>Test accuracy: 0.8303                     |
| Transformer sentence emb.: *all-mpnet-base-v2*<br>NN model type: Shallow feedforward neural network                | Train accuracy: 0.7531<br>Test accuracy: 0.7426       | Train accuracy: 0.9366<br>Test accuracy: 0.7222                         | Train accuracy: 0.8400<br>Test accuracy: 0.8344                     |
| Transformer sentence emb.: *all-distilroberta-v1*<br>NN model type: Shallow feedforward neural network             | Train accuracy: 0.7465<br>Test accuracy: 0.7330       | Train accuracy: 0.9366<br>Test accuracy: 0.7777                         | Train accuracy: 0.8264<br>Test accuracy: 0.8231                     |
| BERT word emb.: *bert-base-uncased*<br>NN model type: Shallow feedforward neural network                           | Train accuracy: 0.7385<br>Test accuracy: 0.7256       | Train accuracy: 0.9929<br>Test accuracy: 0.7777                         | Train accuracy: 0.8115<br>Test accuracy: 0.8073                     |
| BERT word emb.: *bert-base-cased*<br>NN model type: Shallow feedforward neural network                             | Train accuracy: 0.7324<br>Test accuracy: 0.7163       | Train accuracy: 1.0000<br>Test accuracy: 0.7222                         | Train accuracy: 0.7810<br>Test accuracy: 0.7788                     |

#### Results for data with sentences

Test examples has the following syntactical structure:

`s[n[n] (s\\n)[((s\\n)/(s\\n))   (s\\n)]]`

`s[n[(n/n)   n] (s\\n)[((s\\n)/(s\\n))   (s\\n)]]`

`s[n[n[(n/n)   n]] (s\\n)]`

`s[n   (s\\n)[((s\\n)/n)   n[(n/n)   n]]]`

`s[n   (s\\n)[((s\\n)/n)   n[(n/n)   n[(n/n)   n]]]]`

Dataset *labelled_newscatcher_dataset* has only 12 examples and *RAW_interactions* has only 77 examples after filtering. As the number of examples is too small models are not trained for these datasets.

|                                                                                                                    | review.tsv<br>(train: 0, dev: , test: 0)<br>5 classes | amazonreview_train.tsv<br>(train: 16853, dev: 2107, test: 2107)<br>2 classes |
|--------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------|-----------------------------------------------------------------------|
| fastText word emb.: *cc.en.300.bin*<br>NN model type: Convolutional network (max sentence length 6)                | Train accuracy: 0.9524<br>Test accuracy: 0.0             | Train accuracy: 0.9576<br>Test accuracy: 0.8287                             |
| Transformer sentence emb.: *all-mpnet-base-v2*<br>NN model type: Shallow feedforward neural network                | Train accuracy: 0.7726<br>Test accuracy: 0.0             | Train accuracy: 0.8739<br>Test accuracy: 0.8638                             |
| Transformer sentence emb.: *all-distilroberta-v1*<br>NN model type: Shallow feedforward neural network             | Train accuracy: 0.7707<br>Test accuracy: 0.0             | Train accuracy: 0.8587<br>Test accuracy: 0.8415                             |
| BERT word emb.: *bert-base-uncased*<br>NN model type: Shallow feedforward neural network                           | Train accuracy: 0.8208<br>Test accuracy: 0.0             | Train accuracy: 0.8420<br>Test accuracy: 0.8249                             |
| BERT word emb.: *bert-base-cased*<br>NN model type: Shallow feedforward neural network                             | Train accuracy: 0.7964<br>Test accuracy: 0.0             | Train accuracy: 0.8060<br>Test accuracy: 0.7864                             |

