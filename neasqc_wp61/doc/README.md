# NLP

A module implementing processing of the datasets including text vectorization and classifier training.

## Workflow

The Workflow has 8 steps. Data preparation steps (Step 0 to Step 4) are common to the classical and the quantum algorithms.

To run each step use the corresponding bash script that is located in the directory *../data/data_preparation*. 

### Step 0 - data download

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

The following datasets are used for the sentence similarity detection task.

- *chatgpt-paraphrases.csv* from the <https://www.kaggle.com/datasets/vladimirvorobevv/chatgpt-paraphrases>
- *Europarl.en-fr.en*, *Europarl.en-fr.fr* from the <https://opus.nlpl.eu/download.php?f=Europarl/v8/moses/en-fr.txt.zip>
- *Tatoeba.en-it.en*, *Tatoeba.en-it.it* from the <https://opus.nlpl.eu/download.php?f=Tatoeba/v2023-04-12/moses/en-it.txt.zip>

### Step 0_1 - Only for the datasets used in the sentence similarity detection task. Text examples are converted to the two-column format - the similar sentence ID (the same ID for the similar sentences) and the sentence.

To perform this step run the script *0_1_Preprocess_similar_sentences.sh*

### Step 1 - Tokenizing and parsing datasets using Bobcat parser. Text examples containing more that 6 tokens are skipped.

To perform this step run the script *1_Filter6Parse.sh* passing the following parameters:

- -i \<dataset\>            Dataset file (with path)
- -d \<delimiter\>          Field delimiter symbol
- -c \<class fiels\>        Name of the class field (only if the first line in the file contains field names)
- -t \<text field\>         Name of the text field (only if the first line in the file contains field names)
- -g \<gpu use\>            Number of GPU to use (from 0 to available GPUs), -1 if use CPU (default is -1)

Examples:

`1_Filter6Parse.sh -i ./data/datasets/Reviews.csv -d ',' -c 'Score' -t 'Summary' -g '0'`

`1_Filter6Parse.sh -i ./data/datasets/train.csv -d ','  -g '1'`

`1_Filter6Parse.sh -i ./data/datasets/labelled_newscatcher_dataset.csv -d ';' -c 'topic' -t 'title' -g '-1'`

Files with syntactic tags `Reviews_alltrees.tsv`, `labelled_newscatcher_dataset_alltrees.tsv`, `RAW_interactions_alltrees.tsv` are included in *../data/datasets* folder.

The name of the dataset *train* in further steps is changed to *amazonreviews_train* for clarity.

Examples with the sentence similarity datasets:

`1_Filter6Parse.sh -i ./data/datasets/sorted_tatoeba_it.txt -d $'\t' -g '-1'`

`1_Filter6Parse.sh -i ./data/datasets/sorted_Europarl_fr.txt -d $'\t' -g '-1'`

`1_Filter6Parse.sh -i ./data/datasets/chatgpt_paraphrases.txt -d $'\t' -g '-1'`

### Step 2 - Selecting data with a certain syntactical structure.

To perform this step run the script *2_FilterSyntacticTrees.sh* passing the following parameters:

- -i \<input file\>               TAB separated 3-column file to filter
- -f \<syntactical trees file\>   File containing list of preferable syntactical tags

Example:

`2_FilterSyntacticTrees.sh -i ./data/datasets/amazonreviews_train_alltrees.tsv -f ./data/datasets/validtrees.txt`

Example with the sentence similarity dataset:

`2_FilterSyntacticTrees.sh -i ./data/datasets/chatgpt_paraphrases_alltrees.tsv -f ./data/datasets/similarity_validtrees.txt`

### Step 2_3 - Only for the datasets used in the sentence similarity detection task. Similar sentences are paired. Unsimilar sentence pairs are created from the random sentences. All 3 datasets are merged.

To perform this step run the script *2_3_Sentence_pairing.sh*
	
### Step 3 - Spliting data in train/dev/test parts.

To perform this step run the script *3_SplitTrainTestDev.sh* passing the following parameters:

- -i \<input file\>         TAB separated 3-column filtered file
- -r         				Parameter for the random stratified splitting. If this parameter is omitted then the split method will guarantee that the words in the test/dev sets are also in the train set. 

3 files fill be created containing suffix '_train', '_test' and '_dev' in the name.

Example:

`3_SplitTrainTestDev.sh -i ./data/datasets/amazonreviews_train_filtered.tsv`

`3_SplitTrainTestDev.sh -i ./data/datasets/paired_similarity_filtered.tsv`

### Step 4 - Acquiring embedding vectors using chosen pre-trained embedding model.

We have experimented with 5 different pre-trained embedding models.

|   | Pre-trained model      | Embedding type | Vectors for unit | About model                                                                                                                                                                                        |
|---|------------------------|----------------|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   | *cc.en.300.bin*        | fasttext       | word             | Model trained on Common Crawl and Wikipedia using fastText.<br><https://fasttext.cc/docs/en/crawl-vectors.html>                                                                                    |
|   | *all-mpnet-base-v2*    | transformer    | sentence         | The best among the sentence transformers general purpose model trained on all available training data (more than 1 billion training pairs).<br><https://www.sbert.net/docs/pretrained_models.html> |
|   | *all-distilroberta-v1* | transformer    | sentence         | The third best sentence transformers model, faster that *all-mpnet-base-v2*.<br><https://www.sbert.net/docs/pretrained_models.html>                                                                |
|   | *bert-base-uncased*    | bert           | sentence, word   | Case insensitive model pretrained on BookCorpus (consisting of 11,038 unpublished books) and English Wikipedia.<br><https://huggingface.co/bert-base-uncased>                                      |
|   | *bert-base-cased*      | bert           | sentence, word   | Model pretrained on BookCorpus (consisting of 11,038 unpublished books) and English Wikipedia.<br><https://huggingface.co/bert-base-cased>                                                         |

Bert models are older; and they are slower that sentence transformer models.

To perform this step run the script *4_GetEmbeddings.sh* passing the following parameters:

- -i <input file>      input file with text examples
- -c <column>          '3' - if 3-column input file containing class, text and parse tree columns, '0' - if the whole line is a text example
- -m <embedding name>  Name of the embedding model
- -t <embedding model type>  Type of the embedding model - 'fasttext', 'transformer' or 'bert'	
- -e <embedding unit>  Embedding unit: 'sentence' or 'word'	
- -g <gpu use>         Number of GPU to use (from 0 to available GPUs), -1 if use CPU (default is -1)

Fasttext model works only on CPU.

Embedding unit is 'word' or 'sentence' for the 'bert' models; 'word' for the 'fasttext' model; 'sentence' for the 'transformer' models. 

Run this step for all 3 parts of the dataset - train, dev and test.

Example:

`4_GetEmbeddings.sh -i ./data/datasets/amazonreview_train_filtered_train.tsv -c '3' -m 'all-distilroberta-v1' -t 'transformer' -e 'sentence' -g '0'`

`4_GetEmbeddings.sh -i ./data/datasets/amazonreview_train_filtered_test.tsv -c '3' -m 'all-distilroberta-v1' -t 'transformer' -e 'sentence' -g '0'`

`4_GetEmbeddings.sh -i ./data/datasets/amazonreview_train_filtered_dev.tsv -c '3' -m 'all-distilroberta-v1' -t 'transformer' -e 'sentence' -g '0'`

`4_GetEmbeddings.sh -i ./data/datasets/amazonreview_train_filtered_train.tsv -c '3' -m 'cc.en.300.bin' -t 'fasttext' -e 'word'`

`4_GetEmbeddings.sh -i ./data/datasets/amazonreview_train_filtered_test.tsv -c '3' -m 'cc.en.300.bin' -t 'fasttext' -e 'word'`

`4_GetEmbeddings.sh -i ./data/datasets/amazonreview_train_filtered_dev.tsv -c '3' -m 'cc.en.300.bin' -t 'fasttext' -e 'word'`

`4_GetEmbeddings.sh -i ./data/datasets/amazonreview_train_filtered_train.tsv -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '0'`

`4_GetEmbeddings.sh -i ./data/datasets/amazonreview_train_filtered_test.tsv -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '0'`

`4_GetEmbeddings.sh -i ./data/datasets/amazonreview_train_filtered_dev.tsv -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '0'`


### Step 5 - Training model.

#### Classical NLP

The folder *./models/classical* contains the source code of a class implementing neural network classifiers. Currently, a shallow feedforward neural network, a convolutional network and Bidirectional LSTM neural network are implemented.

To perform this step for the classification task run the script *5_TrainNNModel.sh* passing the following parameters:

- -t \<train data file\> Json data file for classifier training (with embeddings) or tsv file (if not using pre-trained embeddings, acquired using script 3_SplitTrainTestDev.sh)
- -d \<dev data file\>   Json data file for classifier validation (with embeddings) or tsv file (if not using pre-trained embeddings, acquired using script 3_SplitTrainTestDev.sh)
- -f \<field\>           Classify by field
- -e \<embedding unit\>  Embedding unit: 'sentence', 'word', or '-' (if not using pre-trained embeddings)
- -m \<model directory\> Directory where to save trained model
- -g \<gpu use\>         Number of GPU to use (from 0 to available GPUs), -1 if use CPU (default is -1)
	
Examples:

`5_TrainNNModel.sh -t ./data/datasets/amazonreview_train_filtered_train_bert-base-cased.json -d ./data/datasets/amazonreview_train_filtered_dev_bert-base-cased.json -f 'class' -e 'sentence' -m ./models/classical/amazonreview_train_bert-cased -g '0'`

`5_TrainNNModel.sh -t ./data/datasets/amazonreview_train_filtered_train_all-distilroberta-v1.json -d ./data/datasets/amazonreview_train_filtered_dev_all-distilroberta-v1.json -f 'class' -e 'sentence' -m ./models/classical/amazonreview_train_distilroberta -g '0'`

`5_TrainNNModel.sh -t ./data/datasets/amazonreview_train_filtered_train_fasttext.json -d ./data/datasets/amazonreview_train_filtered_dev_fasttext.json -f 'class' -e 'word' -m ./models/classical/amazonreview_train_fasttext -g '0'`

`5_TrainNNModel.sh -t ./data/datasets/amazonreview_train_filtered_train.tsv -d ./data/datasets/amazonreview_train_filtered_dev.tsv -f 'class' -e '-' -m ./models/classical/amazonreview_train -g '0'`

To perform this step for the sentence similarity task run the script *5_TrainEmbeddings.sh* passing the following parameters:

- -t \<train data file\> tsv file acquired using script 3_SplitTrainTestDev.sh
- -m \<model directory\> Directory where to save trained embedding model

Embeddings are trained only on the texts found in the training data.
	
Examples:

`5_TrainEmbeddings.sh -t ./data/datasets/paired_similarity_filtered_train.tsv -m ./models/classical/fasttext`

#### Quantum NLP

????

### Step 6 - Using classifier or predicting sentence similarity.

#### Classical NLP

For classification run the script *6_ClassifyWithNNModel.sh* passing the following parameters:

- -i \<input file\> 	   Json data file for classifier testing (with embeddings acquired using script 4_GetEmbeddings.sh) or tsv file (if not using pre-trained embeddings, acquired using script 3_SplitTrainTestDev.sh)
- -o \<output file\>     Result file with predicted classes
- -e \<embedding unit\>  Embedding unit: 'sentence', 'word', or '-' (if not using pre-trained embeddings)
- -m \<model directory\> Directory of pre-tained classifier model
- -g \<gpu use\>         Number of GPU to use (from 0 to available GPUs), -1 if use CPU (default is -1)

Examples for the classification task:

`6_ClassifyWithNNModel.sh -i ./data/datasets/amazonreview_train_filtered_test_bert-base-uncased.json -o ./benchmarking/results/raw/amazonreview_train_bert-base-uncased.txt -e 'sentence' -m ./models/classical/amazonreview_train_bert-uncased -g '0'`

`6_ClassifyWithNNModel.sh -i ./data/datasets/amazonreview_train_filtered_test_all-distilroberta-v1.json -o ./benchmarking/results/raw/amazonreview_train_distilroberta.txt -e 'sentence' -m ./models/classical/amazonreview_train_distilroberta -g '0'`

`6_ClassifyWithNNModel.sh -i ./data/datasets/amazonreview_train_filtered_test_fasttext.json -o ./benchmarking/results/raw/amazonreview_train_fasttext.txt -e 'word' -m ./models/classical/amazonreview_train_fasttext -g '0'`

`6_ClassifyWithNNModel.sh -i ./data/datasets/amazonreview_train_filtered_test.tsv -o ./benchmarking/results/raw/amazonreview_train.txt -e '-' -m ./models/classical/amazonreview_train -g '0'`

For similarity detection run the script *6_DetectSimilarity.sh* passing the following parameters:

- -i \<input file\> 	 tsv file (acquired using script 3_SplitTrainTestDev.sh)
- -o \<output file\>     Result file with predicted classes
- -m \<model directory\> Directory of the vectorization model trained in Step 5

Examples for the sentence similarity task:

If model directory not specified we use pre-trained LaBSE embeddings from <https://huggingface.co/sentence-transformers/LaBSE>

`6_DetectSimilarity.sh -i ./data/datasets/paired_similarity_filtered_test.tsv -o ./benchmarking/results/raw/paired_similarity_labse.txt`

`6_DetectSimilarity.sh -i ./data/datasets/paired_similarity_filtered_test.tsv -o ./benchmarking/results/raw/paired_similarity_fasttext.txt -m ./models/classical/fasttext`

#### Quantum NLP

????

### Step 7 - Evaluating results.

To perform this step run the script *7_EvaluateResults.sh* passing the following parameters:

- -e \<expected results file\> Expected results file containing class in the first column and optionaly other columns
- -c \<classifier results file\> Results acquired using classifier.
- -o \<accuracy file\>         File for calculated test accuracy

Example:

`7_EvaluateResults.sh -e ./data/datasets/amazonreview_train_filtered_test.tsv -c ./benchmarking/results/raw/amazonreview_train_distilroberta.txt -o ./benchmarking/results/amazonreview_train_distilroberta_acc.txt`

`7_EvaluateResults.sh -e ./data/datasets/paired_similarity_filtered_test.tsv -c ./benchmarking/results/raw/paired_similarity_labse.txt -o ./benchmarking/results/paired_similarity_labse_acc.txt`

`7_EvaluateResults.sh -e ./data/datasets/paired_similarity_filtered_test.tsv -c ./benchmarking/results/raw/paired_similarity_fasttext.txt -o ./benchmarking/results/paired_similarity_fasttext_acc.txt`

## Results

### Results for classification task (data with noun phrases)

Test examples have the following syntactical structure:

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
| BERT sentence emb.: *bert-base-uncased*<br>NN model type: Shallow feedforward neural network                       | Train accuracy: 0.7385<br>Test accuracy: 0.7256       | Train accuracy: 0.9929<br>Test accuracy: 0.7777                         | Train accuracy: 0.8115<br>Test accuracy: 0.8073                     |
| BERT sentence emb.: *bert-base-cased*<br>NN model type: Shallow feedforward neural network                         | Train accuracy: 0.7324<br>Test accuracy: 0.7163       | Train accuracy: 1.0000<br>Test accuracy: 0.7222                         | Train accuracy: 0.7810<br>Test accuracy: 0.7788                     |

### Results for classification task (data with sentences)

Test examples have the following syntactical structure:

`s[n[n] (s\\n)[((s\\n)/(s\\n))   (s\\n)]]`

`s[n[(n/n)   n] (s\\n)[((s\\n)/(s\\n))   (s\\n)]]`

`s[n[n[(n/n)   n]] (s\\n)]`

`s[n   (s\\n)[((s\\n)/n)   n[(n/n)   n]]]`

`s[n   (s\\n)[((s\\n)/n)   n[(n/n)   n[(n/n)   n]]]]`

Dataset *labelled_newscatcher_dataset* has only 12 examples and *RAW_interactions* has only 77 examples after filtering. As the number of examples is too small models are not trained for these datasets.
The train/test/dev sets are split using the random stratified method.

|                                                                                                                    | reviews.tsv<br>(train: 1596, dev: 199, test: 199)<br>5 classes | amazonreview_train.tsv<br>(train: 16853, dev: 2107, test: 2107)<br>2 classes |
|--------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------------------|
| fastText word emb.: *cc.en.300.bin*<br>NN model type: Convolutional network (max sentence length 6)                | Train accuracy: 0.9411<br>Test accuracy: 0.6985                | Train accuracy: 0.9576<br>Test accuracy: 0.8330                              |
| Transformer sentence emb.: *all-mpnet-base-v2*<br>NN model type: Shallow feedforward neural network                | Train accuracy: 0.7757<br>Test accuracy: 0.7387                | Train accuracy: 0.8746<br>Test accuracy: 0.8647                              |
| Transformer sentence emb.: *all-distilroberta-v1*<br>NN model type: Shallow feedforward neural network             | Train accuracy: 0.7726<br>Test accuracy: 0.7186                | Train accuracy: 0.8590<br>Test accuracy: 0.8420                              |
| BERT sentence emb.: *bert-base-uncased*<br>NN model type: Shallow feedforward neural network                       | Train accuracy: 0.8189<br>Test accuracy: 0.6533                | Train accuracy: 0.8353<br>Test accuracy: 0.8287                              |
| BERT sentence emb.: *bert-base-cased*<br>NN model type: Shallow feedforward neural network                         | Train accuracy: 0.8020<br>Test accuracy: 0.6533                | Train accuracy: 0.8044<br>Test accuracy: 0.7902                              |
| BERT word emb.: *bert-base-uncased*<br>NN model type: Convolutional network (max sentence length 6)                | Train accuracy: 0.8791<br>Test accuracy: 0.6683                | Train accuracy: 0.8838<br>Test accuracy: 0.8396                              |
| BERT word emb.: *bert-base-cased*<br>NN model type: Convolutional network (max sentence length 6)                  | Train accuracy: 0.8847<br>Test accuracy: 0.5678                | Train accuracy: 0.8740<br>Test accuracy: 0.8187                              |


Different split method. The words in the test/dev sets are also in the train set.

|                                                                                                                    | reviews.tsv<br>(train: 1864, dev: 66, test: 65)<br>5 classes   | amazonreview_train.tsv<br>(train: 18961, dev: 1053, test: 1053)<br>2 classes |
|--------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------------------|
| fastText word emb.: *cc.en.300.bin*<br>NN model type: Convolutional network (max sentence length 6)                | Train accuracy: 0.9759<br>Test accuracy: 0.7538                | Train accuracy: 0.9647<br>Test accuracy: 0.8794                              |
| Transformer sentence emb.: *all-distilroberta-v1*<br>NN model type: Shallow feedforward neural network             | Train accuracy: 0.7715<br>Test accuracy: 0.7846                | Train accuracy: 0.8539<br>Test accuracy: 0.8955                              |
| BERT sentence emb.: *bert-base-uncased*<br>NN model type: Shallow feedforward neural network                       | Train accuracy: 0.8106<br>Test accuracy: 0.7846                | Train accuracy: 0.8331<br>Test accuracy: 0.8708                              |
| BERT word emb.: *bert-base-uncased*<br>NN model type: Convolutional network (max sentence length 6)                | Train accuracy: 0.8471<br>Test accuracy: 0.8000                | Train accuracy: 0.8817<br>Test accuracy: 0.9002                              |
| No pre-trained embeddings<br>NN model type: Bidirectional LSTM NN (max sentence length 6)                          | Train accuracy: 0.8026<br>Test accuracy: 0.7538                | Train accuracy: 0.8944<br>Test accuracy: 0.8984                              |


### Results for sentence similarity task

Test examples have the following syntactical structure:

`s[s[n   (s\\n)[((s\\n)/(s\\n))   (s\\n)]] punc]`

`s[s[n[n] (s\\n)[((s\\n)/(s\\n))   (s\\n)]] punc]`

`s[s[n   (s\\n)[((s\\n)/n)   n[(n/n)   n]]] punc]`

`s[s[n   (s\\n)[((s\\n)/n)   n]] punc]`

`s[s[n   (s\\n)[((s\\n)/(s\\n))   (s\\n)[((s\\n)/(s\\n))   (s\\n)]]] punc]`

`s[s[n   (s\\n)[((s\\n)/(s\\n))[((s\\n)/(s\\n))   ((s\\n)\\(s\\n))] (s\\n)]] punc]`

`s[s[n   (s\\n)[((s\\n)/n)   n[n]]] punc]`

`s[s[n   (s\\n)[((s\\n)/n)   n[(n/n)   n[(n/n)   n]]]] punc]`

`s[s[n   (s\\n)[((s\\n)/(s\\n))   (s\\n)[((s\\n)/n)   n]]] punc]`

`s[s[n   (s\\n)[((s\\n)/(s\\n))[((s\\n)/(s\\n))   ((s\\n)\\(s\\n))] (s\\n)[((s\\n)/n)   n]]] punc]`

`s[s[n[n] (s\\n)[((s\\n)/(s\\n))   (s\\n)[((s\\n)/(s\\n))   (s\\n)]]] punc]`

`s[s[n[(n/n)   n] (s\\n)[((s\\n)/(s\\n))   (s\\n)]] punc]`

`s[s[n   (s\\n)[((s\\n)/(s\\n))   (s\\n)[((s\\n)/n)   n[(n/n)   n]]]] punc]`

`s[s[n[n] (s\\n)[((s\\n)/n)   n[(n/n)   n]]] punc]`

`s[s[n   (s\\n)[((s\\n)/(s\\n))   (s\\n)[((s\\n)/p)   p[(p/n)   n]]]] punc]`

`s[s[n   (s\\n)[((s\\n)/(s\\n))   (s\\n)[((s\\n)/n)   n[n]]]] punc]`

`s[s[n[n] (s\\n)[((s\\n)/(s\\n))[((s\\n)/(s\\n))   ((s\\n)\\(s\\n))] (s\\n)]] punc]`

`s[s[n   (s\\n)[((s\\n)/n)   n[n[(n/n)   n]]]] punc]`

`s[s[n   (s\\n)[((s\\n)/(s\\n))   (s\\n)[((s\\n)/(s\\n))   (s\\n)[((s\\n)/n)   n]]]] punc]`

`s[s[(s/(s\\n))[((s/(s\\n))/n)   n] (s\\n)[((s\\n)/n)   n]] punc]`

`s[s[n   (s\\n)[((s\\n)/n)[((s\\n)/n)   ((s\\n)\\(s\\n))] n[(n/n)   n]]] punc]`

`s[s[n[n] (s\\n)[((s\\n)/n)   n[(n/n)   n[(n/n)   n]]]] punc]`

`s[s[(s/(s\\n))[((s/(s\\n))/n)   n] (s\\n)[((s\\n)/n)   n[(n/n)   n]]] punc]`

`s[s[n   (s\\n)] punc]`

`s[s[n   (s\\n)[((s\\n)/p)   p[(p/n)   n[(n/n)   n]]]] punc]`

|                                                                                                                                                                                                                 | paired_similarity_filtered.tsv<br>(train: 26218, dev: 1456, test: 1456) |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| pre-trained LaBSE embeddings from <https://huggingface.co/sentence-transformers/LaBSE><br>using Euclidean distance between sentence embedding vectors<br>Sentence pair assumed similar if Euclidean score < 0.8 | Test accuracy: 0.9588                                                   |
| fastText embeddings trained on train set sentences<br>using Euclidean distance between sentence embedding vectors<br>Sentence pair assumed similar if Euclidean score < 0.5                                     | Test accuracy: 0.8915                                                   |