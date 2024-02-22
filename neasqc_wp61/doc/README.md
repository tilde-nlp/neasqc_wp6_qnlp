# NLP

A module implementing processing of the datasets including text vectorization and classifier training.

## Workflow

The Workflow has 8 steps. Data preparation steps (Step 0 to Step 4) are common to the classical and the quantum algorithms.

To run each step use the corresponding bash script that is located in the directory *../data/data_preparation*. 

### Step 0 - data download

We use the following datasets from the kaggle.com. 

- *Reviews.csv* from the <https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/versions/1>
- *labelled_newscatcher_dataset.csv* from the <https://www.kaggle.com/datasets/kotartemiy/topic-labeled-news-dataset>
- *ag-news_train.csv* from the <https://www.kaggle.com/datasets/kk0105/ag-news>

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

`1_Filter6Parse.sh -i ./data/datasets/reviews.csv -i ./data/datasets/reviews_alltrees.tsv -d ',' -c 'Score' -t 'Summary' -g '0'`

`1_Filter6Parse.sh -i ./data/datasets/ag_news_csv_train.csv -o ./data/datasets/ag_news_alltrees.tsv -d ','  -g '7'`

`1_Filter6Parse.sh -i ./data/datasets/labelled_newscatcher_dataset.csv -o ./data/datasets/labelled_newscatcher_dataset_alltrees.tsv -d ';' -c 'topic' -t 'title' -g '-1'`

Files with syntactic tags `reviews_alltrees.tsv`, `labelled_newscatcher_dataset_alltrees.tsv`, `ag_news_alltrees.tsv` are included in *../data/datasets* folder.

Examples with the sentence similarity datasets:

`1_Filter6Parse.sh -i ./data/datasets/sorted_tatoeba_it.txt -d $'\t' -g '-1'`

`1_Filter6Parse.sh -i ./data/datasets/sorted_Europarl_fr.txt -d $'\t' -g '-1'`

`1_Filter6Parse.sh -i ./data/datasets/chatgpt_paraphrases.txt -d $'\t' -g '-1'`

### Step 2 - Selecting data with a certain syntactical structure.

To perform this step run the script *2_FilterSyntacticTrees.sh* passing the following parameters:

- -i \<input file\>               TAB separated 3-column file to filter
- -f \<syntactical trees file\>   File containing list of preferable syntactical tags. If this parameter is missing all tags with the structure  *s[...]* are selected.
- -c \<classes to ignore\>		  Optionally classes to ignore in dataset separated by ','

Example:

`2_FilterSyntacticTrees.sh -i ./data/datasets/reviews_alltrees.tsv -c '2,4'`

`2_FilterSyntacticTrees.sh -i ./data/datasets/labelled_newscatcher_dataset_alltrees.tsv -c 'SCIENCE'`

`2_FilterSyntacticTrees.sh -i ./data/datasets/ag_news_alltrees.tsv`

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

`3_SplitTrainTestDev.sh -i ./data/datasets/reviews_filtered.tsv`

`3_SplitTrainTestDev.sh -i ./data/datasets/ag_news_filtered.tsv`

`3_SplitTrainTestDev.sh -i ./data/datasets/labelled_newscatcher_dataset_filtered.tsv`

### Step 4 - Acquiring embedding vectors using chosen pre-trained embedding model.

We have experimented with 2 different pre-trained embedding models.

|   | Pre-trained model      | Embedding type | Vectors for unit | About model                                                                                                                                                                                        |
|---|------------------------|----------------|------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|   | *ember-v1*             | transformer    | sentence         | 1024-dimentional sentence transformer model. <br><https://huggingface.co/llmrails/ember-v1>                                                                                                         |
|   | *bert-base-uncased*    | bert           | sentence         | Case insensitive model pretrained on BookCorpus (consisting of 11,038 unpublished books) and English Wikipedia.<br><https://huggingface.co/bert-base-uncased>                                      |

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

Examples:

`4_GetEmbeddings.sh -i ./data/datasets/labelled_newscatcher_dataset_filtered_test.tsv -o ./data/datasets/labelled_newscatcher_dataset_filtered_test_ember.json -c '3' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/labelled_newscatcher_dataset_filtered_dev.tsv -o ./data/datasets/labelled_newscatcher_dataset_filtered_dev_ember.json -c '3' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/labelled_newscatcher_dataset_filtered_train.tsv -o ./data/datasets/labelled_newscatcher_dataset_filtered_train_ember.json -c '3' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/reviews_filtered_test.tsv -o ./data/datasets/reviews_filtered_test_ember.json -c '3' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/reviews_filtered_dev.tsv -o ./data/datasets/reviews_filtered_dev_ember.json -c '3' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/reviews_filtered_train.tsv -o ./data/datasets/reviews_filtered_train_ember.json -c '3' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/ag_news_filtered_test.tsv -o ./data/datasets/ag_news_filtered_test_ember.json -c '3' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/ag_news_filtered_dev.tsv -o ./data/datasets/ag_news_filtered_dev_ember.json -c '3' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/ag_news_filtered_train.tsv -o ./data/datasets/ag_news_filtered_train_ember.json -c '3' -m 'llmrails/ember-v1' -t 'transformer' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/labelled_newscatcher_dataset_filtered_test.tsv -o ./data/datasets/labelled_newscatcher_dataset_filtered_test_bert.json -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/labelled_newscatcher_dataset_filtered_dev.tsv -o ./data/datasets/labelled_newscatcher_dataset_filtered_dev_bert.json -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/labelled_newscatcher_dataset_filtered_train.tsv -o ./data/datasets/labelled_newscatcher_dataset_filtered_train_bert.json -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/reviews_filtered_test.tsv -o ./data/datasets/reviews_filtered_test_bert.json -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/reviews_filtered_dev.tsv -o ./data/datasets/reviews_filtered_dev_bert.json -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/reviews_filtered_train.tsv -o ./data/datasets/reviews_filtered_train_bert.json -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/ag_news_filtered_test.tsv -o ./data/datasets/ag_news_filtered_test_bert.json -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/ag_news_filtered_dev.tsv -o ./data/datasets/ag_news_filtered_dev_bert.json -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '1'`

`4_GetEmbeddings.sh -i ./data/datasets/ag_news_filtered_train.tsv -o ./data/datasets/ag_news_filtered_train_bert.json -c '3' -m 'bert-base-uncased' -t 'bert' -e 'sentence' -g '1'`


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
	
Each model is trained 30 times (runs).

Examples:

`bash ./5_TrainNNModel.sh -t ./data/datasets/labelled_newscatcher_dataset_filtered_train_ember.json -d ./data/datasets/labelled_newscatcher_dataset_filtered_dev_ember.json -f 'class' -e 'sentence' -m ./models/classical/labelled_newscatcher_dataset_ember -g '-1'`

`bash ./5_TrainNNModel.sh -t ./data/datasets/labelled_newscatcher_dataset_filtered_train_bert.json -d ./data/datasets/labelled_newscatcher_dataset_filtered_dev_bert.json -f 'class' -e 'sentence' -m ./models/classical/labelled_newscatcher_dataset_bert -g '-1'`

`bash ./5_TrainNNModel.sh -t ./data/datasets/reviews_filtered_train_ember.json -d ./data/datasets/reviews_filtered_dev_ember.json -f 'class' -e 'sentence' -m ./models/classical/reviews_ember -g '-1'`

`bash ./5_TrainNNModel.sh -t ./data/datasets/reviews_filtered_train_bert.json -d ./data/datasets/reviews_filtered_dev_bert.json -f 'class' -e 'sentence' -m ./models/classical/reviews_bert -g '-1'`

`bash ./5_TrainNNModel.sh -t ./data/datasets/ag_news_filtered_train_ember.json -d ./data/datasets/ag_news_filtered_dev_ember.json -f 'class' -e 'sentence' -m ./models/classical/ag_news_ember -g '-1'`

`bash ./5_TrainNNModel.sh -t ./data/datasets/ag_news_filtered_train_bert.json -d ./data/datasets/ag_news_filtered_dev_bert.json -f 'class' -e 'sentence' -m ./models/classical/ag_news_bert -g '-1'`



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

Examples for the classification task, classification is performed with each of 30 models (trained in 30 runs):

`bash ./6_ClassifyWithNNModel.sh -i ./data/datasets/labelled_newscatcher_dataset_filtered_test_ember.json -o ./benchmarking/results/raw/labelled_newscatcher_dataset_ember.txt -e 'sentence' -m ./models/classical/labelled_newscatcher_dataset_ember -g '-1'`

`bash ./6_ClassifyWithNNModel.sh -i ./data/datasets/labelled_newscatcher_dataset_filtered_test_bert.json -o ./benchmarking/results/raw/labelled_newscatcher_dataset_bert.txt -e 'sentence' -m ./models/classical/labelled_newscatcher_dataset_bert -g '-1'`

`bash ./6_ClassifyWithNNModel.sh -i ./data/datasets/reviews_filtered_test_ember.json -o ./benchmarking/results/raw/reviews_ember.txt -e 'sentence' -m ./models/classical/reviews_ember -g '-1'`

`bash ./6_ClassifyWithNNModel.sh -i ./data/datasets/reviews_filtered_test_bert.json -o ./benchmarking/results/raw/reviews_bert.txt -e 'sentence' -m ./models/classical/reviews_bert -g '-1'`

`bash ./6_ClassifyWithNNModel.sh -i ./data/datasets/ag_news_filtered_test_ember.json -o ./benchmarking/results/raw/ag_news_ember.txt -e 'sentence' -m ./models/classical/ag_news_ember -g '-1'`

`bash ./6_ClassifyWithNNModel.sh -i ./data/datasets/ag_news_filtered_test_bert.json -o ./benchmarking/results/raw/ag_news_bert.txt -e 'sentence' -m ./models/classical/ag_news_bert -g '-1'`

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

Evaluation examples, evaluation is performed for each set of results acquired with 30 models (trained in 30 runs):

`bash ./7_EvaluateResults.sh -e ./data/datasets/labelled_newscatcher_dataset_filtered_test.tsv -c ./benchmarking/results/raw/labelled_newscatcher_dataset_ember.txt -o ./benchmarking/results/labelled_newscatcher_dataset_ember.txt`

`bash ./7_EvaluateResults.sh -e ./data/datasets/labelled_newscatcher_dataset_filtered_test.tsv -c ./benchmarking/results/raw/labelled_newscatcher_dataset_bert.txt -o ./benchmarking/results/labelled_newscatcher_dataset_bert.txt`

`bash ./7_EvaluateResults.sh -e ./data/datasets/reviews_filtered_test.tsv -c ./benchmarking/results/raw/reviews_ember.txt -o ./benchmarking/results/reviews_ember.txt`

`bash ./7_EvaluateResults.sh -e ./data/datasets/reviews_filtered_test.tsv -c ./benchmarking/results/raw/reviews_bert.txt -o ./benchmarking/results/reviews_bert.txt`

`bash ./7_EvaluateResults.sh -e ./data/datasets/ag_news_filtered_test.tsv -c ./benchmarking/results/raw/ag_news_ember.txt -o ./benchmarking/results/ag_news_ember.txt`

`bash ./7_EvaluateResults.sh -e ./data/datasets/ag_news_filtered_test.tsv -c ./benchmarking/results/raw/ag_news_bert.txt -o ./benchmarking/results/ag_news_bert.txt`

## Results

### Results for classification task (data with sentences)

The syntactical structure of the test examples starts with *s[*. The examples contain no more that 20 tokens. Accuracies are given as average value of 30 runs.

|                                                                                                              | reviews<br>(train: 15972, dev: 1995, test: 1995)<br>3 classes            | ag_news<br>(train: 57412, dev: 7176, test: 7176)<br>4 classes            | labelled_newscatcher_dataset<br>(train: 43225, dev: 5397, test: 5397)<br>7 classes |
|--------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|------------------------------------------------------------------------------------|
| 1024-dimentional transformer sentence emb.: *ember-v1*<br>NN model type: Shallow feedforward neural network  | Train accuracy: 0.7958<br>Dev. accuracy: 0.7846<br>Test accuracy: 0.7607 | Train accuracy: 0.9086<br>Dev. accuracy: 0.8965<br>Test accuracy: 0.8900 | Train accuracy: 0.8227<br>Dev. accuracy: 0.8067<br>Test accuracy: 0.8037           |
| 768-dimentional BERT sentence emb.: *bert-base-uncased*<br>NN model type: Shallow feedforward neural network | Train accuracy: 0.7066<br>Dev. accuracy: 0.7105<br>Test accuracy: 0.6749 | Train accuracy: 0.8407<br>Dev. accuracy: 0.8317<br>Test accuracy: 0.8265 | Train accuracy: 0.7639<br>Dev. accuracy: 0.7512<br>Test accuracy: 0.7532           |


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