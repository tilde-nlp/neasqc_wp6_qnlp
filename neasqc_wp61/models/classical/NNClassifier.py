import sys
import os
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Input, Dense, Activation, Conv1D,
                          Dropout, MaxPooling1D, Flatten, LSTM, Embedding, Bidirectional)
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelEncoder
import argparse
import matplotlib.pyplot as plt

def ts():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

class NNClassifier:
    """
    A class implementing neural network classifiers.
    
    Currently, a shallow feedforward neural network and a convolutional network are implemented.
    """
    def __init__(self, **kwargs):
        # default values
        self.params = {
            "model": "FFNN",
            "vectorSpaceSize": 768,
            "nClasses": 2,
            "learning_rate": 0.001,
            "beta_1": 0.9,
            "beta_2": 0.999,
            "epochs": 100,
            "epsilon": None,
            "amsgrad": False,
            "gpu": -1
        }
        if "model" in kwargs:
            if kwargs["model"] == "CNN": #defaults for CNN
                self.params["epochs"] = 30
                self.params["filterCounts"] = [self.params["vectorSpaceSize"], self.params["vectorSpaceSize"]]
                self.params["maxSentenceLen"] = 6
                self.params["dropout"] = 0.5
            elif kwargs["model"] == "LSTM":
                self.params["epochs"] = 100
                self.params["vectorSpaceSize"] = 128
                self.params["vocab_size"] = 5000
                self.params["maxSentenceLen"] = 6
                self.params["dropout"] = 0.5
                
        self.params.update(kwargs)
        gpudevices = tf.config.list_physical_devices('GPU')
        cpudevices = tf.config.list_physical_devices('CPU')

        if (self.params["gpu"] == -1 and len(cpudevices) > 0): # want CPU, has CPU
            print("Training/running model on CPU.")
            tf.config.set_visible_devices([], 'GPU')
        elif (self.params["gpu"] > -1 and len(gpudevices) > self.params["gpu"]): # want GPU, has GPU with that number
            tf.config.set_visible_devices(gpudevices[self.params["gpu"]], 'GPU')
            print("Training/running model on GPU: ", gpudevices[self.params["gpu"]])

    @staticmethod
    def createModel1(vectorSpaceSize, nClasses, **kwargs):
        model = Sequential()
        model.add(Dense(nClasses, input_dim=vectorSpaceSize, activation='softmax'))
        return model

    @staticmethod
    def createModel1LSTM(vectorSpaceSize, nClasses, vocab_size, dropout, **kwargs):
        model = Sequential()
        model.add(Embedding(vocab_size, vectorSpaceSize))
        model.add(Dropout(dropout))
        model.add(Bidirectional(LSTM(vectorSpaceSize)))
        model.add(Dense(nClasses, activation='softmax'))
        return model
        
    @staticmethod
    def createModelCNN(vectorSpaceSize, maxSentenceLen, nClasses, filterCounts, dropout, **kwargs):
        inp = Input(shape=(maxSentenceLen, vectorSpaceSize))
        filterLayer = []
        for ws, filters in enumerate(filterCounts, start=1):
            if filters > 0:
                conv = Conv1D(filters=filters,
                                kernel_size=ws,
                                activation='relu'
                                )(inp)
                conv = MaxPooling1D(pool_size=maxSentenceLen - ws + 1)(conv)
                filterLayer.append(conv)
        if len(filterLayer) > 1:
            merged = tf.keras.layers.concatenate(filterLayer)
        else:
            merged = filterLayer[0]
        merged = Flatten()(merged)
        if dropout>0:
            merged = Dropout(rate=dropout)(merged)
        out = Dense(units=nClasses, activation='softmax')(merged)
        model = Model(inp, out)
        return model


    @staticmethod
    def createAdamOptimizer(learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                            epsilon=None, amsgrad=False, **kwargs):
        opt = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                    epsilon=epsilon, amsgrad=amsgrad)
        return opt


    def train(self, trainX, trainY, devX=[], devY=[]):
        '''Trains the model.
        @param trainX: training data
        @param trainY: training targets (labels)
        @return: training history data as returned by Keras
        '''
        self.params["nClasses"] = len(trainY[0])
        if self.params["model"] == "CNN":
            self.model = NNClassifier.createModelCNN(**self.params)
        elif self.params["model"] == "FFNN":
            self.model = NNClassifier.createModel1(**self.params)
        elif self.params["model"] == "LSTM":
            self.model = NNClassifier.createModel1LSTM(**self.params)
        else:
            raise NotImplementedError(f"Unknown model type: {self.params['model']}")
        optimizer = NNClassifier.createAdamOptimizer(**self.params)
        self.model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])

        if len(devX) == 0: #without early stopping
            return self.model.fit(trainX, trainY, epochs=self.params["epochs"], verbose=2,)
        else:
            print("Training with early stopping.")       
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            history = self.model.fit(trainX, trainY, epochs=self.params["epochs"], validation_data=(devX, devY), verbose=2, callbacks=[callback])
            plot_graphs(history, "accuracy")
            plot_graphs(history, "loss")
            return history

    def predict(self, testX):
        res = self.model.predict(testX, verbose=0)
        topPrediction = np.argmax(res, axis=1)
        return topPrediction

    def save(self, folder):
        """Saves model."""
        self.model.save(folder)
        return 

    def load(self, folder):
        """Loads model."""
        try:
            self.model = load_model(folder)
        except:
            print(f"Failed to load model from {folder}")
            return False
        return True
 
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epoch Count")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
  
def loadData(file):
    dataset = []
    
    if '.json' in file:
        with open(file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    elif '.tsv' in file:
        rowlist = open(file,encoding="utf-8").read().splitlines()
        for line in rowlist:
            d = {}
            linecols = line.split('\t')
            d["class"]=linecols[0].rstrip()
            d["sentence"]=linecols[1].rstrip()
            dataset.append(d)
            
    return dataset

def prepareXYSentence(data, classify_by_field="truth_value",idxdict={}):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @return: 2 numpy arrays corresponding to input data and labels
    '''
    if len(idxdict) == 0:
        idxdict = prepareClassValueDict(data, classify_by_field)
    
    arrX = []
    arrY = []
    for ex in data:
        arrX.append(ex["sentence_vectorized"][0])
        arrY.append(idxdict[ex[classify_by_field]])
    X = np.array(arrX)
    labelencoder = LabelEncoder()
    Y = tf.keras.utils.to_categorical(arrY, num_classes=len(idxdict))
    return X, Y

def prepareXSentence(data):
    '''Prepares the data as numpy array suitable for testing.
    @param data: the data to process
    @return: numpy array corresponding to input data
    '''
  
    arrX = []
    for ex in data:
        arrX.append(ex["sentence_vectorized"][0])
    X = np.array(arrX)
    return X
    
def prepareTrainTestXYSentence(data, classify_by_field="truth_value"):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @return: four numpy arrays corresponding to training and test input data and labels
    '''
    dX = []
    dY = []
    for i, dd in enumerate(["train_data", "test_data"]):
        arrX = []
        arrY = []
        for ex in data[dd]:
            arrX.append(ex["sentence_vectorized"][0])
            arrY.append(ex[classify_by_field])
        X = np.array(arrX)
        labelencoder = LabelEncoder()
        Y = tf.keras.utils.to_categorical(labelencoder.fit_transform(arrY), None)
        dX.append(X)
        dY.append(Y)
    return dX[0], dY[0], dX[1], dY[1]

def prepareClassValueDict(dataset, classify_by_field):
    idxdict = {}
    idx=0
    for ex in dataset: #training set has all class values
        if ex[classify_by_field] not in idxdict.keys():
            idxdict[ex[classify_by_field]] = idx
            idx = idx + 1
    return idxdict
 
def prepareTokenDict(dataset):
    sentences = []
    for ex in dataset:
        sentences.append(ex["sentence"])
    tokenizer = Tokenizer(num_words = 5000, oov_token='<oov>')
    tokenizer.fit_on_texts(sentences)
    return tokenizer.word_index
    
def prepareTrainTestDevXYSentence(data, classify_by_field="truth_value"):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @return: 6 numpy arrays corresponding to training, test and dev input data and labels
    '''
    idxdict = prepareClassValueDict(data["train_data"], classify_by_field)
    
    dX = []
    dY = []
    for i, dd in enumerate(["train_data", "test_data", "dev_data"]):
        arrX = []
        arrY = []
        for ex in data[dd]:
            arrX.append(ex["sentence_vectorized"][0])
            arrY.append(idxdict[ex[classify_by_field]])
        X = np.array(arrX)
        Y = tf.keras.utils.to_categorical(arrY, num_classes=len(idxdict))
        dX.append(X)
        dY.append(Y)
    return dX[0], dY[0], dX[1], dY[1], dX[2], dY[2]
    
def appendZeroVector(x, N, dim):
    if len(x) > N:
        return x[:N]
    return x + [[0] * dim] * (N - len(x))

def prepareXYWords(data, maxLen, classify_by_field="truth_value",idxdict={}):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @param maxLen: maximum sentence length. Longer sentences are truncated,
        shorter sentences are padded with all-zero vectors
    @return: 2 numpy arrays corresponding to input data and labels
    '''
    if len(idxdict) == 0:
        idxdict = prepareClassValueDict(data, classify_by_field)

    arrX = []
    arrY = []
    for ex in data:
        sentenceVectors = []
        for w in ex["sentence_vectorized"]:
            sentenceVectors.append(w["vector"])
            dim = len(w["vector"])
        arrX.append(sentenceVectors)
        arrY.append(idxdict[ex[classify_by_field]])
    arrX = [appendZeroVector(sv, maxLen, dim) for sv in arrX]
    X = np.array(arrX)
    Y = tf.keras.utils.to_categorical(arrY, num_classes=len(idxdict))

    return X, Y

def prepareXYWordsNoEmbedd(data, tokdict, maxLen, classify_by_field="truth_value",idxdict={}):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @param maxLen: maximum sentence length. Longer sentences are truncated,
        shorter sentences are padded with zeros
    @return: 2 numpy arrays corresponding to input data and labels
    '''
    if len(idxdict) == 0:
        idxdict = prepareClassValueDict(data, classify_by_field)

    arrX = []
    arrY = []
    for ex in data:
        arrX.append(ex["sentence"])
        arrY.append(idxdict[ex[classify_by_field]])

    tokenizer = Tokenizer(num_words = 5000, oov_token='<oov>')
    tokenizer.word_index = tokdict
    train_sequences = tokenizer.texts_to_sequences(arrX)

    arrX = pad_sequences(train_sequences, maxlen=maxLen, padding='post', truncating='post')

    X = np.array(arrX)
    Y = tf.keras.utils.to_categorical(arrY, num_classes=len(idxdict))

    return X, Y

def prepareXWordsNoEmbedd(data, tokdict, maxLen=6):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @param maxLen: maximum sentence length. Longer sentences are truncated,
        shorter sentences are padded with zeros
    @return: numpy array corresponding to input data
    '''
    arrX = []

    for ex in data:
        arrX.append(ex["sentence"])

    tokenizer = Tokenizer(num_words = 5000, oov_token='<oov>')
    tokenizer.word_index = tokdict
    train_sequences = tokenizer.texts_to_sequences(arrX)

    arrX = pad_sequences(train_sequences, maxlen=maxLen, padding='post', truncating='post')

    X = np.array(arrX)

    return X
    
def prepareXWords(data, maxLen=6):
    '''Prepares the data as numpy array suitable for testing.
    @param data: the data to process
    @param maxLen: maximum sentence length. Longer sentences are truncated,
        shorter sentences are padded with all-zero vectors
    @return: numpy array corresponding to input data
    '''
    arrX = []
    for ex in data:
        sentenceVectors = []
        for w in ex["sentence_vectorized"]:
            sentenceVectors.append(w["vector"])
            dim = len(w["vector"])
        arrX.append(sentenceVectors)
    arrX = [appendZeroVector(sv, maxLen, dim) for sv in arrX]
    X = np.array(arrX)

    return X
    
def prepareTrainTestXYWords(data, maxLen, classify_by_field="truth_value"):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @param maxLen: maximum sentence length. Longer sentences are truncated,
        shorter sentences are padded with all-zero vectors
    @return: four numpy arrays corresponding to training and test input data and labels
    '''
    dX = []
    dY = []
    for i, dd in enumerate(["train_data", "test_data"]):
        arrX = []
        arrY = []
        for ex in data[dd]:
            sentenceVectors = []
            for w in ex["sentence_vectorized"]:
                sentenceVectors.append(w["vector"])
                dim = len(w["vector"])
            arrX.append(sentenceVectors)
            arrY.append(ex[classify_by_field])
        arrX = [appendZeroVector(sv, maxLen, dim) for sv in arrX]
        X = np.array(arrX)
        labelencoder = LabelEncoder()
        Y = tf.keras.utils.to_categorical(labelencoder.fit_transform(arrY), None)
        dX.append(X)
        dY.append(Y)
    return dX[0], dY[0], dX[1], dY[1]

def prepareTrainTestDevXYWords(data, maxLen, classify_by_field="truth_value"):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @param maxLen: maximum sentence length. Longer sentences are truncated,
        shorter sentences are padded with all-zero vectors
    @return: four numpy arrays corresponding to training, test and dev input data and labels
    '''
    idxdict = prepareClassValueDict(data["train_data"], classify_by_field)
    dX = []
    dY = []
    for i, dd in enumerate(["train_data", "test_data", "dev_data"]):
        arrX = []
        arrY = []
        for ex in data[dd]:
            sentenceVectors = []
            print(ex["sentence"])
            for w in ex["sentence_vectorized"]:
                sentenceVectors.append(w["vector"])
                dim = len(w["vector"])
            arrX.append(sentenceVectors)
            arrY.append(idxdict[ex[classify_by_field]])
        arrX = [appendZeroVector(sv, maxLen, dim) for sv in arrX]
        X = np.array(arrX)
        Y = tf.keras.utils.to_categorical(arrY, num_classes=len(idxdict))
        dX.append(X)
        dY.append(Y)
    return dX[0], dY[0], dX[1], dY[1], dX[2], dY[2]
  
def evaluate(predictions, testY):
    """Evaluates the accuracy of the predictions."""
    return np.sum(predictions == np.argmax(testY, axis=1))/len(testY)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help = "Json data file for classifier training (with embeddings)")
    parser.add_argument("-f", "--field", help = "Classify by field")
    parser.add_argument("-t", "--type", help = "Embedding type: sentence or word")
    args = parser.parse_args()
    print(args)
    data = loadData(args.input)
	
    if args.type == "word":
        maxLen = 6
        trainX, trainY, testX, testY = prepareTrainTestXYWords(data, maxLen, args.field)
        vecsize = len(trainX[0])
        print(F"Vec size: {vecsize}")
        classifier = NNClassifier(model='CNN',vectorSpaceSize=vecsize)
    elif args.type == "sentence":
        classifier = NNClassifier()
        trainX, trainY, testX, testY = prepareTrainTestXYSentence(data, args.field)
    else:
        print("Invalid embedding type. it must be 'word' or 'sentence'.")
        sys.exit(0)
	
    history = classifier.train(trainX, trainY)

    res = classifier.predict(testX)
    score = evaluate(res, testY)
    print(score)

    resFileName = datetime.datetime.now().strftime('results_%Y-%m-%d_%H-%M-%S.json')
    with open(resFileName , "w", encoding="utf-8") as f:
        json.dump({
            "train accuracy": str(history.history['accuracy'][-1]),
            "test accuracy": str(score),
        }, f, indent=2)


if __name__ == "__main__":
    sys.exit(int(main() or 0))
