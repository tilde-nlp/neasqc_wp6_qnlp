import sys
import os
import json
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (Input, Dense, Activation, Conv1D,
                          Dropout, MaxPooling1D, Flatten)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelEncoder
import argparse

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
            "decay": 0,
            "epochs": 100,
            "epsilon": None,
            "amsgrad": False,
        }
        if "model" in kwargs and kwargs["model"] == "CNN": #defaults for CNN
            self.params["epochs"] = 30
            self.params["filterCounts"] = [300, 300]
            self.params["maxSentenceLen"] = 6
            self.params["dropout"] = 0.5

        self.params.update(kwargs)

    @staticmethod
    def createModel1(vectorSpaceSize, nClasses, **kwargs):
        model = Sequential()
        model.add(Dense(nClasses, input_dim=vectorSpaceSize, activation='softmax'))
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
                            decay=0, epsilon=None, amsgrad=False, **kwargs):
        opt = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2,
                    epsilon=epsilon, decay=decay, amsgrad=amsgrad)
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
        else:
            raise NotImplementedError(f"Unknown model type: {self.params['model']}")
        optimizer = NNClassifier.createAdamOptimizer(**self.params)
        self.model.compile(optimizer=optimizer,
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        if len(devX) == 0: #without early stopping
            return self.model.fit(trainX, trainY, epochs=self.params["epochs"], verbose=2,)
        else:
            callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
            return self.model.fit(trainX, trainY, epochs=self.params["epochs"], validation_data=(devX, devY), verbose=2, callbacks=[callback])

    def predict(self, testX):
        res = self.model.predict(testX)
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
        
def loadData(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def prepareXYSentence(data, classify_by_field="truth_value"):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @return: 2 numpy arrays corresponding to input data and labels
    '''
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

def prepareXYWords(data, maxLen, classify_by_field="truth_value"):
    '''Prepares the data as numpy arrays suitable for training.
    @param data: the data to process
    @param maxLen: maximum sentence length. Longer sentences are truncated,
        shorter sentences are padded with all-zero vectors
    @return: 2 numpy arrays corresponding to input data and labels
    '''
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
        classifier = NNClassifier(model='CNN',vectorSpaceSize=300)
        maxLen = 6
        trainX, trainY, testX, testY = prepareTrainTestXYWords(data, maxLen, args.field)
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
