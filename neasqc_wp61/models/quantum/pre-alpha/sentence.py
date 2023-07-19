import spacy
import numpy as np
import math
import random


class Sentence:

    def __init__(self, sentence, dataset=None, dictionary=None, label=None, stype=None):
        if not dataset:
            if type(sentence) != str:
                raise Exception('If no dataset is provided, sentence type must be a string')

            self.nlp = spacy.load('en_core_web_lg')
            self.sentence = self.nlp(sentence)
            self.dictionary = dictionary
            self.dictionary.addwords(self)
            self.qubitsarray = []
            self.categoriesarray = []
            self.catqubits = []
            self.stype = stype
            self.label = label
            nwords = 0
            for token in self.sentence:
                word = token.text.strip()
                wordcategories = self.dictionary.dictionary[word].category
                self.categoriesarray.append(wordcategories)

                if nwords < 1:
                    qubitlist = list(np.arange(dictionary.dictionary[word].nqubits))
                elif nwords == 1:
                    lastqubit = qubitlist[-1]
                    qubitlist = list(1 + lastqubit + np.arange(dictionary.dictionary[word].nqubits))
                elif nwords > 1:
                    lastqubit = qubitlist[-1]
                    qubitlist = list(1 + lastqubit + np.arange(dictionary.dictionary[word].nqubits))
                self.qubitsarray.append(qubitlist)
                nwords += 1

                wordqubits = []
                for category in wordcategories:
                    if category in ['nl', 'nr', 'n', 'nrr', 'nll']:
                        wordqubits.append(dictionary.qn)
                    elif category in ['s', 'sl', 'sr']:
                        wordqubits.append(dictionary.qs)
                    else:
                        print('category not found')
                self.catqubits.append(wordqubits)

        elif dataset:
            self.dictionary = dict()
            self.sentence = sentence['sentence']
            self.sentencestructure = sentence['sentence_type']
            self.qubitsarray = []
            self.categoriesarray = []
            self.catqubits = []

            if not sentence['truth_value']:
                self.label = 0
            elif sentence['truth_value']:
                self.label = 1
            self.stype = sentence['sentence_type_code']
            nwords = 0
            for word, cat in zip(sentence['sentence'].split(' '), sentence['sentence_type'].split('-')):
                wordcategories = dictionary.dictionary[word].category[cat]
                self.categoriesarray.append(wordcategories)

                if nwords < 1:
                    qubitlist = list(np.arange(dictionary.dictionary[word].nqubits[cat]))
                elif nwords == 1:
                    lastqubit = qubitlist[-1]
                    qubitlist = list(1 + lastqubit + np.arange(dictionary.dictionary[word].nqubits[cat]))
                elif nwords > 1:
                    lastqubit = qubitlist[-1]
                    qubitlist = list(1 + lastqubit + np.arange(dictionary.dictionary[word].nqubits[cat]))
                self.qubitsarray.append(qubitlist)
                nwords += 1

                wordqubits = []
                for category in wordcategories:
                    if category in ['nl', 'nr', 'n', 'nrr', 'nll']:
                        wordqubits.append(dictionary.qn)
                    elif category in ['s', 'sl', 'sr']:
                        wordqubits.append(dictionary.qs)
                    else:
                        print('category not found')
                self.catqubits.append(wordqubits)

    def getstypecontractions(self):
        stypedict = {
            0: {'words': [[0, 1], [1, 2]], 'categories': [[['n', 'nr']], [['nl', 'n']]]},  # Noun-TransitiveVerb-Noun
            1: {'words': [[0, 2], [1, 2], [2, 3]],
                'categories': [[['n', 'nr']], [['nr', 'nrr'], ['s', 'sr']], [['nl', 'n']]]},
            # Noun-IntransitiveVerb-Preposition-Noun
            2: {'words': [[0, 1], [0, 2], [2, 3], [3, 4]],
                'categories': [[['n', 'nr']], [['nl', 'n']], [['n', 'nr']], [['n', 'nr']]]},  # Adj-Noun-TVerb-Adj-Noun
            3: {'words': [[0, 1]], 'categories': [[['n', 'nr']]]}
        }
        return stypedict[self.stype]

    def getqbitcontractions(self):
        contractions = []
        squbit = list(range(self.qubitsarray[-1][-1] + 1))
        styperelations = self.getstypecontractions()
        words = styperelations['words']
        cats = styperelations['categories']

        for i, wordpair in enumerate(words):
            for catpair in cats[i]:
                cup1 = self.searchqubit(wordpair[0], catpair[0])
                for qbit in cup1:
                    squbit.remove(qbit)
                cup2 = self.searchqubit(wordpair[1], catpair[1])
                for qbit in cup2:
                    squbit.remove(qbit)
                contraction = [cup1, cup2]
                contractions.append(contraction)
        self.sentencequbit = squbit[0]
        self.contractions = contractions

    def setwordparameters(self, myword, randompar=True, parameterization='Simple', layers=1, params=None, wordposition=None):
        gates = []
        wordposition = self.dictionary.dictionary[myword].pos
        wordqubits = self.qubitsarray[wordposition]
        if randompar:
            if parameterization == 'Simple':  # Two rotations + C-NOT gate per layer
                for layer in range(layers):
                    for qubit in wordqubits:
                        ry = 2 * math.pi * random.random()
                        rz = 2 * math.pi * random.random()
                        gates.append(dict({'Gate': 'RY', 'Angle': ry, 'Qubit': qubit}))
                        gates.append(dict({'Gate': 'RZ', 'Angle': rz, 'Qubit': qubit}))

                    for qubit in wordqubits[:-1]:
                        gates.append(dict({'Gate': 'CX', 'Qubit': [qubit, qubit + 1]}))
                self.dictionary.dictionary[myword].gateset = gates

        elif not randompar:
            wordparams = params[wordposition]
            paramid = 0
            if parameterization == 'Simple':  # Two rotations + C-NOT gate per layer
                for layer in range(layers):
                    for qubit in wordqubits:
                        gates.append(dict({'Gate': 'RY', 'Angle': wordparams[paramid], 'Qubit': qubit}))
                        paramid += 1
                        gates.append(dict({'Gate': 'RZ', 'Angle': wordparams[paramid], 'Qubit': qubit}))
                        paramid += 1

                    for qubit in wordqubits[:-1]:
                        gates.append(dict({'Gate': 'CX', 'Qubit': [qubit, qubit + 1]}))
                self.dictionary.dictionary[myword].gateset = gates

    def setsentenceparameters(self, randompar=True, parameterization='Simple', layers=1, params=None):
        for word, qword in self.dictionary.dictionary.items():
            self.setwordparameters(word, randompar, parameterization, layers, params)



    def setparamsfrommodel(self, mydict, ansatz = 'Simple', layers = 1):
        sentenceparams = []
        iword = 0
        for word,cat in zip(self.sentence.split(' '), self.sentencestructure.split('-')):
            wordparams = []
            for gate in mydict.dictionary[word].gateset[cat]:
                if (gate['Gate'] == 'RY') or (gate['Gate'] == 'RZ'):
                    wordparams.append(gate['Angle'])
            sentenceparams.append(wordparams)
            self.setwordparametersfrommodel(myword=word,
                                            wordparams=wordparams,
                                            wordposition=iword,
                                            parameterization='Simple',
                                            layers=1,
                                            )
            iword +=1


    def setwordparametersfrommodel(self, myword, wordparams, wordposition, parameterization='Simple', layers=1):
        gates = []
        wordqubits = self.qubitsarray[wordposition]
        paramid = 0
        if parameterization == 'Simple':  # Two rotations + C-NOT gate per layer
            for layer in range(layers):
                for qubit in wordqubits:
                    gates.append(dict({'Gate': 'RY', 'Angle': wordparams[paramid], 'Qubit': qubit}))
                    paramid += 1
                    gates.append(dict({'Gate': 'RZ', 'Angle': wordparams[paramid], 'Qubit': qubit}))
                    paramid += 1

                for qubit in wordqubits[:-1]:
                    gates.append(dict({'Gate': 'CX', 'Qubit': [qubit, qubit + 1]}))
            self.dictionary[myword] = dict()
            self.dictionary[myword]['gateset'] = gates




    def searchqubit(self, word, cat):
        for icat, category in enumerate(self.categoriesarray[word]):
            if category == cat:
                firstqubit = 0
                for prevword in self.catqubits[:word]:
                    firstqubit += sum(prevword)
                firstqubit += sum(self.catqubits[word][:icat])
                nqubits = self.catqubits[word][icat]
        return np.arange(firstqubit, firstqubit + nqubits)

    def getparameters(self, dataset=False):
        sentenceparams = []
        if not dataset:
            for word in self.dictionary.dictionary.keys():
                wordparams = []
                for gate in self.dictionary.dictionary[word].gateset:
                    if (gate['Gate'] == 'RY') or (gate['Gate'] == 'RZ'):
                        wordparams.append(gate['Angle'])
                sentenceparams.append(wordparams)
            return sentenceparams

        elif dataset:
            for word in self.dictionary.keys():
                wordparams = []
                for gate in self.dictionary[word]['gateset']:
                    if (gate['Gate'] == 'RY') or (gate['Gate'] == 'RZ'):
                        wordparams.append(gate['Angle'])
                sentenceparams.append(wordparams)
            return sentenceparams



    def setsentenceparamsfromlist(self, params):
        iword=0
        for word in self.dictionary.keys():
            wordparams = params[iword]
            self.setwordparametersfromlist(word=word,
                                           wordparams=wordparams,
                                           wordposition=iword
                                           )
            iword += 1



    def setwordparametersfromlist(self, word, wordparams, wordposition, parameterization='Simple', layers=1):
            wordqubits = self.qubitsarray[wordposition]
            paramid = 0
            gates = []
            if parameterization == 'Simple':  # Two rotations + C-NOT gate per layer
                for layer in range(layers):
                    for qubit in wordqubits:
                        gates.append(dict({'Gate': 'RY', 'Angle': wordparams[paramid], 'Qubit': qubit}))
                        paramid += 1
                        gates.append(dict({'Gate': 'RZ', 'Angle': wordparams[paramid], 'Qubit': qubit}))
                        paramid += 1

                    for qubit in wordqubits[:-1]:
                        gates.append(dict({'Gate': 'CX', 'Qubit': [qubit, qubit + 1]}))
                self.dictionary[word] = dict()
                self.dictionary[word]['gateset'] = gates
