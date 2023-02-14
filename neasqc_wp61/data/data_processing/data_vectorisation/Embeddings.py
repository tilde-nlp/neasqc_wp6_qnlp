import sys
import json
from keras.utils import pad_sequences
import numpy as np
import fasttext
import fasttext.util
from sentence_transformers import SentenceTransformer
import torch
from transformers import BertTokenizer, BertModel

class Embeddings:
    def __init__(self, path, embtype):
        self.embtype=embtype
        self.model=None
        self.tokenizer=None       
        try:
            if (embtype=='fasttext'):
                if not os.path.exists(path):
                    fasttext.util.download_model('en', if_exists='ignore')              
                self.model = fasttext.load_model(path)
            elif (embtype == 'bert'):
                self.model=BertModel.from_pretrained(path, output_hidden_states = True, )
                self.tokenizer=BertTokenizer.from_pretrained(path)
            elif (embtype == 'transformer'):
                self.model=SentenceTransformer(path)
                
            print(path + ' loaded!')
        except:
            print('Failed to load ' + path)
        pass

 
    def getEmbeddingVector(self, sentence):
        embvec=[]
        try:
            if self.model:
                if (self.embtype=='fasttext'):
                    words = sentence.split()
                    if (len(words)>0):
                        embvec = [{'word': w, 'vector': self.model.get_word_vector(w).tolist()} for w in words]
                elif (self.embtype=='bert'):
                    input_tokens = []
                    encodedsent = self.tokenizer.tokenize("[CLS] " + sentence + " [SEP]")
                    input_tokens.append(self.tokenizer.convert_tokens_to_ids(encodedsent))
                    input_ids = pad_sequences(input_tokens, maxlen=50, dtype="long", value=0, truncating="post", padding="post")
                    input_masks = []
                    input_masks.append([int(token_id > 0) for token_id in input_ids[0]])
                    input_ids = torch.tensor(input_ids)
                    attention_mask = torch.tensor(input_masks)
                    
                    with torch.no_grad():
                        last_hidden_states  = self.model(input_ids, attention_mask=attention_mask)

 #The BERT uses [CLS] token to represent sentence information - tensor position [0].
 #See https://github.com/VincentK1991/BERT_summarization_1/blob/master/notebook/Primer_to_BERT_extractive_summarization_March_25_2020.ipynb 
                      
                    embvec = [last_hidden_states[0][0,0,:].detach().numpy().tolist()]
                elif (self.embtype=='transformer'):
                    res = self.model.encode([sentence])
                    embvec = [res.tolist()[0]]
        except:
            print('Failed to get embeddings for sentence: ' + sentence)
        return embvec