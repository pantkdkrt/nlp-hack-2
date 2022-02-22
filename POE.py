import pandas as pd
import numpy as np
import os
import math
import pickle
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

############# input ###################
new_word = ['เมื่อ','ตอนเช้า','กกก','ยิง','ขขข']
#######################################

model = load_model('model_POS.h5')

with open('word2index.p', "rb") as fh:
  word2index = pickle.load(fh)

with open('tag2index.p', "rb") as fh:
  tag2index = pickle.load(fh)

def logits_to_tokens(sequences, index):
    token_sequences = []
    for categorical_sequence in sequences:
        token_sequence = []
        for categorical in categorical_sequence:
            token_sequence.append(index[np.argmax(categorical)])
 
        token_sequences.append(token_sequence)
 
    return token_sequences


##### เข้าเป็น listt #########


