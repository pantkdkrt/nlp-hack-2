from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import json
import math
import pickle
from keras.models import load_model
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from POE import model,logits_to_tokens,word2index,tag2index
from deepcut import DeepcutTokenizer, tokenize

app = Flask(__name__)
CORS(app)


@app.route('/', methods=['GET'])
@cross_origin()
def base_url():
    """Base url to test API."""

    response = {
        'response': 'Hello world!'
    }

    return jsonify(response)

@app.route('/pos', methods=['POST'])
@cross_origin()

def get_POS_Tag():
    record = json.loads(request.data)['sentence']
    tokens = tokenize(record.replace(" ", ""))
    #words = jsonify(tokens).reshape(tokens.shape[1],tokens.shape[0])
    new_word = tokens
    new_word_model = []
    for i in new_word :
        if i == ' ':
            continue
        else :
            new_word_model.append(i)
    len_sen = 3 # window_size
    count = 0
    test_word = [ ]
    sen2 = []
    for i in new_word_model :
        sen2.append(i)
        count += 1
        if count == len_sen :
            test_word.append(sen2)
            sen2 = []
            count = 0

    test_word.append(sen2)

    test_word.append(sen2)
    test_samples_X = []
    test_samples_X_count = []
    for s in test_word:
        s_int = []
        for w in s:
            try:
                s_int.append(word2index[w])
            except KeyError:
                s_int.append(word2index['-OOV-'])

        test_samples_X_count.append(len(s_int))
        test_samples_X.append(s_int)
    test_samples_X = pad_sequences(test_samples_X, maxlen=len_sen, padding='post')
    predictions = model.predict(test_samples_X)
    result = logits_to_tokens(predictions, {i: t for t, i in tag2index.items()})
   
    output_data = []
    for num_l in range(0, len(test_samples_X_count)) :
        for i in range(0, test_samples_X_count[num_l]) :
            output_data.append(result[num_l][i])

    new_output = []
    sam = 0
    for idx in range(len(output_data)):
        if record[idx] == ' ' :
            new_output.append("PU")
        elif record[idx] != ' ' : 
            new_output.append(output_data[sam])
            sam += 1
    #poe = []
    x = zip(new_word,new_output)

    return jsonify(list(x))