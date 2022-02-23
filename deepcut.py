import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import pickle
import numpy as np
import scipy.sparse as sp
import six
import re
import sys
from itertools import chain
import numbers
from keras.models import Model
from keras.layers import Input, Dense, Embedding, \
    Concatenate, Flatten, SpatialDropout1D, \
    BatchNormalization, Conv1D, Maximum, ZeroPadding1D
from keras.layers import TimeDistributed
from keras.optimizers import adam_v2


from glob import glob
import pandas as pd
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

N_LEN = 21
TOKENIZER = None
WEIGHT_PATH1 = "cnn_without_ne_ab.h5"
WEIGHT_PATH2 = "weight_lst20_w_space1.h5"
#
THAI_STOP_WORDS = frozenset([
    u'ไว้', u'ไม่', u'ไป', u'ได้', u'ให้', u'ใน', u'โดย', u'แห่ง',
    u'แล้ว', u'และ', u'แรก', u'แบบ', u'แต่', u'เอง', u'เห็น',
    u'เลย', u'เริ่ม', u'เรา', u'เมื่อ', u'เพื่อ', u'เพราะ', u'เป็นการ',
    u'เป็น', u'เปิดเผย', u'เปิด', u'เนื่องจาก', u'เดียวกัน', u'เดียว',
    u'เช่น', u'เฉพาะ', u'เคย', u'เข้า', u'เขา', u'อีก', u'อาจ',
    u'อะไร', u'ออก', u'อย่าง', u'อยู่', u'อยาก', u'หาก', u'หลาย',
    u'หลังจาก', u'หลัง', u'หรือ', u'หนึ่ง', u'ส่วน', u'ส่ง', u'สุด',
    u'สําหรับ', u'ว่า', u'วัน', u'ลง', u'ร่วม', u'ราย', u'รับ', u'ระหว่าง',
    u'รวม', u'ยัง', u'มี', u'มาก', u'มา', u'พร้อม', u'พบ', u'ผ่าน',
    u'ผล', u'บาง', u'น่า', u'นี้', u'นํา', u'นั้น', u'นัก', u'นอกจาก',
    u'ทุก', u'ที่สุด', u'ที่', u'ทําให้', u'ทํา', u'ทาง', u'ทั้งนี้', u'ทั้ง',
    u'ถ้า', u'ถูก', u'ถึง', u'ต้อง', u'ต่างๆ', u'ต่าง', u'ต่อ', u'ตาม',
    u'ตั้งแต่', u'ตั้ง', u'ด้าน', u'ด้วย', u'ดัง', u'ซึ่ง', u'ช่วง', u'จึง',
    u'จาก', u'จัด', u'จะ', u'คือ', u'ความ', u'ครั้ง', u'คง', u'ขึ้น',
    u'ของ', u'ขอ', u'ขณะ', u'ก่อน', u'ก็', u'การ', u'กับ', u'กัน',
    u'กว่า', u'กล่าว',u'ลั่น'
])
CHAR_TYPE = {
    u'กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ': 'c',
    u'ฅฉผฟฌหฮ': 'n',
    u'ะาำิีืึุู': 'v',  # า ะ ำ ิ ี ึ ื ั ู ุ
    u'เแโใไ': 'w',
    u'่้๊๋': 't', # วรรณยุกต์ ่ ้ ๊ ๋
    u'์ๆฯ.': 's', # ์  ๆ ฯ .
    u'0123456789๑๒๓๔๕๖๗๘๙': 'd',
    u'"': 'q',
    u"‘": 'q',
    u"’": 'q',
    u"'": 'q',
    u' ': 'p',
    u'abcdefghijklmnopqrstuvwxyz': 's_e',
    u'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e'
}
CHAR_TYPE_FLATTEN = {}
for ks, v in CHAR_TYPE.items():
    for k in ks:
        CHAR_TYPE_FLATTEN[k] = v
#
# create map of dictionary to character
CHARS = [
    u'\n', u' ', u'!', u'"', u'#', u'$', u'%', u'&', "'", u'(', u')', u'*', u'+',
    u',', u'-', u'.', u'/', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8',
    u'9', u':', u';', u'<', u'=', u'>', u'?', u'@', u'A', u'B', u'C', u'D', u'E',
    u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R',
    u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z', u'[', u'\\', u']', u'^', u'_',
    u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm',
    u'n', u'o', u'other', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
    u'z', u'}', u'~', u'ก', u'ข', u'ฃ', u'ค', u'ฅ', u'ฆ', u'ง', u'จ', u'ฉ', u'ช',
    u'ซ', u'ฌ', u'ญ', u'ฎ', u'ฏ', u'ฐ', u'ฑ', u'ฒ', u'ณ', u'ด', u'ต', u'ถ', u'ท',
    u'ธ', u'น', u'บ', u'ป', u'ผ', u'ฝ', u'พ', u'ฟ', u'ภ', u'ม', u'ย', u'ร', u'ฤ',
    u'ล', u'ว', u'ศ', u'ษ', u'ส', u'ห', u'ฬ', u'อ', u'ฮ', u'ฯ', u'ะ', u'ั', u'า',
    u'ำ', u'ิ', u'ี', u'ึ', u'ื', u'ุ', u'ู', u'ฺ', u'เ', u'แ', u'โ', u'ใ', u'ไ',
    u'ๅ', u'ๆ', u'็', u'่', u'้', u'๊', u'๋', u'์', u'ํ', u'๐', u'๑', u'๒', u'๓',
    u'๔', u'๕', u'๖', u'๗', u'๘', u'๙', u'‘', u'’', u'\ufeff'
]
CHARS_MAP = {v: k for k, v in enumerate(CHARS)}

CHAR_TYPES = [
    'b_e', 'c', 'd', 'n', 'o',
    'p', 'q', 's', 's_e', 't',
    'v', 'w'
]
CHAR_TYPES_MAP = {v: k for k, v in enumerate(CHAR_TYPES)}
#
def create_feature_array(text, n_pad=N_LEN):
    """
    Create feature array of character and surrounding characters
    """
    n = len(text)
    n_pad_2 = int((n_pad - 1)/2)
    text_pad = [' '] * n_pad_2  + [t for t in text] + [' '] * n_pad_2
    x_char, x_type = [], []
    for i in range(n_pad_2, n_pad_2 + n):
        char_list = text_pad[i + 1: i + n_pad_2 + 1] + \
                    list(reversed(text_pad[i - n_pad_2: i])) + \
                    [text_pad[i]]
        char_map = [CHARS_MAP.get(c, 80) for c in char_list]
        char_type = [CHAR_TYPES_MAP.get(CHAR_TYPE_FLATTEN.get(c, 'o'), 4)
                     for c in char_list]
        x_char.append(char_map)
        x_type.append(char_type)
    x_char = np.array(x_char).astype(float)
    x_type = np.array(x_type).astype(float)
    return x_char, x_type
#
def create_n_gram_df(df, n_pad):
    """
    Given input dataframe, create feature dataframe of shifted characters
    """
    n_pad_2 = int((n_pad - 1)/2)
    for i in range(n_pad_2):
        df['char-{}'.format(i+1)] = df['char'].shift(i + 1)
        df['type-{}'.format(i+1)] = df['type'].shift(i + 1)
        df['char{}'.format(i+1)] = df['char'].shift(-i - 1)
        df['type{}'.format(i+1)] = df['type'].shift(-i - 1)
    return df[n_pad_2: -n_pad_2]
#
def _document_frequency(X):
    """
    Count the number of non-zero values for each feature in sparse X.
    """
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    return np.diff(sp.csc_matrix(X, copy=False).indptr)
class DeepcutTokenizer(object):
    """
    Class for tokenizing given Thai text documents using deepcut library

    Parameters
    ==========
    ngram_range : tuple, tuple for ngram range for vocabulary, (1, 1) for unigram
        and (1, 2) for bigram
    stop_words : list or set, list or set of stop words to be removed
        if None, max_df can be set to value [0.7, 1.0) to automatically remove
        vocabulary. If using "thai", this will use list of pre-populated stop words
    max_features : int or None, if provided, only consider number of vocabulary
        ordered by term frequencies
    max_df : float in range [0.0, 1.0] or int, default=1.0
        ignore terms that have a document frequency higher than the given threshold
    min_df : float in range [0.0, 1.0] or int, default=1
        ignore terms that have a document frequency lower than the given threshold
    dtype : type, optional


    Example
    =======
    raw_documents = ['ฉันอยากกินข้าวของฉัน',
                     'ฉันอยากกินไก่',
                     'อยากนอนอย่างสงบ']
    tokenizer = DeepcutTokenizer(ngram_range=(1, 1))
    X = tokenizer.fit_tranform(raw_documents) # document-term matrix in sparse CSR format

    >> X.todense()
    >> [[0, 0, 1, 0, 1, 0, 2, 1],
        [0, 1, 1, 0, 1, 0, 1, 0],
        [1, 0, 0, 1, 1, 1, 0, 0]]
    >> tokenizer.vocabulary_
    >> {'นอน': 0, 'ไก่': 1, 'กิน': 2, 'อย่าง': 3, 'อยาก': 4, 'สงบ': 5, 'ฉัน': 6, 'ข้าว': 7}

    """

    def __init__(self, ngram_range=(1, 1), stop_words=None,
                 max_df=1.0, min_df=1, max_features=None, dtype=np.dtype('float64')):
        self.model1 = get_convo_nn2()
        self.model1.load_weights(WEIGHT_PATH1)
        self.model2 = get_convo_nn2()
        self.model2.load_weights(WEIGHT_PATH2)
        self.vocabulary_ = {}
        self.ngram_range = ngram_range
        self.dtype = dtype
        self.max_df = max_df
        self.min_df = min_df
        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        self.max_features = max_features
        self.stop_words = _check_stop_list(stop_words)


    def _word_ngrams(self, tokens):
        """
        Turn tokens into a tokens of n-grams

        ref: https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/feature_extraction/text.py#L124-L153
        """
        # handle stop words
        if self.stop_words is not None:
            tokens = [w for w in tokens if w not in self.stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                # no need to do any slicing for unigrams
                # just iterate through the original tokens
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

            n_original_tokens = len(original_tokens)

            # bind method outside of loop to reduce overhead
            tokens_append = tokens.append
            space_join = " ".join

            for n in range(min_n,
                           min(max_n + 1, n_original_tokens + 1)):
                for i in range(n_original_tokens - n + 1):
                    tokens_append(space_join(original_tokens[i: i + n]))

        return tokens


    def _limit_features(self, X, vocabulary,
                        high=None, low=None, limit=None):
        """Remove too rare or too common features.

        ref: https://github.com/scikit-learn/scikit-learn/blob/ef5cb84a/sklearn/feature_extraction/text.py#L734-L773
        """
        if high is None and low is None and limit is None:
            return X, set()

        # Calculate a mask based on document frequencies
        dfs = _document_frequency(X)
        mask = np.ones(len(dfs), dtype=bool)
        if high is not None:
            mask &= dfs <= high
        if low is not None:
            mask &= dfs >= low
        if limit is not None and mask.sum() > limit:
            tfs = np.asarray(X.sum(axis=0)).ravel()
            mask_inds = (-tfs[mask]).argsort()[:limit]
            new_mask = np.zeros(len(dfs), dtype=bool)
            new_mask[np.where(mask)[0][mask_inds]] = True
            mask = new_mask

        new_indices = np.cumsum(mask) - 1  # maps old indices to new
        removed_terms = set()
        for term, old_index in list(vocabulary.items()):
            if mask[old_index]:
                vocabulary[term] = new_indices[old_index]
            else:
                del vocabulary[term]
                removed_terms.add(term)
        kept_indices = np.where(mask)[0]
        if len(kept_indices) == 0:
            raise ValueError("After pruning, no terms remain. Try a lower"
                             " min_df or a higher max_df.")
        return X[:, kept_indices], removed_terms


    def transform(self, raw_documents, new_document=False):
        """
        raw_documents: list, list of new documents to be transformed
        new_document: bool, if True, assume seeing documents and build a new self.vobabulary_,
            if False, use the previous self.vocabulary_
        """
        n_doc = len(raw_documents)
        tokenized_documents = []
        for doc in raw_documents:
            tokens = tokenize(doc) # method in this file
            tokens = self._word_ngrams(tokens)
            tokenized_documents.append(tokens)

        if new_document:
            self.vocabulary_ = {v: k for k, v in enumerate(set(chain.from_iterable(tokenized_documents)))}

        values, row_indices, col_indices = [], [], []
        for r, tokens in enumerate(tokenized_documents):
            tokens = self._word_ngrams(tokens)
            feature = {}
            for token in tokens:
                word_index = self.vocabulary_.get(token)
                if word_index is not None:
                    if word_index not in feature.keys():
                        feature[word_index] = 1
                    else:
                        feature[word_index] += 1
            for c, v in feature.items():
                values.append(v)
                row_indices.append(r)
                col_indices.append(c)

        # document-term matrix in CSR format
        X = sp.csr_matrix((values, (row_indices, col_indices)),
                          shape=(n_doc, len(self.vocabulary_)),
                          dtype=self.dtype)

        # truncate vocabulary by max_df and min_df
        if new_document:
            max_df = self.max_df
            min_df = self.min_df
            max_doc_count = (max_df
                            if isinstance(max_df, numbers.Integral)
                            else max_df * n_doc)
            min_doc_count = (min_df
                            if isinstance(min_df, numbers.Integral)
                            else min_df * n_doc)
            if max_doc_count < min_doc_count:
                raise ValueError(
                    "max_df corresponds to < documents than min_df")
            X, _ = self._limit_features(X, self.vocabulary_,
                                        max_doc_count,
                                        min_doc_count,
                                        self.max_features)

        return X


    def fit_tranform(self, raw_documents):
        """
        Transform given list of raw_documents to document-term matrix in
        sparse CSR format (see scipy)
        """
        X = self.transform(raw_documents, new_document=True)
        return X

    def tokenize(self, text, custom_dict=None):
        n_pad = N_LEN

        if not text:
            return [''] # case of empty string

        if isinstance(text, str) and sys.version_info.major == 2:
            text = text.decode('utf-8')

        x_char, x_type = create_feature_array(text, n_pad=n_pad)
        word_end = []
        # Fix thread-related issue in Keras + TensorFlow + Flask async environment
        # ref: https://github.com/keras-team/keras/issues/2397
 
        y_predict1 = self.model1.predict([x_char, x_type])
        y_predict2 = self.model2.predict([x_char, x_type])
#         y_predict = (y_predict1.ravel()+y_predict2.ravel() > 0.5*2).astype(int)
        y_predict = np.logical_or((y_predict1.ravel()> 0.5), (y_predict2.ravel()> 0.5)).astype(int)
#         y_predict = self.model.predict(x_char)
#         y_predict = np.argmax(y_predict,axis = 1)
        word_end = y_predict[1:].tolist() + [1]

        if custom_dict is not None:
            if isinstance(custom_dict, list):
                word_list = custom_dict
            else:
                word_list = []
                try:
                    with open(custom_dict) as f:
                        word_list = f.readlines()
                except:
                    pass
            if len(word_list) > 0:
                for word in word_list:
                    if isinstance(word, str) and sys.version_info.major == 2:
                        word = word.decode('utf-8')
                    word = word.strip('\n')
                    word_end = _custom_dict(word, text, word_end)

        tokens = []
        word = ''
        for char, w_e in zip(text, word_end):
            word += char
            if w_e:
                tokens.append(word)
                word = ''
        return tokens

    def save_model(self, file_path):
        """
        Save tokenizer to pickle format
        """
        self.model1 = None # set model to None to successfully save the model
        self.model2 = None # set model to None to successfully save the model
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
#
def tokenize(text, custom_dict=None):
    """
    Tokenize given Thai text string

    Input
    =====
    text: str, Thai text string
    custom_dict: str (or list), path to customized dictionary file
        It allows the function not to tokenize given dictionary wrongly.
        The file should contain custom words separated by line.
        Alternatively, you can provide list of custom words too.

    Output
    ======
    tokens: list, list of tokenized words

    Example
    =======
    >> deepcut.tokenize('ตัดคำได้ดีมาก')
    >> ['ตัดคำ','ได้','ดี','มาก']

    """
    global TOKENIZER
    if not TOKENIZER:
        TOKENIZER = DeepcutTokenizer()
    return TOKENIZER.tokenize(text, custom_dict=custom_dict)
#
def _custom_dict(word, text, word_end):
    word_length = len(word)
    initial_loc = 0

    while True:
        try:
            start_char = re.search(word, text).start()
            first_char = start_char + initial_loc
            last_char = first_char + word_length - 1

            initial_loc += start_char + word_length
            text = text[start_char + word_length:]
            word_end[first_char:last_char] = (word_length - 1) * [0]
            word_end[last_char] = 1
        except:
            break
    return word_end
#
def _check_stop_list(stop):
    """
    Check stop words list
    ref: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/feature_extraction/text.py#L87-L95
    """
    if stop == "thai":
        return THAI_STOP_WORDS
    elif isinstance(stop, six.string_types):
        raise ValueError("not a built-in stop list: %s" % stop)
    elif stop is None:
        return None
    # assume it's a collection
    return frozenset(stop)
#
def conv_unit(inp, n_gram, no_word=200, window=2):
    out = Conv1D(no_word, window, strides=1, padding="valid", activation='relu')(inp)
    out = TimeDistributed(Dense(5, input_shape=(n_gram, no_word)))(out)
    out = ZeroPadding1D(padding=(0, window - 1))(out)
    return out
#
def get_convo_nn2(no_word=200, n_gram=N_LEN, no_char=178):
#     return CreateModel(no_word, n_gram, no_char)
    input1 = Input(shape=(n_gram,))
    input2 = Input(shape=(n_gram,))

    a = Embedding(no_char, 32, input_length=n_gram)(input1)
    a = SpatialDropout1D(0.15)(a)
    a = BatchNormalization()(a)

    a_concat = []
    for i in range(1,9):
        a_concat.append(conv_unit(a, n_gram, no_word, window=i))
    for i in range(9,12):
        a_concat.append(conv_unit(a, n_gram, no_word - 50, window=i))
    a_concat.append(conv_unit(a, n_gram, no_word - 100, window=12))
    a_sum = Maximum()(a_concat)

    b = Embedding(12, 12, input_length=n_gram)(input2)
    b = SpatialDropout1D(0.15)(b)

    x = Concatenate(axis=-1)([a, a_sum, b])
    #x = Concatenate(axis=-1)([a_sum, b])
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input1, input2], outputs=out)
    model.compile(
                  loss='binary_crossentropy', metrics=['acc'])
    return model
# Load Model
def load_model(file_path):
    """
    Load saved pickle file of DeepcutTokenizer

    Parameters
    ==========
    file_path: str, path to saved model from ``save_model`` method in DeepcutTokenizer 
    """
    tokenizer = pickle.load(open(file_path, 'rb'))
    tokenizer.model1 = get_convo_nn2()
    tokenizer.model2 = get_convo_nn2()
    tokenizer.model1 = tokenizer.model1.load_weights(WEIGHT_PATH1)
    tokenizer.model2 = tokenizer.model2.load_weights(WEIGHT_PATH2)
    return tokenizer