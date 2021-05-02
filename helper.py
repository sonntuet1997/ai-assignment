import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
import symspell_python as spell_checkers
from symspellpy import SymSpell, Verbosity
import pkg_resources

import numpy as np

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import constants
import os

import pandas as pd
import nltk
nltk.download('wordnet')
cList = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

c_re = re.compile('(%s)' % '|'.join(cList.keys()))


def expandContractions(text, c_re=c_re):
    def replace(match):
        return cList[match.group(0)]
    return c_re.sub(replace, text)


def process_comments(comments_column):
    # Apostrophe expansion
    comments_column = comments_column.apply(lambda x: x.replace("’", "'"))
    comments_column = comments_column.apply(lambda x: expandContractions(x))
    # Lowercase tweets
    comments_column = comments_column.apply(lambda x: x.lower())
    # Remove url, hashtags, cashtags, twitter handles, and RT. Only words
    comments_column = comments_column.apply(lambda x: ' '.join(re.sub(
        r"(@[A-Za-z0-9]+)|^rt |(#[A-Za-z0-9]+) |(\w+:\/*\S+)|[^a-zA-Z\s]", "", x).split()))
    # Remove url token
    comments_column = comments_column.apply(lambda x: x.replace('url', ''))
    # Lemmatisation
    tokeniser = TweetTokenizer()
    wordnet_lemmatizer = WordNetLemmatizer()
    comments_column = comments_column.apply(lambda x: [word for word in tokeniser.tokenize(x)])
    sym_spell = SymSpell()
    dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
    sym_spell.load_dictionary(dictionary_path, 0, 1)
    # spell_checkers.create_dictionary("eng_dict.txt")
    print("Spell checker...")
    for i in range(len(comments_column)):
        try:
            if i == (len(comments_column) - 1) or i % 10000 == 0:
                print('%i out of %i' % (i, len(comments_column)))
            for j in range(len(comments_column[i])):
                suggestions = sym_spell.lookup(comments_column[i][j], Verbosity.CLOSEST, max_edit_distance=2)
                # suggestions = spell_checkers.get_suggestions(comments_column[i][j])
                if suggestions:
                    best_sugg = str(suggestions[0].split(',')[0].strip())
                    # best_sugg = str(suggestions[0])
                    comments_column[i][j] = best_sugg
        except:
            continue
    
    comments_column = comments_column.apply(lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word, pos="v") for word in x]))
    
    return comments_column


def data_preparation(X, y, num_classes):
    # Create train_sequences
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X)
    vocabulary_size = len(tokenizer.word_counts) + 1
    train_sequences = tokenizer.texts_to_sequences(X)
    train_data = pad_sequences(train_sequences, maxlen=50)
    
    # Encode class values as integers
    if num_classes == 2:
        encoder = LabelEncoder()
        encoded_train_labels = encoder.fit_transform(y)
    else:
        return train_data, y, tokenizer, vocabulary_size
    # encoder.fit(y)
    # print(encoder.classes_)
    print('\n')
    print("Training labels: ")
    print(y[:5])
    print('\n')
    print("Encoded labels: ")
    print(encoded_train_labels[:5])
    
    return train_data, encoded_train_labels, tokenizer, vocabulary_size


def get_labels(data, classes, multiclass=False):
    res = []
    if not multiclass:
        for i in range(len(data)):
            flag = False
            for c in classes:
                if data.iloc[[i]][c].values == [1]:
                    flag = True
            if flag:
                res.append('OFF')
            else:
                res.append('NOT')
    else:
        return data[classes]
    return res


def remove_excess(dataset):
    labels = pd.read_csv('data/test_labels.csv')
    res = pd.merge(dataset, labels, on='id')
    res = res[res.toxic != -1]
    return res
