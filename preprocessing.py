import pandas as pd
from constants import model_parameters, environment
import helper as hp
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import os


def get_train_data(data_paths, multiclass=False):
    temp = []
    for data_path in data_paths:
        dataset = pd.read_csv(data_path)
        if data_path == 'data/test.csv':
            dataset = hp.remove_excess(dataset)
        temp.append(dataset)
    train_data = pd.concat(temp)
    train_data.drop(['id'], axis=1, inplace=True)
    for c in environment['classes']:
        train_data[c].value_counts()
    train_data['comment_text'] = hp.process_comments(train_data['comment_text'])
    # print(train_data['comment_text'])
    # print(train_data['comment_text'])
    train_labels = hp.get_labels(train_data, environment['classes'], multiclass)
    if not multiclass:
        model_parameters['classes'] = np.unique(train_labels)
        model_parameters['num_classes'] = len(np.unique(train_labels))
    else:
        model_parameters['classes'] = environment['classes']
        model_parameters['num_classes'] = len(environment['classes'])
    # print(train_labels)
    train_data, encoded_train_labels, tokenizer, size_of_vocab = hp.data_preparation(train_data['comment_text'],
                                                                                     train_labels,
                                                                                     model_parameters['num_classes'])
    model_parameters['tokenizer'] = tokenizer
    model_parameters['vocab_size'] = size_of_vocab

    print("Train/Test split...")
    x_train, x_test, y_train, y_test = train_test_split(train_data, encoded_train_labels, test_size=0.2, random_state=2)
    return x_train, x_test, y_train, y_test


def SMOTE_handling(x_train, y_train):
    print("To counter imbalanced dataset: SMOTE Oversampling...")
    sm = SMOTE(random_state=12)
    x_train, y_train = sm.fit_sample(x_train, y_train)
    return x_train, y_train


def class_weights_handling(x_train, y_train):
    print("To counter imbalanced dataset: Balancing class weights...")
    if model_parameters['num_classes'] == 2:
        cw = compute_class_weight("balanced", np.unique(y_train), y_train)
    elif model_parameters['num_classes'] > 2:
        y_integers = np.argmax(y_train, axis=1)
        cw = compute_class_weight('balanced', np.unique(y_integers), y_integers)
    weights = dict(enumerate(cw))
    return x_train, y_train, weights
