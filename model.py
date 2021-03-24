import keras.backend as K
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Bidirectional, GRU, SpatialDropout1D
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from constants import model_parameters, environment
import itertools
import helper as hp

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class OffensiveDetector:

    def __init__(self, m_hyperparameters):
        self.hparams = m_hyperparameters
        self.model = None

    def _load_build_embeddings(self):
        print("Loading and building embedding matrix")
        embeddings_index = dict()
        #        f = open('glove.twitter.27B.200d.txt')
        #        f = open('glove.twitter.27B.100d.txt')
        if model_parameters['embedding'] == 'glove':
            f = open('embeddings/glove.42B.300d.txt',  encoding="utf8")  # LSTM Console 1
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        else:
            f = open('embeddings/word2vec.300d.txt', encoding="utf8")
            for line in f:
                values = line.split()
                word = values[0][:values[0].find("_")].lower()
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((self.hparams['vocab_size'], 300))
        for word, index in self.hparams['tokenizer'].word_index.items():
            if index > self.hparams['vocab_size'] - 1:
                break
            else:
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index] = embedding_vector

        self.embedding_matrix = embedding_matrix

    def f1_macro(self, y_true, y_pred):
        y_pred = K.round(y_pred)
        tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
        # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
        fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
        fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

        p = tp / (tp + fp + K.epsilon())
        r = tp / (tp + fn + K.epsilon())

        f1 = 2 * p * r / (p + r + K.epsilon())
        f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
        return K.mean(f1)

    def build(self, optimizer='adam'):
        if self.hparams['use_pretrained_embedding']:
            self._load_build_embeddings()
        # Network architecture
        model = Sequential()
        if self.hparams['use_pretrained_embedding']:
            print("\n")
            print("Using pretrained embedding matrix")
            model.add(Embedding(self.hparams['vocab_size'], 300, input_length=50, weights=[self.embedding_matrix],
                                trainable=False))  # trainable = false/true
        else:
            model.add(Embedding(self.hparams['vocab_size'], 100, input_length=50))
        model.add(Bidirectional(LSTM(300, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)))
        #        model.add(Bidirectional(GRU(300, return_sequences = True, dropout=0.35, recurrent_dropout=0.35)))
        if self.hparams['convolutional_layer']:
            model.add(Conv1D(128, 4, activation='relu'))
            model.add(MaxPooling1D(pool_size=4))
            # model.add(Flatten())
        if self.hparams['rnn_layer_after_cnn']:
            if self.hparams['rnn_layer'] == 'LSTM':
                model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.5))
            elif self.hparams['rnn_layer'] == 'Bidirectional LSTM':
                model.add(Bidirectional(LSTM(100, dropout=0.5, recurrent_dropout=0.5)))
            elif self.hparams['rnn_layer'] == 'GRU':
                model.add(GRU(100, dropout=0.5, recurrent_dropout=0.5))
            elif self.hparams['rnn_layer'] == 'Bidirectional GRU':
                model.add(Bidirectional(GRU(100, dropout=0.5, recurrent_dropout=0.5)))
        #        model.add(SpatialDropout1D(0.5))
        model.add(Dense(128, activation='relu'))
        #        model.add(SpatialDropout1D(0.5))
        model.add(Flatten())
        #        model.add(Dropout(0.2))
        if self.hparams['num_classes'] == 2:
            model.add(Dense(1, activation='sigmoid'))
        elif self.hparams['num_classes'] > 2:
            model.add(Dense(7, activation='softmax'))

        self.model = model
        if self.hparams['num_classes'] == 2:
            if self.hparams['custom_metrics_f1']:
                self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', self.f1_macro])
            else:
                self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        elif self.hparams['num_classes'] > 2:
            if self.hparams['custom_metrics_f1']:
                self.model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                                   metrics=['accuracy', self.f1_macro])
            else:
                self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        model.summary()
        return model

    def train(self, X_train_val, y_train_val, computed_weights=None, callbacks=None):
        self.model.fit(X_train_val, y_train_val, batch_size=256, validation_split=0.25, epochs=self.hparams['epochs'],
                       class_weight=computed_weights, callbacks=callbacks)
        self.model.save('trained_models/' + environment['task'] + "_" + model_parameters['embedding'] + "_model.h5")
        # self.model.save_weights('trained_models/')

    def evaluate(self, X_test, y_test):
        score = self.model.evaluate(X_test, y_test)
        print('Test accuracy:', score[1])

        predictions = self.model.predict(X_test)
        predictions_round = [np.round(x) for x in predictions]
        print(classification_report(y_test, np.array(predictions_round)))

        if self.hparams['num_classes'] == 2:
            cnf_matrix = confusion_matrix(y_test, np.array(predictions_round))
        else:
            cnf_matrix = confusion_matrix(y_test.argmax(axis=1), np.array(predictions_round).argmax(axis=1))
        self.plot_confusion_matrix(cnf_matrix, classes=model_parameters['classes'], title='Confusion matrix')
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            """

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()

    def predict(self, data_path, task):
        dataset = pd.read_csv(data_path)
        dataset = hp.remove_excess(dataset)
        dataset['comment_text'] = hp.process_tweets(dataset['comment_text'])
        # tokenizer = Tokenizer()
        # tokenizer.fit_on_texts(dataset['comment_text'])
        # self.hparams['tokenizer'] = tokenizer
        # vocabulary_size = len(tokenizer.word_counts) + 1
        # self.hparams['vocab_size'] = vocabulary_size
        test_sequences = self.hparams['tokenizer'].texts_to_sequences(dataset['comment_text'])
        test_data_sequence = pad_sequences(test_sequences, maxlen=50)
        self.model = load_model('trained_models/' + environment['task'] + "_" + model_parameters['embedding'] +
                                "_model.h5")
        # self.model.load_weights('trained_models/')
        predictions = self.model.predict(test_data_sequence)
        predictions_round = [np.round(x) for x in predictions]
        out = np.concatenate(predictions_round).ravel()
        if task == 'binary':
            dataset['label'] = ["NOT" if x == 0 else "OFF" for x in out]
        # if task == 'multi':
        #     class_labels = np.argmax(np.array(predictions), axis=1)
        #     out = np.concatenate(predictions).ravel()
        #     dataset['label'] = ["IND" if x == 1 else "GRP" if x == 0 else "OTH" for x in class_labels]
        return dataset
