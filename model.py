from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from constants import model_parameters, environment
import helper as hp

class OffensiveDetector:

    def __init__(self, m_hyperparameters):
        self.hparams = m_hyperparameters
        self.model = None

    def predict_single(self, sentence, task, embedding):
        column = pd.Series([sentence])
        column = hp.process_comments(column)
        test_sequences = self.hparams['tokenizer'].texts_to_sequences(column)
        test_data_sequence = pad_sequences(test_sequences, maxlen=50)
        print('model path: ' + 'trained_models/' + task + "_" + embedding +
                                "_model.h5")
        self.model = load_model('trained_models/' + task + "_" + embedding +
                                "_model.h5")
        predictions = self.model.predict(test_data_sequence)
        predictions_round = [np.round(x) for x in predictions]
        out = np.concatenate(predictions_round).ravel()
        result = []
        if task == 'binary':
            result = ["NOT" if x == 0 else "OFF" for x in out]
        return result
