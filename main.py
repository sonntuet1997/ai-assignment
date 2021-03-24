import pandas as pd
from preprocessing import get_train_data, SMOTE_handling, class_weights_handling
from model import OffensiveDetector
from constants import model_parameters, environment
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
import tensorflow as tf


print("Initialising classifier...")
OFF_Detector = OffensiveDetector(model_parameters)

x_train, x_test, y_train, y_test = get_train_data('data/train.csv', multiclass=False, multilabel=False)

if environment['is_training']:
    if environment['SMOTE_flag']:
        x_train, y_train = SMOTE_handling(x_train, y_train)
    elif environment['class_weights_flag']:
        x_train, y_train, weights = class_weights_handling(x_train, y_train)

    # K-fold
    kfold = KFold(n_splits=5, shuffle=True, random_state=2)
    skf = StratifiedKFold(n_splits=5, random_state=2, shuffle=True)

    # Wrapping our classifier with Keras wrapper + Cross validation
    if environment['cross_validation_flag']:
        print('\n')
        print("Performing cross validation...")
        estimator = KerasClassifier(build_fn=OFF_Detector.build, epochs=model_parameters['epochs'], verbose=1)
        scoring = {'acc': 'accuracy',
                   'f1_macro': 'f1_macro'}
        if environment['class_weights_flag']:
            results = cross_validate(estimator, x_train, y_train, cv=skf, scoring=scoring,
                                     fit_params={'class_weight': weights}, return_estimator=True)
        else:
            results = cross_validate(estimator, x_train, y_train, cv=skf, scoring=scoring, return_estimator=True)
        print("Average accuracy: %.2f" % results['test_acc'].mean())
        print("Average F1_macro: %.2f" % results['test_f1_macro'].mean())
    #    else:
    #        OFF_Detector.build()
    #        es = [EarlyStopping(monitor='val_f1_macro', patience = 3, mode='max', verbose = 1)]
    #        if environment['class_weights_flag'] == True:
    #            OFF_Detector.train(X_train, y_train, weights, es)
    #        else:
    #            OFF_Detector.train(X_train, y_train, es)
    #        OFF_Detector.evaluate(X_test, y_test)

    # Finalise model and train and evaluate
    if environment['build_final_model_flag']:
        print('\n')
        print("Building final model...")
        with tf.device('/device:GPU:0'):
            callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
            OFF_Detector.build()
            if environment['class_weights_flag']:
                OFF_Detector.train(x_train, y_train, weights)
            else:
                OFF_Detector.train(x_train, y_train)
            OFF_Detector.evaluate(x_test, y_test)


if environment['predict_test_set_flag']:
    print('\n')
    print("Predicting test set...")
    data = OFF_Detector.predict("data/test.csv", environment['task'])
    print(data)
    # data[['id', 'comment_text', 'label']].to_csv('result.csv', index=False)


