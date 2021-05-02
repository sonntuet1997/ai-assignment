environment = {'SMOTE_flag': True,
               'class_weights_flag': False,
               'is_training': False,
               'cross_validation_flag': False,
               'build_final_model_flag': False,
               'predict_test_set_flag': True,
               'classes': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
               'task': 'binary'
               }

model_parameters = {'epochs': 10,
                    'classes': None,
                    'num_classes': None,
                    'tokenizer': None,
                    'vocab_size': None,
                    'embedding': 'bert',
                    'optimizer': 'adam',
                    'rnn_layer_after_cnn': True,
                    'rnn_layer': 'Bidirectional GRU',
                    'use_pretrained_embedding': True,
                    'convolutional_layer': True,
                    'custom_metrics_f1': False,
                    }
