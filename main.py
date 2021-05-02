from model import OffensiveDetector
import pickle
import nltk
nltk.download('wordnet')

print("Initialising classifier...")
f = open('data/model_params.pkl', 'rb')
model_parameters = pickle.load(f)
OFF_Detector = OffensiveDetector(model_parameters)
print('\n')
print("Predicting test set...")
data = OFF_Detector.predict_single("fuck you", 'binary','bert')
# data = OFF_Detector.predict("data/test.csv", environment['task'])
print(data)
# data[['id', 'comment_text', 'label']].to_csv('result.csv', index=False)
