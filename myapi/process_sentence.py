from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Sentence
from .serializers import SentenceSerializer
from rest_framework import status
from rest_framework.response import Response

from model import OffensiveDetector
from constants import environment
import pickle
import nltk
nltk.download('wordnet')

f = open('data/model_params.pkl', 'rb')
model_parameters = pickle.load(f)
OFF_Detector = OffensiveDetector(model_parameters)

@api_view(['GET', 'POST'])
def snippet_list(request):
    if request.method == 'GET':
        try:
            type = request.query_params['type']
        except:
            type = 'binary'
        try:
            embedding = request.query_params['embedding']
        except:
            embedding = 'bert'
        try:
            sentence = request.query_params['sentence']
        except:
            content = {'error': 'input something'}
            return Response(content, status=status.HTTP_404_NOT_FOUND)
        try:
            sentence_obj = Sentence.objects.get(name=sentence, type=type, embedding=embedding)
            serializer = SentenceSerializer(sentence_obj)
            return Response(serializer.data)

        except Sentence.DoesNotExist:
            result = OFF_Detector.predict_single(sentence, type, embedding)[0]
            data = {'name': sentence, 'type': type, 'result': result, 'embedding': embedding}
            serializer = SentenceSerializer(data=data)
            if serializer.is_valid():
                serializer.save()
                return Response(serializer.data, status=status.HTTP_201_CREATED)
            return Response(status=status.HTTP_404_NOT_FOUND)
