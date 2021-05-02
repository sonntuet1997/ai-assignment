from rest_framework import serializers

from .models import Sentence

class SentenceSerializer(serializers.HyperlinkedModelSerializer):
    class Meta:
        model = Sentence
        fields = ('id','name', 'type', 'result', 'embedding')
