from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets

from .serializers import SentenceSerializer
from .models import Sentence


class SentenceViewSet(viewsets.ModelViewSet):
    queryset = Sentence.objects.all().order_by('name')
    serializer_class = SentenceSerializer