from django.db import models

# Create your models here.
# models.py
from django.db import models
class Sentence(models.Model):
    name = models.CharField(max_length=200)
    type = models.CharField(max_length=200)
    embedding = models.CharField(max_length=200)
    result = models.CharField(max_length=10)
    def __str__(self):
        return self.name + ' ' + self.type