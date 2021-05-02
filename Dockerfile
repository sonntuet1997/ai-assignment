FROM python:latest

RUN pip install Keras  numpy   pandas  opencv-python  scikit-learn nltk  symspellpy virtualenv django djangorestframework django-cors-headers && \
    pip install --upgrade tensorflow && \
    pip install --upgrade pip

WORKDIR /app
COPY . /app

CMD ['/bin/sh', 'python manage.py runserver 0.0.0.0:8000']