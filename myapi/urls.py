from django.urls import include, path
from rest_framework import routers
from . import process_sentence

router = routers.DefaultRouter()

# Wire up our API using automatic URL routing.
# Additionally, we include login URLs for the browsable API.
urlpatterns = [
    path('', include(router.urls)),
    path('sentence/', process_sentence.snippet_list),
    path('api-auth/', include('rest_framework.urls', namespace='rest_framework'))
]