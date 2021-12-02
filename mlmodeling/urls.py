from django.urls import path
from . import views
app_name = 'mlm'
urlpatterns = [
    path('mlHome/',views.mlHome, name="mlHome"), #### 이거 추가
    path('downloadFile/', views.downloadFile, name="downloadFile"),  #### 이거 추가
    path('compareml/', views.compareml, name="compareml"),  #### 이거 추가
    path('selectml/', views.selectml, name="selectml"),  #### 이거 추가
    path('validationScore/', views.validationScore, name="validationScore"),  #### 이거 추가
    path('predict_lr/', views.predict_lr, name="predict_lr"),  #### 이거 추가
    path('predict_rf/', views.predict_rf, name="predict_rf"),  #### 이거 추가
]


"""
path('<int:pk>/', views.DetailView.as_view(), name='detail'),
path('<int:pk>/result', views.ResultsView.as_view(), name='results'),
path('<int:question_id>/vote/', views.vote, name='vote'),
"""