from django.urls import path
from . import views
app_name = 'prep'
urlpatterns = [
    path('home/',views.home, name="home"), #### 이거 추가
    path('reset/',views.reset, name="reset"), #### 이거 추가
    path('load/', views.load, name="load"),  #### 이거 추가
    path('meta/', views.meta, name="meta"),  #### 이거 추가
    path('changeType/', views.changeType, name="changeType"),  #### 이거 추가
    path('target/', views.target, name="target"),  #### 이거 추가
    path('feature_ext/', views.feature_ext, name="feature_ext"),  #### 이거 추가
    path('na_handling/', views.na_handling, name="na_handling"),  #### 이거 추가
    path('<int:col_id>/graph/', views.graph, name="graph"),  #### 이거 추가
    path('finalfile/', views.finalfile, name="finalfile"),  #### 이거 추가
    path('validationCheck/', views.validationCheck, name="validationCheck"),

]


"""
path('<int:pk>/', views.DetailView.as_view(), name='detail'),
path('<int:pk>/result', views.ResultsView.as_view(), name='results'),
path('<int:question_id>/vote/', views.vote, name='vote'),
"""