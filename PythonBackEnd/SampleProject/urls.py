"""SampleProject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from MyApp import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('idealweight/', views.IdealWeight),
    path('my-data/', views.MyData),
    path('my-array-data/', views.MyArrayData),
    path('alawi/', views.Alawi),
    path('api/simple/', views.Simple),
    path('api/double/', views.Double),
    path('api/triple/', views.Triple),
    path('api/arima/', views.Arima)
]
