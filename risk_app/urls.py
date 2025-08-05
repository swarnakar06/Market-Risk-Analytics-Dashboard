from django.urls import path
from . import views  # This imports views.py from risk_app

urlpatterns = [
    path('', views.home, name='home'),  # Link root URL to home() view
    path('export_csv/', views.export_csv, name='export_csv'),
    path('export_pdf/', views.export_pdf, name='export_pdf'),
    path('email_report/', views.email_report, name='email_report')
]
