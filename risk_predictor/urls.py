from django.contrib import admin
from django.urls import path, include  # include is important!

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('risk_app.urls')),  # 👈 connects your app URLs
]
