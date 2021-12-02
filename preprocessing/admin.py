from django.contrib import admin

# Register your models here.
from django.contrib import admin
from .models import Rawfile, metadata, metadataForReset, MLscore, predictTable

admin.site.register(Rawfile)
admin.site.register(metadata)
admin.site.register(metadataForReset)
admin.site.register(MLscore)
admin.site.register(predictTable)