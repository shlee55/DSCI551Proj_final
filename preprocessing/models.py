from django.db import models

def upload_location(instance,filename):
    return "%s/%s" %(instance.file_name, filename)

class Rawfile(models.Model):
    file_name = models.CharField(max_length=200, primary_key=True, blank=True)
    file = models.FileField(upload_to=upload_location)
    rows = models.IntegerField(default=0)
    columns = models.IntegerField(default=0)
    pub_date = models.DateTimeField(auto_now_add=True, blank=True)
    def __str__(self):
         return self.file_name

class metadata(models.Model):
    colname = models.CharField(max_length=200)
    datatype = models.CharField(max_length=200)
    contValues = models.TextField(blank=True)
    catValues = models.TextField(blank=True)
    changeable = models.BooleanField(default=False)
    navalue = models.IntegerField(default = 0)
    nahandling = models.TextField(default = "None") ## remove, average, char
    target = models.BooleanField(default=False)
    selected = models.BooleanField(default=True) ## 최종 칼럼으로 선택 됐는지 확인
    pub_date = models.DateTimeField(auto_now_add=True, blank=True)
    file_name = models.CharField(max_length=200, blank=True)
    def __str__(self):
         return self.colname

class metadataForReset(models.Model):
    colname = models.CharField(max_length=200)
    datatype = models.CharField(max_length=200)
    contValues = models.TextField(blank=True)
    catValues = models.TextField(blank=True)
    changeable = models.BooleanField(default=False)
    navalue = models.IntegerField(default = 0)
    nahandling = models.TextField(default = "No Action") ## remove, average, char
    target = models.BooleanField(default=False)
    selected = models.BooleanField(default=True) ## 최종 칼럼으로 선택 됐는지 확인
    pub_date = models.DateTimeField(auto_now_add=True, blank=True)
    file_name = models.CharField(max_length=200, blank=True)
    def __str__(self):
         return self.colname

class MLscore(models.Model):
    modelType = models.CharField(max_length=100)
    score = models.FloatField()
    file_name = models.CharField(max_length=200, blank=True)
    pub_date = models.DateTimeField(auto_now_add=True, blank=True)
    def __str__(self):
         return self.modelType

class predictTable(models.Model):
    modelType = models.CharField(max_length=100, blank=True)
    indepVariable = models.CharField(max_length=100, blank=True)
    targetVariable = models.CharField(max_length=100, blank=True)
    file_name = models.CharField(max_length=200, blank=True)
    pub_date = models.DateTimeField(auto_now_add=True, blank=True)
    def __str__(self):
         return self.modelType