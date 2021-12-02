from django.shortcuts import render, redirect
from django.conf import settings
from .models import Rawfile, metadata, metadataForReset, MLscore, predictTable
from pyspark import SparkFiles
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col,isnan,when,count
from pyspark.sql.types import *
from io import StringIO
import boto3
import boto
import pandas as pd
from .utils import groupBarChart, barChart, boxPlot, densityCurvesWithHistogram, histogram
from datetime import datetime
from datetime import timezone

def resetDataset(username):
    Rawfile.objects.filter(file_name=username).delete()  ## remove every data from Rawfile database
    metadata.objects.filter(file_name=username).delete()  ## remove every data from metadata database
    metadataForReset.objects.filter(file_name=username).delete()  ## remove every data from metadataForReset database
    MLscore.objects.filter(file_name=username).delete()  ## remove every data from MLscore database
    predictTable.objects.filter(file_name=username).delete()  ## remove every data from MLscore database

    checkAll = Rawfile.objects.all()
    for rawfile in checkAll:
        filename = rawfile.file_name
        delta = datetime.now(timezone.utc) - rawfile.pub_date
        if (delta.seconds//60)%3600 >= 24: # remove queryset after 24 hours
            Rawfile.objects.filter(file_name=filename).delete()  ## remove every data from Rawfile database
            metadata.objects.filter(file_name=filename).delete()  ## remove every data from metadata database
            metadataForReset.objects.filter(file_name=filename).delete()  ## remove every data from metadataForReset database
            MLscore.objects.filter(file_name=filename).delete()  ## remove every data from MLscore database
            predictTable.objects.filter(file_name=filename).delete()  ## remove every data from MLscore database

def rawFileExist(username):
    try:
        Rawfile.objects.get(file_name=username)
    except:
        raw_file = False
    else:
        raw_file = True
    return(raw_file)

def home(request):
    posted = False
    username=""
    if request.method == "POST":  ## 뭔가 입력이 들어왔을때
        posted = True
        username = request.POST.get("username")
        request.session['username'] = username
    return render(request, "home.html", {"posted":posted, "username":username})

def reset(request):
    username = request.session['username']
    resetDataset(username)
    return render(request, "home.html")


def load(request):
    username = request.session['username']
    posted = False
    loaded = False
    if request.method == "POST":  ## 뭔가 입력이 들어왔을때
        posted = True
        resetDataset(username)

        file = request.FILES.get("myfile")
        if str(file).split(".")[-1] == "csv":
            RawFileNewRow = Rawfile(file_name=username, file=file)
            RawFileNewRow.save()
            df = ""
            try:
                row = Rawfile.objects.get(file_name=username)
                spark = SparkSession(SparkContext.getOrCreate())
                spark.sparkContext.addFile(row.file.url)
                df = spark.read.csv("file:///" + SparkFiles.get(str(file)), header=True, inferSchema=True)
                NoRows = df.count()
                NoCols = len(df.columns)
                Rawfile.objects.filter(file_name=username).update(rows=NoRows)
                Rawfile.objects.filter(file_name=username).update(columns=NoCols)
            except: ## if file can't be transformed to dataframe, remove data
                pass
            else:
                loaded = True  ## file loaded successfully
                column_list = df.columns
                for colname in column_list: ## metadata 테이블 입력을 위한 for loop
                    contValues = ""
                    catValues = ""
                    noNA = df.select(count(when( (col(colname) == "None") | \
                                         (col(colname) == "NULL") | \
                                         (col(colname) == "null") | \
                                         (col(colname) == "" ) | \
                                         col(colname).isNull() | \
                                         isnan(colname),colname))).head()[0] ## count number of NA values
                    typeOfData = df.schema[colname].dataType ## type of variable
                    if isinstance(typeOfData, IntegerType) or \
                        isinstance(typeOfData, FloatType) or isinstance(typeOfData, DoubleType): ## numeric variable인지 확인
                        changeable = True
                        datatype = "Continuous"
                        contValues = "range from {} to {} and mean is {}".format(df.agg({colname: 'min'}).head()[0],
                                                                             df.agg({colname: 'max'}).head()[0],
                                                                             round(df.agg({colname: 'mean'}).head()[0],2))
                        ## catValues: values when the variable type changes to categorical variable.
                        dfPd = df.select(colname).distinct().orderBy(df[colname].asc()).dropna()
                        dfPd_cont = pd.unique(dfPd.toPandas()[colname])
                        valueList = dfPd_cont.astype(str).tolist()
                        catValues = ", ".join(valueList)
                    else:
                        changeable = False
                        datatype = "Categorical"
                        dfPd = df.select(colname).distinct().orderBy(df[colname].asc()).toPandas().dropna()
                        valueList = dfPd[colname].values.tolist()
                        if len(valueList) is not None:
                            catValues = ", ".join(valueList)
                            try:
                                dfPd[colname].astype(float)
                            except:
                                pass
                            else:
                                contValues = "range from {} to {} and mean is {}".format(dfPd[colname].min(),
                                                                                         dfPd[colname].max(),
                                                                                         round(dfPd[colname].mean(),2))
                                changeable = True

                    meta = metadata(colname=colname, datatype=datatype, contValues=contValues, catValues = catValues, changeable=changeable, navalue = noNA,file_name=username)
                    meta.save()
                    metaForReset = metadataForReset(colname=colname, datatype=datatype, contValues=contValues, catValues = catValues, changeable=changeable, navalue = noNA,file_name=username)
                    metaForReset.save()
                spark.stop()
    context = {"loaded": loaded, "posted": posted}
    return render(request, "load.html", context)

def meta(request):
    username = request.session['username']
    raw_file = rawFileExist(username)
    try:
        rawRow = Rawfile.objects.get(file_name=username)
    except:
        rawRow = ""
    return render(request, "meta.html",{"metadata":metadata.objects.filter(file_name=username), "RawfileQuery": rawRow, "raw_file":raw_file, "MLValidFile":True})

def changeType(request):
    username = request.session['username']
    raw_file = rawFileExist(username)

    if request.method == "POST":  ## 뭔가 입력이 들어왔을때
        meta = metadata.objects.filter(file_name=username)
        for row in meta:
            if row.changeable == True:
                typeChange = request.POST.get(row.colname)
                metadata.objects.filter(file_name=username, colname=row.colname).update(datatype=typeChange)
        return redirect("/prep/meta")

    return render(request, "changeType.html",{"metadata":metadata.objects.filter(file_name=username),"raw_file":raw_file})

def target(request):
    username = request.session['username']
    raw_file = rawFileExist(username)

    if request.method == "POST":  ## 뭔가 입력이 들어왔을때
        ## set target variable as true. otherwise false.
        target = request.POST.get("target")
        indepVar = metadata.objects.filter(file_name=username).exclude(colname = target)
        indepVar.update(target=False)
        targVar = metadata.objects.filter(file_name=username, colname = target)
        targVar.update(target=True)
        targVar.update(selected=True)
        return redirect("/prep/meta")

    return render(request, "target.html",{"metadata":metadata.objects.filter(file_name=username),"raw_file":raw_file})


def feature_ext(request):
    username = request.session['username']
    raw_file = rawFileExist(username)

    if request.method == "POST":  ## 뭔가 입력이 들어왔을때
        ## set target variable as true. otherwise false.
        meta = metadata.objects.filter(file_name=username)
        for row in meta:
            rowstatus = request.POST.get(row.colname)
            if rowstatus == "on" or row.target == True:
                metadata.objects.filter(file_name=username, colname=row.colname).update(selected=True)
            else:
                metadata.objects.filter(file_name=username, colname=row.colname).update(selected=False)

        return redirect("/prep/meta")
    return render(request, "feature_ext.html",{"metadata":metadata.objects.filter(file_name=username),"raw_file":raw_file})


def graph(request, col_id):
    username = request.session['username']
    try:
        row = Rawfile.objects.get(file_name=username)
        targetRow = metadata.objects.filter(file_name=username).get(target=True)
        targetValue = targetRow.colname

        metaRowValue = metadata.objects.filter(file_name=username).get(id=col_id)
        colnameValue = metaRowValue.colname
        datatypeValue = metaRowValue.datatype
        file = row.file.name

        spark = SparkSession(SparkContext.getOrCreate())
        spark.sparkContext.addFile(row.file.url)
        df = spark.read.csv("file:///" + SparkFiles.get(file.split("/")[-1]), header=True, inferSchema=True)
    except:
        targetExist = False; chart1=""; chart2 = ""; chart3 = ""; colnameValue=""
    else:
        targetExist = True
        if colnameValue == targetValue:
            df_cat = df.groupBy(targetValue).count()
            df_cat = df_cat.toPandas()
            dict_cat = {}
            for column in df_cat.columns:
                dict_cat[column] = df_cat[column].values.tolist()
            chart1 = barChart(dict_cat, targetValue)
            chart2 = False
            chart3 = False

        elif datatypeValue == "Categorical":
            df_cat = df.groupBy(targetValue, colnameValue).count()
            df_cat = df_cat.groupBy(targetValue).pivot(colnameValue).sum('count')
            df_cat = df_cat.toPandas()
            dict_cat = {}
            for column in df_cat.columns:
                dict_cat[column] = df_cat[column].values.tolist()

            df_cat2 = df.groupBy(colnameValue,targetValue).count()
            df_cat2 = df_cat2.groupBy(colnameValue).pivot(targetValue).sum('count')
            df_cat2 = df_cat2.toPandas()
            dict_cat2 = {}
            for column in df_cat2.columns:
                dict_cat2[column] = df_cat2[column].values.tolist()
            chart1 = barChart(dict_cat2,colnameValue)
            chart2 = groupBarChart(dict_cat,targetValue,colnameValue)
            chart3 = groupBarChart(dict_cat2,colnameValue,targetValue)

        else:
            df_cont = df.toPandas()
            dict_cont = {}
            for column in df_cont.columns:

                dict_cont[column] = df_cont[column].values.tolist()

            chart1 = boxPlot(dict_cont,targetValue,colnameValue)
            chart2 = densityCurvesWithHistogram(dict_cont,targetValue,colnameValue)
            chart3 = histogram(dict_cont,colnameValue)
        spark.stop()
    context = {'chart1':chart1,'chart2':chart2,'chart3':chart3,"colnameValue":colnameValue,"targetExist":targetExist}
    return render(request, "graph.html",context)


#    if row.target == True:
#        pass

#    dict_ex= {"labels":[9,8,3,2,5],"man":[5,3,1,4,2],"woman":[9,8,5,9,5]}
#    dict_ex2 = {"labels": ["avb", "bsd", "asdc", "d", "e"], "man": [10, 23, 31, 44, 22], "woman": [9, 8, 5, 9, 5]}
#    dict_ex3 = {"labels": ['G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2'],\
#        "men": [20, 34, 30, 35, 2, 27, 13, 24, 35, 213, 12, 123, 1, 2, 42, 1, 23, 12, 31, 13],\
#        "women": [25, 32, 34, 20, 25, 123, 42, 35, 46, 74, 24, 12, 42, 12, 32, 2, 5, 34, 234, 345]}


def na_handling(request):
    username = request.session['username']
    raw_file = rawFileExist(username)


    if request.method == "POST":  ## 뭔가 입력이 들어왔을때
        ## set target variable as true. otherwise false.
        meta = metadata.objects.filter(file_name=username)
        for row in meta:
            rowstatus = request.POST.get(row.colname)
            if rowstatus == "on":
                metadata.objects.filter(file_name=username, colname=row.colname).update(nahandling="remove")
            else:
                metadata.objects.filter(file_name=username, colname=row.colname).update(nahandling="None")
        return redirect("/prep/meta")
    return render(request, "na_handling.html",{"metadata":metadata.objects.filter(file_name=username),"raw_file":raw_file})

def finalfile(request):
    username = request.session['username']
    fileDir = "media/{}".format(username)
    filePath = "{}/{}_finalfile.csv".format(fileDir, username)
    conn = boto.connect_s3(aws_access_key_id = settings.AWS_ACCESS_KEY_ID, aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY)
    bucket = conn.get_bucket('slfolder')
    bucket.delete_key(filePath)
    s3 = boto3.client('s3', aws_access_key_id = settings.AWS_ACCESS_KEY_ID, aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY)

    row = Rawfile.objects.get(file_name=username)
    spark = SparkSession(SparkContext.getOrCreate())
    spark.sparkContext.addFile(row.file.url)

    df = spark.read.csv("file:///" + SparkFiles.get(row.file.name.split("/")[-1]), header=True, inferSchema=True)

    """칼럼 삭제"""
    meta = metadata.objects.filter(file_name=username)

    for row in meta:
        if row.selected == False:
            df = df.drop(row.colname)
        elif row.selected == True:
            if row.nahandling == "remove":
                df = df.na.drop(subset=[row.colname])
    """nahandling이 remove인 칼럼의 rows 삭제"""

    dfPd = df.toPandas()

    csv_buf = StringIO()
    dfPd.to_csv(csv_buf, header=True, index=False)
    csv_buf.seek(0)
    s3.put_object(Bucket='slfolder', Body=csv_buf.getvalue(), Key=filePath)
    csv_buf.close()

    bucket = boto3.resource('s3', aws_access_key_id = settings.AWS_ACCESS_KEY_ID, aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY).Bucket('slfolder')
    for obj in bucket.objects.filter(Prefix=fileDir):
        obj.Acl().put(ACL='public-read')
    spark.stop()

    return redirect("/mlm/mlHome")


def validationCheck(request):
    username = request.session['username']
    MLValidFile = True
    raw_file = rawFileExist(username)
    try:
        metadata.objects.filter(file_name=username).get(target=True)
    except:
        MLValidFile = False

    try:
        naExist = metadata.objects.filter(file_name=username).exclude(navalue=0)
        for row in naExist:
            if row.nahandling == "None":
                MLValidFile = False
    except:
        pass
    if MLValidFile:
        return redirect("/prep/finalfile")
    return render(request, "meta.html", {"metadata": metadata.objects.filter(file_name=username), "raw_file": raw_file,"MLValidFile":MLValidFile})
