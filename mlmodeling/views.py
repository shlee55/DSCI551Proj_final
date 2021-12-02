from django.shortcuts import render, redirect
from preprocessing.models import Rawfile, metadata, MLscore, predictTable
from pyspark.sql.session import SparkSession
from pyspark.context import SparkContext
from pyspark import SparkFiles
from pyspark.sql.types import *
from django.conf import settings
import pandas as pd
from io import StringIO, BytesIO
import pickle
import boto3
import boto
import webbrowser
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def mlHome(request):
    username = request.session['username']
    MLValidFile = True
    try:
        metadata.objects.filter(file_name=username).get(target=True).colname
    except:
        MLValidFile = False
    return render(request, "mlHome.html", {"MLValidFile" : MLValidFile})

def downloadFile(request):
    username = request.session['username']
    fileDir = "media/{}".format(username)
    filePath = "{}/{}_finalfile.csv".format(fileDir, username)
    if request.method == "POST":
        try:
            key = filePath
            url = f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{key}"
            webbrowser.open(url)  # Go to example.com
        except:  ## if file can't be transformed to dataframe, remove sdata
            pass
    return redirect("/mlm/mlHome")

def compareml(request):
    username = request.session['username']
    fileDir = "media/{}".format(username)
    filePath = "{}/{}_finalfile.csv".format(fileDir, username)
    picklePath_lr = "{}/{}_lr_model.pickle".format(fileDir, username)
    picklePath_rf = "{}/{}_rf_model.pickle".format(fileDir, username)

    pred_var = metadata.objects.filter(file_name=username).get(target=True).colname
    disc_list = list(metadata.objects.filter(file_name=username, datatype="Categorical", selected=True, target=False).values_list('colname', flat=True))
    cont_list = list(metadata.objects.filter(file_name=username, datatype="Continuous", selected=True, target=False).values_list('colname', flat=True))

    s3 = boto3.client('s3', aws_access_key_id = settings.AWS_ACCESS_KEY_ID, aws_secret_access_key = settings.AWS_SECRET_ACCESS_KEY)
    key = filePath
    url = f"https://{settings.AWS_S3_CUSTOM_DOMAIN}/{key}"
    spark = SparkSession(SparkContext.getOrCreate())
    spark.sparkContext.addFile(url)
    dfSp = spark.read.csv("file:///" + SparkFiles.get(key.split("/")[-1]), header=True, inferSchema=True)
    df = dfSp.toPandas()
    MLscore.objects.filter(file_name=username).delete()
    for column in disc_list:
        typeOfData = dfSp.schema[column].dataType  ## type of variable ###########################
        if isinstance(typeOfData, IntegerType) or \
                        isinstance(typeOfData, FloatType) or isinstance(typeOfData, DoubleType): ## numeric variable인지 확인 ####################df[column].toPandas().dtypes != "object":
            df[column] = df[column].astype(str)

    for column in cont_list:
        typeOfData = dfSp.schema[column].dataType  ## type of variable ###########################
        if not isinstance(typeOfData, IntegerType) and \
                        not isinstance(typeOfData, FloatType) and not isinstance(typeOfData, DoubleType):##########df[column].dtypes == "object":
            df[column] = df[column].astype(float)

    X = df.drop(pred_var, axis='columns')
    y = df[pred_var]
    col_transform = make_column_transformer(
        (OneHotEncoder(), disc_list),
        (StandardScaler(with_mean=False), cont_list)
    )

    lr = LogisticRegression(multi_class='ovr',max_iter=5000,random_state=1)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=1)

    pipeline_lr = make_pipeline(col_transform, lr)
    pipeline_rf = make_pipeline(col_transform, rf)

    score_lr = cross_val_score(pipeline_lr,X,y,cv=5,scoring='accuracy').mean()
    score_rf = cross_val_score(pipeline_rf,X,y,cv=5,scoring='accuracy').mean()

    pipeline_lr.fit(X, y)#################################
    pipeline_rf.fit(X, y)#################################

    pklPath = BytesIO()
    pd.to_pickle(pipeline_lr,pklPath)
    pklPath.seek(0)
    s3.put_object(Bucket='slfolder', Body=pklPath.getvalue(), Key=picklePath_lr)
    pklPath.close()

    pklPath = BytesIO()
    pd.to_pickle(pipeline_rf,pklPath)
    pklPath.seek(0)
    s3.put_object(Bucket='slfolder', Body=pklPath.getvalue(), Key=picklePath_rf)
    pklPath.close()

    MLs = MLscore(modelType = "Logistic Regression",score = round(score_lr,4),file_name=username)
    MLs.save()
    MLs = MLscore(modelType = "Random Forest",score = round(score_rf,4),file_name=username)
    MLs.save()
    spark.stop()
    return redirect("/mlm/validationScore")

def validationScore(request):
    username = request.session['username']
    try:
        validScore_lr = MLscore.objects.filter(file_name=username).get(modelType="Logistic Regression")
        validScore_rf = MLscore.objects.filter(file_name=username).get(modelType="Random Forest")
        MLValidFile = True
    except:
        validScore_lr = ""
        validScore_rf = ""
        MLValidFile = False
    predTable = predictTable.objects.filter(file_name=username)
    context = {"score_lr": validScore_lr, "score_rf": validScore_rf, "MLValidFile":MLValidFile, "predTable":predTable}
    return render(request, "compareml.html", context)



def selectml(request):
    return render(request, "selectml.html")

def predict_lr(request):
    username = request.session['username']
    fileDir = "media/{}".format(username)
    picklePath_lr = "{}/{}_lr_model.pickle".format(fileDir, username)
    wrongInput = False
    cat_dict = {}
    cont_dict = {}
    result = ""
    targetV = metadata.objects.filter(file_name=username).get(target=True).colname
    meta = metadata.objects.filter(file_name=username, selected=True, target=False)

    for col in meta:
        if col.datatype == "Categorical":
            cat_dict[col.colname] = [x.strip() for x in col.catValues.split(",")]
        else:
            cont_dict[col.colname] = col.contValues

    if request.method == "POST":
        dict_var= {}
        for col in meta:
            value = request.POST.get(col.colname)
            dict_var[col.colname] = [value]
            if value is None or value == "":
                wrongInput = True
                context = {"meta": meta, "cat_dict": cat_dict, "cont_dict": cont_dict, "result": result, "targetV": targetV, "wrongInput": wrongInput, "wrongInput": wrongInput}
                return render(request, "predict_lr.html", context)
        try:
            s3 = boto3.resource('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
            model = pickle.loads(s3.Bucket("slfolder").Object(picklePath_lr).get()['Body'].read())
            X_pred = pd.DataFrame.from_dict(dict_var)
            result = model.predict(X_pred)
        except:  ## if file can't be transformed to dataframe, remove sdata
            pass
        predResult = predictTable(modelType = "Logistic Regression", indepVariable = dict_var, targetVariable = result[0], file_name = username)
        predResult.save()
    context = {"meta":meta,"cat_dict":cat_dict,"cont_dict":cont_dict, "result":result,"targetV":targetV, "wrongInput":wrongInput}
    return render(request, "predict_lr.html",context)


def predict_rf(request):
    username = request.session['username']
    fileDir = "media/{}".format(username)
    picklePath_rf = "{}/{}_rf_model.pickle".format(fileDir, username)
    wrongInput = False
    cat_dict = {}
    cont_dict = {}
    result = ""
    targetV = metadata.objects.filter(file_name=username).get(target=True).colname
    meta = metadata.objects.filter(file_name=username, selected=True, target=False)

    for col in meta:
        if col.datatype == "Categorical":
            cat_dict[col.colname] = [x.strip() for x in col.catValues.split(",")]
        else:
            cont_dict[col.colname] = col.contValues

    if request.method == "POST":
        dict_var= {}
        for col in meta:
            value = request.POST.get(col.colname)
            dict_var[col.colname] = [value]
            if value is None or value == "":
                wrongInput = True
                context = {"meta": meta, "cat_dict": cat_dict, "cont_dict": cont_dict, "result": result, "targetV": targetV, "wrongInput":wrongInput}
                return render(request, "predict_rf.html", context)
        try:
            s3 = boto3.resource('s3', aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY)
            model = pickle.loads(s3.Bucket("slfolder").Object(picklePath_rf).get()['Body'].read())
            X_pred = pd.DataFrame.from_dict(dict_var)
            result = model.predict(X_pred)
        except:  ## if file can't be transformed to dataframe, remove sdata
            pass
        predResult = predictTable(modelType = "Random Forest", indepVariable = dict_var, targetVariable = result[0], file_name = username)
        predResult.save()
    context = {"meta":meta,"cat_dict":cat_dict,"cont_dict":cont_dict, "result":result,"targetV":targetV, "wrongInput":wrongInput}
    return render(request, "predict_rf.html",context)