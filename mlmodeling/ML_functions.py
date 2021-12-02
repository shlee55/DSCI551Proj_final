'''
1. file load(csv format) - exploration
2. metadata extraction + exploration
+ (clean data)
3. feature extraction + exploration - (*use Spark)
4. store feature in database
5. create multiple prediction method for users to select
6. store result in database
'''

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#from PIL import Image
#from PIL.ExifTags import TAGS

def file_load():
    '''
    :param file_path:
    :return: dataframe of the file if input is csv or image file if input is image file
    '''
    file_path = input("file path: ")

    if file_path[-4:] == '.csv':
        '''file file path ends with "csv" run this'''
        print("csv file inserted")
        sep = input('separator? (ex "," or ";") :')
        df = pd.read_csv(file_path, sep=sep)
        return df
"""
    if file_path[-4:] == '.jpg' or file_path[-4:] == 'jpeg' or file_path[-4:] == ' .png':
        '''file file path ends with "jpg,jpeg, or png" run this'''
        print("image file inserted")
        image = Image.open(file_path)
        return image
"""

"""
def image_metadata(image):
    '''
    https://www.thepythoncode.com/article/extracting-image-metadata-in-python
    :param image:
    :return:
    '''
    exif_data = image.getexif()
    for tag_id in exif_data:
        tag = TAGS.get(tag_id,tag_id)
        data = exif_data.get(tag_id)

        if isinstance(data, bytes):
            data = data.decode()
        print(f"{tag:25}: {data}")
        #print(tag, " : ", data)
"""

class preprocessing:
    '''
    class preprocessing consist of 9 functions
    print_metadata()                : print metadata (number of rows, variable type(categorical vs continuous variables),
                                                      range of continuous variable and the list of categorical variable)
    obj_to_flo(column_name)         : change object type variable to float type variable
    flo_to_obj(column_name)         : change float type variable to object type variable
    sep_disc_cont()                 : distinguish categorical variables from continuous variables
    na_values()                     : check the amount of empty sells of each column
    rm_rows(rm_columns)             : remove rows with empty sells of specific column
    select_columns(columns)         : select columns to use for next step
    disc_plot(legend_var, x_axis)   : plot
    disc_freq_plot(disc_var)        : plot
    '''

    def __init__(self,df):

        self.df = df
        self.pred_var = "" # target (predict) variable
        self.disc_count = 0 # number of discrete variables (categorical variables)
        self.cont_count = 0 # number of continuous variables
        self.disc_list = []
        self.cont_list = []
        self.meta_list = []

    def select_target(self):
        print(self.df.columns.to_series().groupby(self.df.dtypes).groups)
        tar_var = False
        while tar_var == False:
            pred_var = input("target variable (must be categorical variable): ")
            try:
                if self.df[pred_var].dtypes == "object":
                    tar_var = True
                else:
                    con_to_cat = input("Must be categorical variable, will you change this value into categorical variable? (yes/no)")
                    if con_to_cat == "yes":
                        self.flo_to_obj(pred_var)
                        tar_var = True
                    else:
                        print("try other variable")

            except:
                print("error: you should type one of the column names")
        self.pred_var = pred_var

    def print_metadata(self):
        '''metadata 나열하기'''
        print("The table has {} rows".format(self.df.shape[0]))
        column_list = self.df.columns
        self.disc_list = []
        self.cont_list = []
        self.meta_list = []
        for i_var in column_list:
            if self.df[i_var].dtypes == "object":
                self.disc_list.append(i_var)
                self.meta_list.append("Categorical Variable: \t {} is consist of {}".format(i_var, self.df[i_var].unique()))
            else:
                self.cont_list.append(i_var)
                self.meta_list.append("Continuous Variable: \t {} has range from {} to {} and mean is {}".format(i_var, self.df[i_var].min(), self.df[i_var].max(), round(self.df[i_var].mean(), 2)))
        self.disc_count = len(self.disc_list)
        self.cont_count = len(self.cont_list)

    def obj_to_flo(self, column_name):
        '''categorical variable을 continuous variable로 만들기'''
        try:
            self.df[column_name] = self.df[column_name].astype(float, errors='raise')
        except:
            print("this variable is not changeable to continuous variable. (not a number variable)")
        self.sep_disc_cont()

    def flo_to_obj(self, column_name):
        '''continuous variable을 categorical variable로 만들기'''
        self.df[column_name] = self.df[column_name].astype(str)
        self.df = self.df.replace('nan', np.nan)
        self.sep_disc_cont()

    def sep_disc_cont(self):
        '''categorical variable과 continuous variable 분류 작업'''
        column_list = self.df.columns
        self.disc_list = []
        self.cont_list = []
        for i_var in column_list:
            if i_var == self.pred_var:
                continue
            elif self.df[i_var].dtypes == "object":
                self.disc_list.append(i_var)
            else:
                self.cont_list.append(i_var)
        self.disc_count = len(self.disc_list)
        self.cont_count = len(self.cont_list)

    def na_values(self):
        '''table에서 N/A 찾기'''
        print(self.df.isna().sum())

    def rm_rows(self, rm_columns=[]):
        '''column value에 N/A 있는 row 제거'''
        for col in rm_columns:
            self.df = self.df.loc[self.df[col].notna(), ]

    def select_columns(self):
        '''machine learning에 사용할 columns들만 남기기'''
        columns = [self.pred_var]
        done=True
        while done==True:
            column = input('choose column needed one by one, if done, type "done" : ')
            if column in self.df.columns:
                columns.append(column)
            elif column == self.pred_var:
                continue
            elif column == "done":
                done = False
                print("End")
            else:
                print("wrong column name, try different name")
        columns = list(set(columns))
        self.df = self.df[columns]
        self.sep_disc_cont()

"""
    def disc_plot(self, legend_var, x_axis):  ## discrete variable과 predict variable 묶은 plot
        '''각 variable 마다 plot 만들기'''
        df_agg = self.df.loc[:, [x_axis, legend_var]].groupby(legend_var)
        print(self.df[legend_var].value_counts())
        vals = [tb[x_axis].values.tolist() for i, tb in df_agg]
        plt.figure(figsize=(16, 9), dpi=80)
        colors = [plt.cm.Spectral(i / float(len(vals) - 1)) for i in range(len(vals))]
        print(colors)
        n, bins, patches = plt.hist(vals, self.df[x_axis].unique().__len__(), stacked=True,
                                    density=False)
        plt.legend({group: col for group, col in zip(np.unique(self.df[legend_var]).tolist(), colors[:len(vals)])})
        plt.title(f"Histogram of ${x_axis}$ colored by ${legend_var}$", fontsize=22)
        plt.xlabel(x_axis)
        plt.ylabel("Frequency")
        plt.show()

    def disc_freq_plot(self, disc_var):  ## discrete variable 단독 frequency 확인 테이블
        fig, ax = plt.subplots()
        self.df[disc_var].value_counts().plot(ax=ax, kind='bar', xlabel=disc_var, ylabel='frequency')
        plt.show()

"""

def compare_ml(df, pred_var, disc_list, cont_list):
    '''

    :param df: table
    :param pred_var: target (predict) variable
    :param disc_list: list of the name of the categorical variable
    :param cont_list: list of the name of the continuous variable
    :return: no return. compare the accuracy score of Logistic Regression and Random Forest
    '''
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

    pipeline_lr.fit(X, y)#################################
    pd.to_pickle(pipeline_lr,r'D:/USC/COURSES/3.Fall2021/DSCI551/Project/Django_study/lr_model.pickle')
    pipeline_rf.fit(X, y)#################################
    pd.to_pickle(pipeline_rf,r'D:/USC/COURSES/3.Fall2021/DSCI551/Project/Django_study/rf_model.pickle')

    score_lr = cross_val_score(pipeline_lr,X,y,cv=5,scoring='accuracy').mean()
    print("score of logistic regression: ", score_lr)
    score_rf = cross_val_score(pipeline_rf,X,y,cv=5,scoring='accuracy').mean()
    print("score of random forest: ", score_rf)
    return(score_lr,score_rf)

class run_lr:
    def __init__(self, df, pred_var):
        self.df = df
        self.pred_var = pred_var
        self.X = df.drop(pred_var, axis='columns')
        self.y = df[pred_var]
        self.lr_model = ''
        self.lr_result = ''
        self.pipeline_lr = ''

    def training_lr(self):
        disc_list, cont_list = self.sep_disc_cont()
        col_transform = make_column_transformer(
            (OneHotEncoder(), disc_list),
            (StandardScaler(with_mean=False), cont_list)
        )

        self.lr_model = LogisticRegression(multi_class='ovr', max_iter=5000, random_state=1)
        self.pipeline_lr = make_pipeline(col_transform, self.lr_model)
        self.pipeline_lr.fit(self.X ,self.y)

    def pred_one_row_lr(self):
        pred_list = []
        for col in self.X.columns:
            right_value = False
            while right_value == False:
                try:
                    if self.X[col].dtype == 'O':
                        print("{}\npossible value(object type):\t{}".format(col, sorted(self.X[col].unique())))
                        col_val = input(col + ": ")
                        if col_val in self.X[col].unique():
                            pred_list.append(col_val)
                            right_value = True
                        else:
                            print("error: incorrect value")
                    elif self.X[col].dtype == 'int64':
                        print("{}\npossible value(integer type): any value".format(col))
                        col_val = int(input(col + ": "))
                        pred_list.append(col_val)
                        right_value = True
                    elif self.X[col].dtype == 'float64':
                        print("{}\npossible value(float type): any value".format(col))
                        col_val = float(input(col + ": "))
                        pred_list.append(col_val)
                        right_value = True
                except:
                    print("error: incorrect value")
        print("your input:", pred_list)
        title_list = self.X.columns.tolist()
        X_pred = pd.DataFrame([pred_list], columns=title_list)
        return X_pred

    def predict_lr(self, variable_tb):
        self.lr_result = self.pipeline_lr.predict(variable_tb)
        print("your output:",self.lr_result)
        return self.lr_result

    def sep_disc_cont(self):
        column_list = self.df.columns
        disc_list = []
        cont_list = []
        for i_var in column_list:
            if i_var == self.pred_var:
                continue
            elif self.df[i_var].dtypes == "object":
                disc_list.append(i_var)
            else:
                cont_list.append(i_var)
        return (disc_list, cont_list)

class run_rf:
    def __init__(self, df, pred_var):
        self.df = df
        self.pred_var = pred_var
        self.X = df.drop(pred_var, axis='columns')
        self.y = df[pred_var]
        self.rf_model = ''
        self.rf_result = ''
        self.pipeline_rf = ''

    def training_rf(self):
        disc_list, cont_list = self.sep_disc_cont()
        col_transform = make_column_transformer(
            (OneHotEncoder(), disc_list),
            (StandardScaler(with_mean=False), cont_list)
        )

        self.rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=1)
        self.pipeline_rf = make_pipeline(col_transform, self.rf_model)
        self.pipeline_rf.fit(self.X, self.y)

    def pred_one_row_rf(self, row_list=[]):
        pred_list = []
        for col in self.X.columns:
            right_value = False
            while right_value == False:
                try:
                    if self.X[col].dtype == 'O':
                        print("{}\npossible value(object type):\t{}".format(col, sorted(self.X[col].unique())))
                        col_val = input(col + ": ")
                        if col_val in self.X[col].unique():
                            pred_list.append(col_val)
                            right_value = True
                        else:
                            print("error: incorrect value")
                    elif self.X[col].dtype == 'int64':
                        print("{}\npossible value(integer type): any value".format(col))
                        col_val = int(input(col + ": "))
                        pred_list.append(col_val)
                        right_value = True
                    elif self.X[col].dtype == 'float64':
                        print("{}\npossible value(float type): any value".format(col))
                        col_val = float(input(col + ": "))
                        pred_list.append(col_val)
                        right_value = True
                except:
                    print("error: incorrect value")
        print("your input:",pred_list)
        title_list = self.X.columns.tolist()
        X_pred = pd.DataFrame([pred_list], columns=title_list)
        return X_pred

    def predict_rf(self, variable_tb):
        self.rf_result = self.pipeline_rf.predict(variable_tb)
        print("your output:",self.rf_result)
        return self.rf_result

    def sep_disc_cont(self):
        column_list = self.df.columns
        disc_list = []
        cont_list = []
        for i_var in column_list:
            if i_var == self.pred_var:
                continue
            elif self.df[i_var].dtypes == "object":
                disc_list.append(i_var)
            else:
                cont_list.append(i_var)
        return (disc_list, cont_list)






'''
#https://www.youtube.com/watch?v=_3xj9B0qqps
#https://www.youtube.com/watch?v=u2gLTcL4YlI
#https://www.youtube.com/watch?v=zcALUNZNBUk
#https://www.youtube.com/watch?v=tJbkJDKcZ3w
#https://www.youtube.com/watch?v=CHefrTGsDAM
'''