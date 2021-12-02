import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
import seaborn as sns
import pandas as pd

def get_graph():
    buf = BytesIO()
    plt.savefig(buf,format='png')
    buf.seek(0)
    image_png = buf.getvalue()
    graph = base64.b64encode(image_png)
    graph = graph.decode('utf-8')
    buf.close()
    return graph


def get_plot(x,y):
    plt.switch_backend('AGG')
    plt.figure(figsize = (10,5))
    plt.title('this is my chart')
    plt.plot(x,y)
    plt.xticks(rotation = 45)
    plt.xlabel('item')
    plt.ylabel('price')
    plt.tight_layout()
    graph = get_graph()
    return graph


def barChart(dict_ex,x):
    plt.switch_backend('AGG')

    labels = ""
    Xs = []
    sumValue = []
    for key in dict_ex.keys():
        if key == x:
            labels = key
            labLoc = np.arange(len(dict_ex[labels])) # label location
        else:
            Xs.append(key)
            sumValue.append(dict_ex[key])
            print("Xs:",Xs)
            print("sumValue:", sumValue)
    if len(Xs) > 1:
        sumList = np.sum(sumValue, axis=0)
    else:
        sumList = dict_ex[Xs[0]]
    width = 0.5  # the width of the bars

    fig, ax = plt.subplots(figsize=(12,5))

    rects = ax.bar(labLoc, sumList, width)
    ax.bar_label(rects, padding=3)

    ax.set_xlabel(x)
    ax.set_ylabel("count")
    ax.set_title('BAR CHART: number count of {}'.format(x))
    ax.set_xticks(labLoc, dict_ex[labels])

    fig.tight_layout()
    graph = get_graph()
    return graph

def groupBarChart(dict_ex,x,leg):
    plt.switch_backend('AGG')
#    dict_ex = {"labels":['G1', 'G2', 'G3', 'G4', 'G5'],"men":[20, 34, 30, 35, 27],"women":[25, 32, 34, 20, 25],"baby":[3,2,4,1,5]}
    labels = ""
    print("barchart:1")
    Xs = []
    print("barchart:2")
    for key in dict_ex.keys():
        if key == x:
            print("barchart:3")
            labels = key
            labLoc = np.arange(len(dict_ex[labels])) # label location
        else:
            print("barchart:4")
            Xs.append(key)
    print("barchart:5")
    width = 0.3 # the width of the bars
    print("labels:",labels)
    print("barchart:6")
    print("Xs list:", Xs)
    fig, ax = plt.subplots(figsize=(12,5))
    barSteps = width / len(Xs)

    i=0
    for X in Xs:
        rects = ax.bar(labLoc + 1.2 * i * barSteps , dict_ex[X], width * 2 /len(Xs), label=X)
        ax.bar_label(rects, padding=3)
        i += 1

    ax.set_xlabel(x)
    ax.set_ylabel("count")
    ax.set_title('GROUPED BAR CHART: number count of {} by {}'.format(x,leg))
    ax.set_xticks(labLoc, dict_ex[labels])
    ax.legend(loc = "upper left", title=leg)

    fig.tight_layout()
    graph = get_graph()
    return graph


def boxPlot(dict_ex,x,y):
#    dict_ex = {"labels": ['G1', 'G2', 'G3', 'G4', 'G5','G1', 'G2', 'G3', 'G4', 'G5','G1', 'G2', 'G3', 'G4', 'G5','G1', 'G2', 'G3', 'G4', 'G5'], "men": [20, 34, 30, 35, 2,27,13,24,35,213,12,123,1,2,42,1,23,12,31,13], "women": [25, 32, 34, 20, 25,123,42,35,46,74,24,12,42,12,32,2,5,34,234,345]}
    df = pd.DataFrame(dict_ex)
    plt.switch_backend('AGG')
    fig, ax = plt.subplots(figsize=(12,5))
    sns.violinplot(x=x, y=y, data=df, sscale='width',inner='quartile')
#    sns.boxplot(x=x, y=y, data=df, notch=False)
    plt.title('BOX PLOT: Box Plot of {} by {}'.format(y,x), fontsize=22)
#    plt.ylim(10, 40)
    fig.tight_layout()
    graph = get_graph()
    return graph

def densityCurvesWithHistogram(dict_ex,x,y):
#    dict_ex = {"labels": ['G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2'], "men": [20, 34, 30, 35, 2,27,13,24,35,213,12,123,1,2,42,1,23,12,31,13], "women": [25, 32, 34, 20, 25,123,42,35,46,74,24,12,42,12,32,2,5,34,234,345]}
    df = pd.DataFrame(dict_ex)
    plt.switch_backend('AGG')
    fig, ax = plt.subplots(figsize=(12,5))


    mylist = list(dict.fromkeys(dict_ex[x]))
    print(mylist)

    for value in mylist:
        sns.distplot(df.loc[df[x] == value, y], label=value, hist_kws={'alpha': .3}, kde_kws={'linewidth': 2})

    plt.title('DENSITY CURVES with HISTOGRAM: Density Plot of {} by {}'.format(y,x), fontsize=22)
    plt.legend()

    fig.tight_layout()
    graph = get_graph()
    return graph


def histogram(dict_ex,y):
#    dict_ex = {"labels": ['G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2', 'G3', 'G1', 'G2'], "men": [20, 34, 30, 35, 2,27,13,24,35,213,12,123,1,2,42,1,23,12,31,13], "women": [25, 32, 34, 20, 25,123,42,35,46,74,24,12,42,12,32,2,5,34,234,345]}
    df = pd.DataFrame(dict_ex)
    plt.switch_backend('AGG')
    fig, ax = plt.subplots(figsize=(12,5))

    sns.distplot(df.loc[:, y], hist_kws={'alpha': .3}, kde_kws={'linewidth': 2})
    plt.title('HISTOGRAM: Histogram of {}'.format(y), fontsize=22)
    plt.legend()

    fig.tight_layout()
    graph = get_graph()
    return graph
#categorical
#Grouped bar chart with labels

# continuous https://www.machinelearningplus.com/plots/top-50-matplotlib-visualizations-the-master-plots-python/
# Density Plot,
# 26. Box Plot
# Density Curves with Histogram
## https://www.youtube.com/watch?v=jrT6NiM46jk&t=756s