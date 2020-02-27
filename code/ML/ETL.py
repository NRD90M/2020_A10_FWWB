# coding: utf-8
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

def getPath():
    """
    读取文件当前目录

    Returns:
        文件当前目录
    """
    return os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def getFilename(path):
    """
    读取文件内所有文件名

    Args:
        path: 文件夹目录
    Returns:
        filenames: 文件名列表
    """
    filenames = []
    filenames = os.listdir(path)
    return filenames

def loadData(path,filename):
    """
    读取数据集

    Args:
        path:文件目录
        filename: 文件名
    Returns:
        dataMat: 数据矩阵
    """
    name = path+filename
    dataMat = pd.read_csv(name, index_col='entname')
    return dataMat

def dataConcat(dataMats):
    """
    将同一标签的数据拼接在一起

    Args:
        dataMats: 同一个标签的数据矩阵列表
    Returns:
        labelData: 一个标签的数据矩阵
    """
    labelData = []
    for i in dataMats:
        labelData = pd.concat([labelData, i], axis=1)
    return labelData

def dataStdScale(dataMat):

    """
    将数据标准化

    Args:
        dataMats: 未进行标准化处理的数据矩阵
    Returns:
        scaleData: 标准化后的数据矩阵
    """
    scaleData = StandardScaler().fit_transform(dataMat)
    return scaleData

def dataMMScale(dataMat):
    
    """
    将数据最值缩放

    Args:
        dataMats: 未进行最值缩放处理的数据矩阵
    Returns:
        scaleData: 最值缩放后的数据矩阵
    """
    scaleData =  MinMaxScaler().fit_transform(dataMat)
    return scaleData

def onehotData(dataMat, colName):
    """
    将定性数据转换为哑编码

    Args:
        dataMat: 文本数据列表
        colName: 需要转换的字段列名
    Returns:a
        onehotDatalist: 哑编码后的数据矩阵
    """
    onehotDatalist = LabelBinarizer().fit_transform(dataMat[colName])
    return onehotDatalist

def labencData(dataMat, colName):
    """
    将定性数据哑编码

    Args:
        dataMat: 文本数据矩阵
        colName: 需要转换的字段列名
    Returns:a
        labencDatalist: 哑编码后的数据矩阵
    """
    labencDatalist = LabelEncoder().fit_transform(dataMat[colName])
    return labencDatalist

def misValueDeal(dataMat):
    """
    缺失值处理

    Args:
        dataMat: 数据矩阵
    Returns:a
        labencDatalist: 缺失值处理后的数据矩阵
    """

def allindex(dataMat,labels):
    """
    算法评价指标

    Args:
        dataMat: 数据矩阵
        lables:  簇的标签集,聚类结果内置
    Returns:a
        NULL
    """
    print("calinski_harabasz_score: %0.3f" % metrics.calinski_harabasz_score(dataMat, labels))   #CH指数，越大聚类效果越好
    print("davies_bouldin_score: %0.3f" % metrics.davies_bouldin_score(dataMat, labels))         #DBI指数，越接近0聚类效果越好
    print("Silhouette_score: %0.3f" % metrics.silhouette_score(dataMat, labels))                  #轮廓系数，越大越好，之前输出失败了
    #print('ROC AUC:%.3f' % roc_auc_score(y_true=y_test, y_score=y_pred2))                           #ROC,需要标签，先删了
    #print('Accuracy:%.3f' % accuracy_score(y_true=y_test, y_pred=y_pred2))                          #AUC，需要标签，先删了

def clustering(clu_kind,dataMat):
    """
    聚类算法

    Args:
        cul_kind: 聚类算法类型
        dataMat:  数据矩阵
    Returns:
        db:聚类结果
    """
    if clu_kind==1 :
        # KMeans聚类
        # n_clusters:簇数量
        db = KMeans(n_clusters=4).fit(dataMat)
    elif clu_kind==2 :
        # DBSCAN聚类
        # eps： ϵϵ-邻域距离，过大簇数减少，过小簇数增多
        # min_samples: 成为核心对象所需要的ϵϵ-邻域的样本数阈值,默认为5
        db = DBSCAN(eps=0.1, min_samples=10).fit_predict(dataMat)
    elif clu_kind==3 :
        # DBSCAN聚类
        # eps： ϵϵ-邻域距离，过大簇数减少，过小簇数增多
        # min_samples: 成为核心对象所需要的ϵϵ-邻域的样本数阈值,默认为5
        db = Birch(n_clusters=None).fit_predict(dataMat)
    elif clu_kind==4:
        #GMM聚类
        #n_components ：即聚类的目标个数
        db=GMM(n_components=4).fit(dataMat)
    return db

def allshow(dataMat,labels):
    """
    可视化算法
  
    Args:
        dataMat:  数据矩阵
        lables:  簇的标签集
    Returns:
        NULL
    """
    pca = PCA(n_components=3) #降维到三维
    pca.fit(dataMat)
    XX = pca.transform(dataMat)
    ax1 = plt.figure().add_subplot(111,projection='3d')
    ax1.scatter3D(XX[:, 0], XX[:, 1],XX[ :, 2], c=labels)  #三维
    #plt.scatter(XX[:, 0], XX[:, 1],c=labels)    #二维
    plt.show()