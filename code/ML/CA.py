# coding: utf-8

from sklearn import metrics
from sklearn.cluster import Birch, DBSCAN, KMeans
from sklearn.mixture import GaussianMixture as GMM
"""
Created on 2020年2月25日
@author: 喋喋不休
@site: https://gitee.com/ding0714
@email: 784700848@qq.com
@file: CA
@description: 聚类分析
"""
def KM(dataMat, C):
    """
    K均值聚类

    Args:
        dataMat:  数据矩阵
        C: 簇数量
    Returns:
        KMdata:聚类结果
    """
    KMdata = KMeans(n_clusters=C).fit(dataMat)
    return KMdata

def DBSC(dataMat, EPS, MS=5):
    """
    密度聚类

    Args:
        dataMat:  数据矩阵
        EPS： ϵϵ-邻域距离，过大簇数减少，过小簇数增多
        MS: 成为核心对象所需要的ϵϵ-邻域的样本数阈值,默认为5
    Returns:
        DBSCdata:聚类结果
    """
    DBSCdata = DBSCAN(eps=EPS, min_samples=MS).fit_predict(dataMat)

def BIR(dataMat, CN=None):
    """
    层次聚类

    Args:
        dataMat:  数据矩阵
        CN: 簇数量,默认为None
    Returns:
        BIRdata:聚类结果
    """    
    BIRdata = Birch(n_clusters=CN).fit_predict(dataMat)
    return BIRdata

def GM(dataMat, C):
    """
    高斯混合模型聚类

    Args:
        dataMat:  数据矩阵
        C: 簇数量
    Returns:
        GMdata:聚类结果
    """    
    GMdata = GMM(n_components=C).fit(dataMat)
    return GMdata

