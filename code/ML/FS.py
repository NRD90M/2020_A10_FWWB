# coding: utf-8

"""
Created on 2020年2月25日
@author: Peony
@site: https://github.com/Catsofsuffering
@email: 821621930@qq.com
@file: FS
@description: 特征选择 
"""
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from scipy.stats import pearsonr
from minepy import MINE
from numpy import array #array函数使用存疑？


def VarSelFS(dataMat, Threshold):
    """
    通过方差选择特征

    Args:
        dataMat: 数据矩阵
        Threshold: 设定的方差阈值
    Returns:
        varselDatalist: 方差选择后的数据矩阵
    """
    varselDatalist = VarianceThreshold(threshold=Threshold).fit_transform(dataMat)
    return varselDatalist

def CorrelFS(dataMat, K):
    """
    通过相关系数法选择特征

    Args:
        dataMat: 数据矩阵
        K: 需要选择的特征个数
    Returns:
        CorrelDatalist: 相关系数法选择后的数据矩阵
    """
    CorrelDatalist = SelectKBest(lambda X, Y: array(map(lambda x:pearsonr(x, Y), X.T)).T, k=K).fit_transform(dataMat.data, dataMat.target)
    return CorrelDatalist

def ChiSeqFS(dataMat, K):
    """
    通过卡方检验选择特征

    Args:
        dataMat: 数据矩阵
        K: 需要选择的特征个数
    Returns:
        ChiSeqDatalist: 卡方检验选择后的数据矩阵
    """
    ChiSeqDatalist = SelectKBest(chi2, k=K).fit_transform(dataMat.data, dataMat.target)
    return ChiSeqDatalist

def MutInfFS(dataMat, K):
    """
    通过互信息法选择特征

    Args:
        dataMat: 数据矩阵
        K: 需要选择的特征个数
    Returns:
        MutInfDatalist: 卡方检验选择后的数据矩阵
    """    
    def mic(x, y):
        m = MINE()
        m.compute_score(x, y)
        return (m.mic(), 0.5)
    
    MutInfDatalist = SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=K).fit_transform(dataMat.data, dataMat.target)
    return MutInfDatalist