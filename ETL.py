# coding: utf-8
import pandas as pd
import os
from sklearn import preprocessing

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

def dataScale(dataMat):
    """
    将数据标准化

    Args:
        dataMats: 未进行标准化处理的数据矩阵
    Returns:
        scaleData: 标准化后的数据矩阵
    """
    scaleData = preprocessing.scale(dataMat)
    return scaleData