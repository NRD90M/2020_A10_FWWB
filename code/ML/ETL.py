# coding: utf-8

"""
Created on 2020年2月25日
@author: Peony
@site: https://github.com/Catsofsuffering
@email: 821621930@qq.com
@file: ETL
@description: 数据清洗
"""

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder


def getPath():
    """
    读取数据目录

    Returns:
        数据当前目录
    """
    return os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))+os.sep+'datas'+os.sep+'Data_FCDS_hashed'


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


def loadData(path, filename):
    """
    读取数据集

    Args:
        path:文件目录
        filename: 文件名
    Returns:
        dataMat: 数据矩阵
    """
    name = path+os.sep+filename
    dataMat = pd.read_csv(name, index_col='entname', encoding='ansi', low_memory=False)
    return dataMat


def dataConcat(dataMats):
    """
    将同一标签的数据拼接在一起

    Args:
        dataMats: 同一个标签的数据矩阵列表
    Returns:
        labelData: 一个标签的数据矩阵
    """
    labelData = pd.DataFrame(columns = ["entname"])
    for dataMat in dataMats:
        labelData = pd.concat([labelData, dataMat], axis=1, sort=True)
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
    scaleData = MinMaxScaler().fit_transform(dataMat)
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
        misValueDatalist: 缺失值处理后的数据矩阵
    """
