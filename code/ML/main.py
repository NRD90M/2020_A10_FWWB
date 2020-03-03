# coding: utf-8
from ETL import *
from FS import *
from CA import *
from MS import *
from config import *
import os

"""
Created on 2020年3月1日
@author: Peony
@site: https://github.com/Catsofsuffering
@email: 821621930@qq.com
@file: main
@description: 主脚本
"""

DEBUG = True
dataMats = []
N = 4

def etl():
    """
    数据清洗

    Args:

    Returns:
        etlDatas: 数据清洗后，按标签拼接的数据矩阵列表
    """
    path = getPath()
    if DEBUG:
        etlDatas = []
        print('DEBUG1')
        filenames = getFilename(path)
        for filename in filenames:
            print(filename.replace('.csv',''))
    labelDatas = [dataConcat(loadData(path,(''.join([Labelname,'.csv'])))  for Labelname in Labelnames) for Labelnames in Labels]
    return etlDatas


def fs(dataMat, kind, K=4, Threshold=0.8):
    """
    特征选择

    Args:
        dataMat:  数据矩阵
        kind: 聚类算法类型
        K: 需要选择的特征个数，默认为4
        Threshold: 设定的方差阈值，默认为0.8
    Returns:
        fsData:聚类结果
    """
    if kind == 0:
        fsData = VarSelFS(dataMat, Threshold)
    elif kind == 1:
        fsData = CorrelFS(dataMat, K)
    elif kind == 2:
        fsData = ChiSeqFS(dataMat, K)
    else:
        fsData = MutInfFS(dataMat, K)
    return fsData


def ca(dataMat, kind, EPS, C=4, MS=5, CN=NONE):
    """
    聚类分析

    Args:
        dataMat:  数据矩阵
        kind: 聚类算法类型
        C: 簇数量,默认为4
        EPS： ϵϵ-邻域距离，过大簇数减少，过小簇数增多
        MS: 成为核心对象所需要的ϵϵ-邻域的样本数阈值,默认为5
        CN: 簇数量,默认为None
    Returns:
        caData:聚类结果
    """
    if kind == 0:
        caData = KM(dataMat, C)
    elif kind == 1:
        caData = DBSC(dataMat, EPS, MS)
    elif kind == 2:
        caData = BIR(dataMat, CN)
    else:
        caData = GM(dataMat, C)
    return caData

'''
def ms():
    """
    模型展示

    Args:

    Returns:

    """ 

'''   

if __name__ == "__main__":
    if DEBUG:
        etlDatas = etl()
        for i in range(N):
            fsData = fs(etlDatas[i], i, i, i/N)
            caData = ca(fsData, i, i/N, i, i, NONE)
    else:
        etl()
        fs()
        ca()
        ms()