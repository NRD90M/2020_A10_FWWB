# coding: utf-8
import code.ML.ETL
import code.ML.FS
import code.ML.CA
import code.ML.MS

"""
Created on 2020年3月1日
@author: Peony
@site: https://github.com/Catsofsuffering
@email: 821621930@qq.com
@file: main
@description: 主脚本
"""

def etl():
    """
    数据清洗

    Args:

    Returns:
        etlData:聚类结果
    """

def fs():
    """
    特征选择

    Args:

    Returns:
        fsData:聚类结果
    """    

def ca(clu_kind, dataMat):
    """
    聚类分析

    Args:
        kind: 聚类算法类型
        dataMat:  数据矩阵
    Returns:
        caData:聚类结果
    """
    if clu_kind == 1:

    elif clu_kind == 2:

    elif clu_kind == 3:

    elif clu_kind == 4:

    return caData

def ms():
    """
    模型展示

    Args:

    Returns:

    """    

if __name__ == "__main__":
    etl()
    fs()
    ca()
    ms()