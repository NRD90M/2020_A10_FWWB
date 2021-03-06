# coding: utf-8

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
Created on 2020年2月25日
@author: 喋喋不休
@site: https://gitee.com/ding0714
@email: 784700848@qq.com
@file: MS
@description: 模型展示
"""


def AllIndex(dataMat, labels):
    """
    算法评价指标

    Args:
        dataMat: 数据矩阵
        lables:  簇的标签集,聚类结果内置
    Returns:
        NULL
    """
    print("calinski_harabasz_score: %0.3f" %
          metrics.calinski_harabasz_score(dataMat, labels))  # CH指数，越大聚类效果越好
    print("davies_bouldin_score: %0.3f" %
          metrics.davies_bouldin_score(dataMat, labels))     # DBI指数，越接近0聚类效果越好
    print("Silhouette_score: %0.3f" %
          metrics.silhouette_score(dataMat, labels))         # 轮廓系数，越大越好，之前输出失败了
    # print('ROC AUC:%.3f' % roc_auc_score(y_true=y_test, y_score=y_pred2))                           #ROC,需要标签，先删了
    # print('Accuracy:%.3f' % accuracy_score(y_true=y_test, y_pred=y_pred2))                          #AUC，需要标签，先删了

def AllShow(dataMat, labels):
    """
    可视化算法

    Args:
        dataMat:  数据矩阵
        lables:  簇的标签集
    Returns:
        NULL
    """
    #pca降维，估计不太适合文本数据
    # pca = PCA(n_components=3)  # pca降维到三维
    # pca.fit(dataMat)
    # XX = pca.transform(dataMat)

    #lda降维，估计好一点
    lda = LinearDiscriminantAnalysis(n_components=3)
    lda.fit(dataMat)
    XX = lda.transform(dataMat)
    ax1 = plt.figure().add_subplot(111, projection='3d')
    ax1.scatter3D(XX[:, 0], XX[:, 1], XX[:, 2], c=labels)  # 三维
    # plt.scatter(XX[:, 0], XX[:, 1],c=labels)    #二维
    plt.show()
