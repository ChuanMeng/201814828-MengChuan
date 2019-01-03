# -*- coding: utf-8 -*-
"""
1.先读取文档的tf-idf表示与聚类的label
2.调用sklearn封装好的方法进行聚类
3.调用sklearn封装的评测方法进行NMI、AMI评测
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import SpectralClustering  
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn import mixture
from time import time
import json

def data_load():  #读取文档的tf-idf表示
    data=np.loadtxt("embeding/tf-idf.txt")
    return data

def label_load():  #读取实现给定的聚类label
    label_true=[]
    with open('label.json') as f:
        label_true=json.load(f)
    return label_true    
'''
以下为聚类算法的实现
'''
def k_means(data):
    t0 = time()
    estimator = KMeans(n_clusters=50)#构造聚类器
    estimator.fit(data)#聚类
    label_pred = estimator.labels_ #获取聚类标签
    time_use=time()-t0
    return label_pred,time_use

def Affinity_Propagation(data):
    t0 = time()
    af = AffinityPropagation(damping=0.5,preference=-5.74).fit(data)
    cluster_centers_indices = af.cluster_centers_indices_
    label_pred= af.labels_
    time_use=time()-t0
    n_clusters= len(cluster_centers_indices)
    #print(n_clusters)
    return label_pred,time_use

def Mean_shift(data):
    t0 = time()
    bandwidth=0.9
    ms = MeanShift(bandwidth=bandwidth,bin_seeding=True,cluster_all=True)
    ms.fit(data)
    label_pred = ms.labels_
    labels_unique = np.unique(label_pred)
    #n_clusters = len(labels_unique)
    time_use=time()-t0
    
    #print("bandwidth参数设置为：",bandwidth)
    #print("得到的cluster数目为：",n_clusters)
    return label_pred,time_use

def Spectral_clustering(data):
    t0 = time()
    sc=SpectralClustering(gamma=0.1,n_clusters=70).fit(data) 
    label_pred =sc.labels_
    time_use=time()-t0
    
    return label_pred,time_use

def Ward_hierarchical_clustering(data):
    t0 = time()
    whc = AgglomerativeClustering(affinity='euclidean',n_clusters=80, linkage='average').fit(data)
    label_pred=whc.labels_
    time_use=time()-t0
    
    return label_pred,time_use


def DBSCAN_(data):
    t0 = time()
    db=DBSCAN(eps = 1.1, min_samples =1).fit(data)
    label_pred=db.labels_
    time_use=time()-t0
    n_clusters = len(set(label_pred))
    #print("聚类得到的cluster数目为：",n_clusters)
    return label_pred,time_use


def Gaussian_mixtures(data):
    t0 = time()
    gmm=mixture.GaussianMixture(n_components=70,covariance_type='spherical').fit(data)
    label_pred=gmm.predict(data)
    time_use=time()-t0
    
    return label_pred,time_use

def birch(data):
    t0 = time()
    label_pred = Birch(n_clusters = 70,threshold = 0.4, branching_factor =70).fit_predict(data)
    time_use=time()-t0
    
    return label_pred,time_use


def evaluation(label_pred,label_true): #用NMI来对聚类结果进行评测
    AMI=metrics.adjusted_mutual_info_score(label_pred,label_true)  
    NMI=metrics.normalized_mutual_info_score(label_pred,label_true)
    return AMI,NMI


if __name__ == "__main__": 

    
    tuple1=k_means(data_load())
    KNN_result=evaluation(tuple1[0],label_load())
    print("KNN聚类结果的NMI为:%f，运行时间为%f." %(KNN_result[1],tuple1[1]))
    print("KNN聚类结果的AMI为:%f，运行时间为%f." %(KNN_result[0],tuple1[1]))
    print("--------------------------------------------------------------------------------")
    tuple1=Affinity_Propagation(data_load())
    AP_result=evaluation(tuple1[0],label_load())
    print("Affinity_Propagation聚类结果的NMI为:%f，运行时间为%f." %(AP_result[1],tuple1[1]))
    print("Affinity_Propagation聚类结果的AMI为:%f，运行时间为%f." %(AP_result[0],tuple1[1]))
    print("--------------------------------------------------------------------------------")
    tuple1=Mean_shift(data_load())
    Mean_shift_result=evaluation(tuple1[0],label_load())
    print("Mean_shift聚类结果的NMI为:%f，运行时间为%f." %(Mean_shift_result[1],tuple1[1]))
    print("Mean_shift聚类结果的AMI为:%f，运行时间为%f." %(Mean_shift_result[0],tuple1[1]))
    print("--------------------------------------------------------------------------------")
    tuple1=Spectral_clustering(data_load())
    Spectral_clustering_result=evaluation(tuple1[0],label_load())
    print("Spectral_clustering聚类结果的NMI为:%f，运行时间为%f." %(Spectral_clustering_result[1],tuple1[1]))
    print("Spectral_clustering聚类结果的AMI为:%f，运行时间为%f." %(Spectral_clustering_result[0],tuple1[1]))
    print("--------------------------------------------------------------------------------")
    tuple1=Ward_hierarchical_clustering(data_load())
    Ward_hierarchical_clustering_result=evaluation(tuple1[0],label_load())
    print("Ward_hierarchical_clustering聚类结果的NMI为:%f，运行时间为%f." %(Ward_hierarchical_clustering_result[1],tuple1[1]))
    print("Ward_hierarchical_clustering聚类结果的AMI为:%f，运行时间为%f." %(Ward_hierarchical_clustering_result[0],tuple1[1]))
    print("--------------------------------------------------------------------------------")
    tuple1=DBSCAN_(data_load())
    DBSCAN_result=evaluation(tuple1[0],label_load())
    print("DBSCAN聚类结果的NMI为:%f，运行时间为%f." %(DBSCAN_result[1],tuple1[1]))
    print("DBSCAN聚类结果的AMI为:%f，运行时间为%f." %(DBSCAN_result[0],tuple1[1]))
    print("--------------------------------------------------------------------------------")
    tuple1=Gaussian_mixtures(data_load())
    Gaussian_mixtures_result=evaluation(tuple1[0],label_load())
    print("Gaussian_mixtures聚类结果的NMI为:%f，运行时间为%f." %(Gaussian_mixtures_result[1],tuple1[1]))
    print("Gaussian_mixtures聚类结果的AMI为:%f，运行时间为%f." %(Gaussian_mixtures_result[0],tuple1[1]))
    print("--------------------------------------------------------------------------------")
    tuple1=birch(data_load())
    birch_result=evaluation(tuple1[0],label_load())
    print("birch聚类结果的NMI为:%f，运行时间为%f." %(birch_result[1],tuple1[1]))
    print("birch聚类结果的AMI为:%f，运行时间为%f." %(birch_result[0],tuple1[1]))
    

    