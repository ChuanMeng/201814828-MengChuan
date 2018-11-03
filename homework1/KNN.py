# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:24:52 2018

@author: MengChuan
"""
import numpy as np


def doProcess():
    
    train=np.loadtxt("embeding/trainset embeding.txt")  
    test=np.loadtxt("embeding/testset embeding.txt")
        
    
    train_label_list=[] #训练集的label list
    train_label_value={}
    train_label=open('index_train or test set'+'/'+'trainset.txt','r')
    for i,item in enumerate(train_label.readlines()):
        new_item=item.strip('\n')
        cate,doc=new_item.split('_')
        train_label_list.append(cate)
        train_label_value[new_item]=train[i]
    train_label.close()
    print(train_label_value)    
    
    
    test_label_list=[] #测试集的label list   
    test_label=open('index_train or test set'+'/'+'testset.txt','r')
    for item in test_label.readlines():
        new_item=item.strip('\n')
        cate,doc=new_item.split('_')
        test_label_list.append(cate)
    test_label.close()
    
    N=len(test)
    count = 0
    for i,test_item in enumerate(test):
        classifyResult = KNNComputeCate(train_label_value,test_item,25)
        if classifyResult == test_label_list[i]:
            count=count+1
        print(classifyResult == test_label_list[i])
        print("正在处理第%d条测试数据" %(i+1))
            
    accuracy = float(count)/float(N) 
    print("测试集的精确度是：%f" % accuracy)       


def KNNComputeCate(train_label_value, test_item, k_num):
    simMap={}
    for cate_doc,value in train_label_value.items():
        similarity = computeSim(test_item,value)  # 调用computeSim()
        simMap[cate_doc] = similarity

    sortedSimMap = sorted(simMap.items(), key = lambda x: x[1], reverse=True)
    
    cateSimMap = {} #<类，距离和>
    for i in range(k_num):
        cate = sortedSimMap[i][0].split('_')[0]
        cateSimMap[cate] = cateSimMap.get(cate,0) + sortedSimMap[i][1]

    sortedCateSimMap = sorted(cateSimMap.items(),key=lambda x: x[1],reverse=True)

    return sortedCateSimMap[0][0]   
        
    #计算cosin相似度-----------------------------------------------------------------
def computeSim(test_v, train_v):   

    testVect = np.mat(test_v)  
    trainVect = np.mat(train_v)
    num = float(testVect * trainVect.T)
    denom = np.linalg.norm(testVect) * np.linalg.norm(trainVect)
    #print 'denom:%f' % denom
    return float(num)/(1.0+float(denom))

def main(): 
    doProcess()

if __name__ == "__main__": 
    main()