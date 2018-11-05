# -*- coding: utf-8 -*-
"""
本文档的任务如下：
1.读取训练集和测试集的embeding
2.构建封装KNN的计算方法
3.先计算在N-fold验证集上的结果
4.最后计算在测试集上的结果

@author: MengChuan
"""
import numpy as np

def data_load():       
    #读取已经训练好的训练集与测试集的embeding----------------------------------------
    train=np.loadtxt("embeding/trainset embeding.txt")  
    #test=np.loadtxt("embeding/testset embeding.txt")        
    
    train_label_list=[] #训练集的label list
    train_label_value={}
    train_label=open('index_train or test set'+'/'+'trainset.txt','r')
    for i,item in enumerate(train_label.readlines()):
        new_item=item.strip('\n')
        cate,doc=new_item.split('_')
        train_label_list.append(cate)
        train_label_value[new_item]=train[i]
    train_label.close()   
    
    
    test_label_list=[] #测试集的label list   
    test_label=open('index_train or test set'+'/'+'testset.txt','r')
    for item in test_label.readlines():
        new_item=item.strip('\n')
        cate,doc=new_item.split('_')
        test_label_list.append(cate)
    test_label.close()
   
        
    #开始进行5-fold交叉验证------------------------------------------------------- 
    N_fold_validation_list= N_fold_validation(5,train_label_value)
    sum_N_fold_validation=sum(N_fold_validation_list)
    average_sore=sum_N_fold_validation/len(N_fold_validation_list)
    for i,j in enumerate(N_fold_validation_list):
        print("第%d-fold上的验证集精确度为:%f" %((i+1),j))
    print("在5-fold验证集上的平均精确度为：" , average_sore)
   

    #开始测试-------------------------------------------------------------------
    N=len(test)
    count = 0
    for i,test_item in enumerate(test):
        classifyResult = KNN(train_label_value,test_item,15)
        if classifyResult == test_label_list[i]:
            count=count+1
        print(classifyResult == test_label_list[i])
        print("正在处理第%d条测试数据" %(i+1))
            
    accuracy = float(count)/float(N) 
    print("测试集已经测试完毕") 
    print("最终测试集的精确度是：%f" % accuracy) 

 

    #进行N-fold交叉验证----------------------------------------------------------------
def N_fold_validation(fold_num,train_label_value):
    N_fold_rate=float(1/fold_num)
    N_fold=[]
    cate_total=[]
    cate={}
    #先看看训练集有哪些类
    for cate_doc in train_label_value:
        cate[cate_doc.split('_')[0]]=None
    #把不同类的训练集放到不同的list来存储    
    for cate_name in cate:
        cate_list=[]
        for cate_doc in train_label_value:
            if cate_doc.split('_')[0]==cate_name:
                cate_list.append(cate_doc)
        cate_total.append(cate_list) 
    #进行5-fold训练            
    for i in range(fold_num):
        train_section={}
        test_section={}
        for cate_list in cate_total:
            j=len(cate_list)*N_fold_rate
            for k in range(len(cate_list)):
                if k>=i*j and k<(i+1)*j:
                    test_section[cate_list[k]]=train_label_value[cate_list[k]]
                
                else:
                    train_section[cate_list[k]]=train_label_value[cate_list[k]]                
        
        count=0
        num=0
        for label,test_item in test_section.items():
            num=num+1
            print("正在处理第%d-fold的第%d个验证数据集,当前验证集的总数为%d" %((i+1),num,len(test_section)))
                
            if label.split('_')[0]==KNN(train_section,test_item ,15):
                count=count+1
            print("第%d个验证集的分类结果是：" % num,label.split('_')[0]==KNN(train_section,test_item ,15))
        accuracy=float(count)/float(len(test_section))
        N_fold.append(accuracy)
    return N_fold

    #使用KNN进行运算--------------------------------------------------------------
def KNN(train_label_value, test_item, k_num):
    simMap={}
    for cate_doc,value in train_label_value.items():
        similarity = cosin_similarity(test_item,value) 
        
        simMap[cate_doc] = similarity

    sortedSimMap = sorted(simMap.items(), key = lambda x: x[1], reverse=True)
    
    cateSimMap = {} 
    for i in range(k_num):
        cate = sortedSimMap[i][0].split('_')[0]
        cateSimMap[cate] = cateSimMap.get(cate,0) + sortedSimMap[i][1]

    sortedCateSimMap = sorted(cateSimMap.items(),key=lambda x: x[1],reverse=True)

    return sortedCateSimMap[0][0]   
        
    #计算cosin相似度-----------------------------------------------------------------
def cosin_similarity(test_v, train_v):   

    testVect = np.mat(test_v)  
    trainVect = np.mat(train_v)
    num = float(testVect * trainVect.T)
    denom = np.linalg.norm(testVect) * np.linalg.norm(trainVect)
    return float(num)/(1.0+float(denom))


