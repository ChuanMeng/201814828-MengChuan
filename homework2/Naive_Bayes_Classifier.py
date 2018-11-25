# -*- coding: utf-8 -*-
"""
本文档的任务如下：
1.读取二项式模型和伯努利模型的参数
2.读取训练集分别在这两个模型上做预测
3.做预测时，采用拉普拉斯平滑
4.得到测试结果

@author: MengChuan
"""
import math
from collections import Counter

def data_load():   
    #读取训练集的参数---------------------------------------------------------------------- 
    dict_num=0
    total_word_num=0  #总词数
    cate_word_num_dict={} #每个类下的总词数
    cate_per_word_num_dict={}  #在一个类下，某个词出现的次数
    
    #伯努利模型的统计信息----------------------------------------------------------
    total_doc_num=0 #文档的总个数
    cate_doc_num={}  #每个类的文档个数
    cate_per_word_doc_num_dict={} #在一个类下，某个词出现的文档数     
    
    
    open_dict=open('dict.txt','r')
    new_word_dict=[]#标准字典
    for dict_word in open_dict.readlines():
        new_dict_word=dict_word.strip('\n')
        new_word_dict.append(new_dict_word)
    open_dict.close()  
    
    
    inf_dict=open('Trained parameters'+'/'+'dict.txt','r')
    dict_num=inf_dict.read()
    inf_dict.close()    
    
      
    inf_total_word=open('Trained parameters'+'/'+'total_word_num.txt','r')
    total_word_num=inf_total_word.read()
    inf_total_word.close()    
    
    inf_cate_word=open('Trained parameters'+'/'+'cate_word_num.txt','r')
    for item in inf_cate_word.readlines():     
        new_item=item.strip('\n')
        i,j=new_item.split('  ')
        cate_word_num_dict[i]=j
    inf_cate_word.close() 
    
    per_word_cate=open('Trained parameters'+'/'+'per_word_cate_num.txt','r')
    for item in per_word_cate.readlines():     
        new_item=item.strip('\n')
        i,j=new_item.split('  ')
        cate_per_word_num_dict[i]=j
    per_word_cate.close() 
 
    inf_total_doc=open('Trained parameters'+'/'+'total_doc_num.txt','r')
    total_doc_num=inf_total_doc.read()
    inf_total_doc.close()  
    
    
    inf_cate_doc=open('Trained parameters'+'/'+'cate_doc_num.txt','r')
    for item in inf_cate_doc.readlines():     
        new_item=item.strip('\n')
        i,j=new_item.split('  ')
        cate_doc_num[i]=j
    inf_cate_doc.close() 
    
    
    per_word_cate_doc=open('Trained parameters'+'/'+'per_word_cate_doc_num.txt','r')
    for item in per_word_cate_doc.readlines():     
        new_item=item.strip('\n')
        i,j=new_item.split('  ')
        cate_per_word_doc_num_dict[i]=j
    per_word_cate_doc.close() 

    #读取测试集------------------------------------------------------------------
    test_total_list=[]
    test_total_list_label=[]
    test_doc_r=open('index_train or test set'+'/'+'testset.txt','r')
    for item in test_doc_r.readlines():
        new_item=item.strip('\n')
        cate,doc=new_item.split('_')
        load_dir='Preprocessed data'+'/'+cate+'/'+doc
        word_d=open(load_dir,'r')
        doc_list=[]
        test_total_list_label.append(cate)
        for word in word_d.readlines():
            new_word=word.strip('\n')
            doc_list.append(new_word)
        test_total_list.append(doc_list) 
        
    word_d.close()
    test_doc_r.close()  
    

    #开始测试-------------------------------------------------------------------
    N=len(test_total_list)
    count = 0
    for i,test_item in enumerate(test_total_list):
        '''
        classifyResult = binomial_Naive_Bayes_Classifier(test_item,dict_num,total_word_num,cate_word_num_dict,cate_per_word_num_dict,cate_doc_num)
        '''
        classifyResult = Bernoulli_Naive_Bayes_Classifier(test_item,total_doc_num ,cate_doc_num,cate_per_word_doc_num_dict,new_word_dict)

        if classifyResult == test_total_list_label[i]:
            count=count+1
        print("正在处理第%d条测试数据" %(i+1),",本条测试数据集的结果是:",classifyResult == test_total_list_label[i])
                  
    accuracy = float(count)/float(N) 
    print("测试集已经测试完毕") 
    print("最终测试集的精确度是：%f" % accuracy) 

    #使用多项式朴素贝叶斯进行运算--------------------------------------------------------------
def binomial_Naive_Bayes_Classifier(test_item,dict_num,total_word_num,cate_word_num_dict,cate_per_word_num_dict,cate_doc_num):
    result_score={}
    for i in cate_doc_num:
        result_score[i]=0

    for cate in result_score:
        result_score[cate]=binomial_term(cate,test_item,dict_num,total_word_num,cate_word_num_dict,cate_per_word_num_dict)
        
    sortedresult = sorted(result_score.items(), key = lambda x: x[1], reverse=True)

    return sortedresult[0][0]  
        
def binomial_term(cate,test_item,dict_num,total_word_num,cate_word_num_dict,cate_per_word_num_dict):
    MLE=0.0
    for word in test_item:
        MLE=MLE+binomial_detail(cate,word,cate_per_word_num_dict,dict_num,cate_word_num_dict)
    prior=float(cate_word_num_dict[cate])/float(total_word_num)
    bayes=MLE+math.log(prior)
    
    return bayes
    
def binomial_detail(cate,word,cate_per_word_num_dict,dict_num,cate_word_num_dict):
    dis=cate+'_'+str(word)
    numerator=1+float(cate_per_word_num_dict.get(dis,0))
    Denominator=float(dict_num)+float(cate_word_num_dict[cate])
    term=math.log(numerator/Denominator)
    return term   
    
    #实现伯努利模型朴素贝叶斯-------------------------------------------------------
def Bernoulli_Naive_Bayes_Classifier(test_item,total_doc_num ,cate_doc_num,cate_per_word_doc_num_dict,new_word_dict):
    new_test_item=[] #去重
    for i in Counter(test_item):
        new_test_item.append(i)
    result_score={}
    for i in cate_doc_num:
        result_score[i]=0

    for cate in result_score:
        result_score[cate]=Bernoulli_term(cate,test_item,total_doc_num ,cate_doc_num,cate_per_word_doc_num_dict,new_word_dict)
        
    sortedresult = sorted(result_score.items(), key = lambda x: x[1], reverse=True)

    return sortedresult[0][0]  

def Bernoulli_term(cate,new_test_item,total_doc_num ,cate_doc_num,cate_per_word_doc_num_dict,new_word_dict):
    MLE=0.0
    for word in new_word_dict:
        MLE=MLE+Bernoulli_detail(new_test_item,cate,word,cate_doc_num,cate_per_word_doc_num_dict)
    prior=float(cate_doc_num[cate])/float(total_doc_num)
    bayes=MLE+math.log(prior)
    
    return bayes
    
def Bernoulli_detail(new_test_item,cate,word,cate_doc_num,cate_per_word_doc_num_dict):
    dis=cate+'_'+str(word)
    numerator=1+float(cate_per_word_doc_num_dict.get(dis,0))
    Denominator=float(2)+float(cate_doc_num[cate])
    if word in new_test_item:
        term=math.log(numerator/Denominator)
    else:
        term=math.log(1-(numerator/Denominator))
    
    return term   
