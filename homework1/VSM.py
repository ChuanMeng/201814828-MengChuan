# -*- coding: utf-8 -*-
"""
本文档的任务如下：
1.先对原始文档进行的读取，然后进行预处理（分词、去停止词、抽取词干）
2.划分测试集和训练集
3.构建字典，过滤掉在所有文档上词频<10的词，字典长度为15000+
4.计算训练集的tf-idf值，得到训练集所有文档的embeding，并保存
5.计算测试集的tf-idf值，得到测试集所有文档的embeding，并保存

Created on Mon Oct 22 08:35:39 2018
@author: MengChuan
"""
import nltk.stem
import string
from collections import Counter
from nltk.corpus import stopwords
import math
import random
from os import listdir,mkdir,path
import numpy as np

    #读取所有文档--------------------------------------------------------------------
def load_date():
    origin_dir='20news-18828'
    new_dir='Preprocessed data'
    cate_list=listdir(origin_dir)
    split_dataset()

    for cate in cate_list:
        doc_list=listdir(origin_dir+'/'+cate) 
        for doc in doc_list:
            r=open(origin_dir+'/'+cate+'/'+doc,'r',encoding='UTF-8',errors='ignore')          
            new_cate_dir=new_dir+'/'+cate
            if path.exists(new_cate_dir)==False:
                mkdir(new_cate_dir)
            new_doc_dir=new_dir+'/'+cate+'/'+doc
            w=open(new_doc_dir,'w')
            new_doc=use(r.read())
            
            for word in new_doc:                      
                w.write('%s\n' % word)

            w.close()    
            r.close()                                  
    return        

   #调用分词、过滤停止词、抽取词干
def use(text):
    
    tokens=get_tokens(text)
    filter_stopwords_tokens=filter_stopwords(tokens)
    stemmed_tokens=get_stem_tokens(filter_stopwords_tokens)
    
    return stemmed_tokens

    #将数据集分为训练集和测试集----------------------------------------------------
def split_dataset(rate=0.2):
    trainlist=[]
    testlist=[]
    new_dir='Preprocessed data'
    cate_list=listdir(new_dir)
    for cate in cate_list:
        doc_list=listdir(new_dir+'/'+cate) 
        random.shuffle(doc_list)
        j=len(doc_list)*rate
        for i in range(len(doc_list)):

            if i>=0 and i<j:
                testlist.append(cate+'_'+ doc_list[i])
                
            else:
                trainlist.append(cate+'_'+ doc_list[i])
                
    datew1=open('index_train or test set'+'/'+'trainset.txt','w')
    datew2=open('index_train or test set'+'/'+'testset.txt','w')
    for item in trainlist:
        datew1.write('%s\n' % item)
    for item in testlist:
        datew2.write('%s\n' % item)
    datew1.close()
    datew2.close()
 
    return len(trainlist)

    #进行分词--------------------------------------------------------------------
def get_tokens(text):
    lower = text.lower()
    remove_punctuation_map = {}
    total_string=string.punctuation+string.digits
    space=' '
    remove_punctuation_map = str.maketrans({key:space for key in total_string})
    lower.translate(remove_punctuation_map)
    no_punctuation = lower.translate(remove_punctuation_map)    
    tokens = nltk.word_tokenize(no_punctuation)
    return tokens

    #过滤停止词------------------------------------------------------------------
def filter_stopwords(tokens):
    more=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    stoplist=stopwords.words('english')
    stoplist.extend(more)
    filtered_stemmed=[word for word in tokens if not word in stoplist]
    return filtered_stemmed

    #抽取词干--------------------------------------------------------------------
def get_stem_tokens(tokens):
    stemmed=[]
    s = nltk.stem.SnowballStemmer('english')
    for item in tokens:
        stemmed.append(s.stem(item))
    return stemmed


    #构建字典-------------------------------------------------------------------   
def build_dict(select='trainset.txt'):
    word_dict={}
    new_word_dict={}
    index_r=open('index_train or test set'+'/'+select,'r')
    for item in index_r.readlines():
        new_item=item.strip('\n')
        cate,doc=new_item.split('_')
        load_dir='Preprocessed data'+'/'+cate+'/'+doc
        dict_d=open(load_dir,'r')
        
        for word in dict_d.readlines():
            new_word=word.strip('\n')
            word_dict[new_word]=word_dict.setdefault(new_word,0)+1
            print(cate+' '+doc+' '+new_word)
                
        dict_d.close()
        
    for k,v in word_dict.items():
        if v>=10:
            new_word_dict[k]=v
          
    index_r.close()
    
    dict_w=open('dict.txt','w')

    for word in new_word_dict: 
        dict_w.write('%s\n' %word)
    dict_w.close()  
    
    print(new_word_dict)
    print(len(word_dict))
    print(len(new_word_dict))
    
    return new_word_dict,

    #计算训练集数据的idf值--------------------------------------------------------
def train_idf(word,train_total_list):
       
    contain_doc_num=sum(1 for doc in train_total_list if word in doc)
    idf_sore=math.log(len(train_total_list)/contain_doc_num) 

    return contain_doc_num,idf_sore

    
    #计算测试数据的idf值---------------------------------------------------------
def test_idf(word,train_total_list,train_word_df,test_doc):
       
    N=len(train_total_list)+1
    if word in test_doc:
        idf_sore=math.log(N/(train_word_df[word]+1))
    else:
        idf_sore=math.log(N/(train_word_df[word]))

    return idf_sore

    #计算训练集的tf—idf值--------------------------------------------------------
def train_tf_idf(word,doc,idf_dict):  
    if word not in doc:
        tf_sore=0
    else:
        tf_sore=math.log(doc[word])+1       
    idf_sore=idf_dict[word]
    tf_idf=tf_sore*idf_sore
    
    return tf_idf   

    #计算测试集上的tf-idf值------------------------------------------------------
def test_tf_idf(word,test_doc,idf_dict):  
    if word not in test_doc:
        tf_sore=0
    else:
        tf_sore=math.log(test_doc[word])+1       
    idf_sore=idf_dict[word]
    tf_idf=tf_sore*idf_sore
   
    return tf_idf    

def main_process(): 
    #先读取数据，进行预处理--------------------------------------------------------
    load_date()
    #进行训练集和测试集的划分------------------------------------------------------
    split_dataset()
    #从保存的字典里面，把字典值加载到内存里面-----------------------------------------
    a,n_dict=build_dict()     
    
    #从保存的字典里面，把字典值读取出来----------------------------------------------
    open_dict=open('dict.txt','r')
    new_word_dict=[]#标准字典
    for dict_word in open_dict.readlines():
        new_dict_word=dict_word.strip('\n')
        new_word_dict.append(new_dict_word)
        
    open_dict.close()       
    
    #读取训练集------------------------------------------------------------------
    train_total_list=[] #训练集的文档个数
    index_r=open('index_train or test set'+'/'+'trainset.txt','r')
    for item in index_r.readlines():
        new_item=item.strip('\n')
        cate,doc=new_item.split('_')
        load_dir='Preprocessed data'+'/'+cate+'/'+doc
        dict_d=open(load_dir,'r')
        doc_list=[]
        for word in dict_d.readlines():
            new_word=word.strip('\n')
            doc_list.append(new_word)
        count=Counter(doc_list)
        train_total_list.append(count)
    index_r.close()  
    
    #先计算训练集的idf值，然后保存起来----------------------------------------------
    
    train_idf_dict={}
    train_word_df={}
    
    for word in new_word_dict:   
        train_word_df[word],train_idf_dict[word]=train_idf(word,train_total_list)
 
    #计算训练集的tf-idf值--------------------------------------------------------
    train_tf_idf_doc=[]
    for doc in train_total_list:
        tf_idf_word=[]
        print("读取训练集的文档：")
        print(doc)
        print('\n')

        for word in new_word_dict:
            tf_idf_word.append(train_tf_idf(word,doc,train_idf_dict)) #得到每个词表的值         
        train_tf_idf_doc.append(tf_idf_word)#把每个文档的embeding合成一个总的list
    
    #将训练集计算得到的tf-idf值转化成矩阵，并存储------------------------------------
    train_idf_m=np.mat(train_tf_idf_doc)
    np.savetxt("embeding/trainset embeding.txt", train_idf_m)  
    
    
    #读取测试集------------------------------------------------------------------
    test_total_list=[]
    index_r=open('index_train or test set'+'/'+'testset.txt','r')
    for item in index_r.readlines():
        new_item=item.strip('\n')
        cate,doc=new_item.split('_')
        load_dir='Preprocessed data'+'/'+cate+'/'+doc
        dict_d=open(load_dir,'r')
        doc_list=[]
        for word in dict_d.readlines():
            new_word=word.strip('\n')
            doc_list.append(new_word)
        count=Counter(doc_list)
        test_total_list.append(count)
        
    index_r.close()      
        
    #计算测试集的idf值--------------------------------------------------------    
    test_idf_dict={}
    for i,test_doc in enumerate(test_total_list):   
        test_idf_doc_dict={}
        for word in new_word_dict:  
            test_idf_doc_dict[word]=test_idf(word,train_total_list,train_word_df,test_doc)
        test_idf_dict[i]=test_idf_doc_dict
                
    #计算测试集的tf-idf总值    
    test_tf_idf_doc=[]
    for i,test_doc in enumerate(test_total_list):
        tf_idf_word=[]
        print("读取测试集的文档：")
        print(test_doc)
        print('\n')        
        for word in new_word_dict:
            tf_idf_word.append(test_tf_idf(word,test_doc,test_idf_dict[i]))          
        test_tf_idf_doc.append(tf_idf_word)                
        
    #将测试集计算得到的tf-idf值转化成矩阵，并存储------------------------------------
    
    test_idf_m=np.mat(test_tf_idf_doc)
    np.savetxt("embeding/testset embeding.txt", test_idf_m)   

    #读取、测试训练集与测试集的embeding    
    trainset_embeding=np.loadtxt("embeding/trainset embeding.txt")  
    print(len(trainset_embeding))
    print(trainset_embeding.shape)    
    print('\n') 
    testset_embeding=np.loadtxt("embeding/testset embeding.txt")  
    print(len(testset_embeding))
    print(testset_embeding.shape)    
