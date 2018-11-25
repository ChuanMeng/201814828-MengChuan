# -*- coding: utf-8 -*-
"""
本文档的任务如下：
1.先对原始文档进行的读取，然后进行预处理（分词、去停止词、抽取词干）
2.划分测试集和训练集
3.构建字典，过滤掉在所有文档上词频<10的词，字典长度为15000+
4.多项式模型参数计算：计算总词频、计算每个类下的词频、计算在每一类下某个特征词的词频，计算完成后保存参数
5.伯努利模型参数计算：计算总文档数、计算每个类下的文档数量、计算每一类下包含某个特征词的文档数，计算完成后保存参数

Created on Mon Oct 22 08:35:39 2018
@author: MengChuan
"""
import nltk.stem
import string
from nltk.corpus import stopwords
import random
from os import listdir,mkdir,path

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

    
    return new_word_dict,len(word_dict)
   

def main(): 
    '''
    #先读取数据，进行预处理--------------------------------------------------------
    load_date()
    #进行训练集和测试集的划分------------------------------------------------------
    split_dataset()
    #从保存的字典里面，把字典值加载到内存里面-----------------------------------------
    a,n_dict=build_dict()     
    
    #从保存的字典里面，把字典值读取出来----------------------------------------------
    '''
    open_dict=open('dict.txt','r')
    new_word_dict=[]#标准字典
    for dict_word in open_dict.readlines():
        new_dict_word=dict_word.strip('\n')
        new_word_dict.append(new_dict_word)
    open_dict.close()      
    #多项式模型的统计信息---------------------------------------------------------           
    total_word_num=0  #总词数
    cate_word_num_dict={} #每个类下的总词数
    cate_per_word_num_dict={}  #在一个类下，某个词出现的次数
    
    #伯努利模型的统计信息----------------------------------------------------------
    total_doc_num=0 #文档的总个数
    cate_doc_num={}  #每个类的文档个数
    cate_per_word_doc_num_dict={} #在一个类下，某个词出现的文档数    

    
    #读取训练集------------------------------------------------------------------
    index_r=open('index_train or test set'+'/'+'trainset.txt','r')
    for item in index_r.readlines():     
        new_item=item.strip('\n')
        cate,doc=new_item.split('_')
        total_doc_num=total_doc_num+1       
        cate_doc_num[cate]=cate_doc_num.get(cate,0)+1
        load_dir='Preprocessed data'+'/'+cate+'/'+doc
        train_read=open(load_dir,'r')
        cate_per_word_doc_num_more_dict={}
        for word in train_read.readlines(): 
            new_word=word.strip('\n')
            if new_word not in new_word_dict:
                print(new_word,"舍掉，不属于字典") 
                continue
            else:
                print(new_word,"属于字典，留下计数")
                total_word_num=total_word_num+1
                cate_word_num_dict[cate]=cate_word_num_dict.get(cate,0)+1
                cate_name=cate+'_'+new_word
                cate_per_word_num_dict[cate_name]=cate_per_word_num_dict.get(cate_name,0)+1
                cate_per_word_doc_num_more_dict[cate_name]=cate_per_word_doc_num_more_dict.get(cate_name,0)+1
        for i in cate_per_word_doc_num_more_dict:
            cate_per_word_doc_num_dict[i]=1+cate_per_word_doc_num_dict.get(i,0)
                
    train_read.close()  
    index_r.close()
      
    
    inf_dict=open('Trained parameters'+'/'+'dict.txt','w')
    inf_dict.write('%s' %len(new_word_dict))
    inf_dict.close()    
    
    inf_total_word=open('Trained parameters'+'/'+'total_word_num.txt','w')
    inf_total_word.write('%s\n' %(total_word_num))
    inf_total_word.close()    
    
    inf_cate_word=open('Trained parameters'+'/'+'cate_word_num.txt','w')
    for i in cate_word_num_dict.items():
        item=i[0]+'  '+str(i[1])
        inf_cate_word.write('%s\n' % item)
    inf_cate_word.close() 
    
    per_word_cate=open('Trained parameters'+'/'+'per_word_cate_num.txt','w')
    for i in cate_per_word_num_dict.items():
        item=i[0]+'  '+str(i[1])
        per_word_cate.write('%s\n' %(item))
    per_word_cate.close()    
    
 
    inf_total_doc=open('Trained parameters'+'/'+'total_doc_num.txt','w')
    inf_total_doc.write('%s' %(total_doc_num))
    inf_total_doc.close()  
    
    
    inf_cate_doc=open('Trained parameters'+'/'+'cate_doc_num.txt','w')
    for i in cate_doc_num.items():
        item=i[0]+'  '+str(i[1])
        inf_cate_doc.write('%s\n' %(item))
    inf_cate_doc.close() 
    
    
    per_word_cate_doc=open('Trained parameters'+'/'+'per_word_cate_doc_num.txt','w')
    for i in cate_per_word_doc_num_dict.items():
        item=i[0]+'  '+str(i[1])
        per_word_cate_doc.write('%s\n' %(item))
    per_word_cate_doc.close()    
    
    
    print("字典长度:",len(new_word_dict))
    print('\n')
    print('\n')
    print("训练集总词数(考虑重复):",total_word_num)
    print('\n')
    print('\n')
    print("每个类的特征词数：",cate_word_num_dict)
    print('\n')
    print('\n')
    print("在一个类下，某个词出现的次数：",cate_per_word_num_dict)
    print('\n')
    print('\n') 
    print("总的文档个数：",total_doc_num)
    print('\n')
    print('\n')
    print("每个类的文档个数：",cate_doc_num)
    print('\n')
    print('\n')
    print("在一个类下，某个词出现的文档数：",cate_per_word_doc_num_dict)
    print('\n')
    print('\n')
        
    
