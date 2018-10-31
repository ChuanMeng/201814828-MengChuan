# -*- coding: utf-8 -*-
"""
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

nltk.download('punkt')
nltk.download('stopwords')

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

     
    
def split_dataset(rate=0.2):
    trainlist=[]
    testlist=[]
    new_dir='processed data'
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
                
    datew1=open('index_train or test set'+'trainset.txt','w')
    datew2=open('index_train or test set'+'testset.txt','w')
    for item in trainlist:
        datew1.write('%s\n' % item)
    for item in testlist:
        datew2.write('%s\n' % item)
    datew1.close()
    datew2.close()
 
    return len(trainlist)

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

def filter_stopwords(tokens):
    more=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    stoplist=stopwords.words('english')
    stoplist.extend(more)
    filtered_stemmed=[word for word in tokens if not word in stoplist]
    return filtered_stemmed


def get_stem_tokens(tokens):
    stemmed=[]
    s = nltk.stem.SnowballStemmer('english')
    for item in tokens:
        stemmed.append(s.stem(item))
    return stemmed

def use(text):
    
    tokens=get_tokens(text)
    filter_stopwords_tokens=filter_stopwords(tokens)
    stemmed_tokens=get_stem_tokens(filter_stopwords_tokens)
    '''
    count=Counter(stemmed_tokens)
    '''
    return stemmed_tokens

        
def build_dict_feature_words(select):
    word_dict={}
    new_word_dict={}
    new_dir='Preprocessed data'
    N=0
    index_r=open('index_train or test set'+'/'+select,'r')
    for item in index_r.readlines():
        N=N+1
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
        if v>=5:
            new_word_dict[k]=v
          
    index_r.close()
    
    dict_w=open('dict.txt','w')
    dict_w.write('%s\n' % len(new_word_dict)) 
    for word in new_word_dict: 
        dict_w.write('%s\n' %word)
    dict_w.close()
    
    
    print(new_word_dict)
    print(len(word_dict))
    print(len(new_word_dict))
    
    return new_word_dict,N,

def tf_idf(word,doc,list1):
    tf_sore=doc[word]/ sum(num for num in doc.values())
    contain_doc_num=sum(1 for doc in list1 if word in doc)
    idf_sore=math.log(len(list1)/contain_doc_num)
    tf_idf=tf_sore*idf_sore
    
    return tf_idf    


def tf(f_dict):
    
    return

def computeIDF():
    fileDir = 'processedSampleOnlySpecial_2'
    wordDocMap = {} 
    IDFPerWordMap = {}  
    countDoc = 0.0
    cateList = listdir(fileDir)
    for i in range(len(cateList)):
        sampleDir = fileDir + '/' + cateList[i]
        sampleList = listdir(sampleDir)
        for j in range(len(sampleList)):
            sample = sampleDir + '/' + sampleList[j]
            for line in open(sample).readlines():
                word = line.strip('\n')
                if word in wordDocMap.keys():
                    wordDocMap[word].add(sampleList[j]) # set结构保存单词word出现过的文档
                else:
                    wordDocMap.setdefault(word,set())
                    wordDocMap[word].add(sampleList[j])

    for word in wordDocMap.keys():
        countDoc = len(wordDocMap[word]) # 统计set中的文档个数
        IDF = math.log(20000/countDoc)/log
        
        
        IDFPerWordMap[word] = IDF
 
    return IDFPerWordMap
      

def main(): 
    f_dict,N=build_dict('trainset.txt')
    select='trainset.txt'
    vector={}
'''
    index_r=open('index_train or test set'+'/'+select,'r')
    for item in index_r.readlines():
        new_item=item.strip('\n')
        cate,doc=new_item.split('_')
        load_dir='processed data'+'/'+cate+'/'+doc
        dict_d=open(load_dir,'r')
        for word in dict_d.readlines():

'''

if __name__ == "__main__": 
    main()


