# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:24:52 2018

@author: MengChuan
"""
from Vecter Space Model

def doProcess():
    trainFiles = 'docVector/wordTFIDFMapTrainSample0'
    testFiles = 'docVector/wordTFIDFMapTestSample0'
    kNNResultFile = 'docVector/KNNClassifyResult'

    trainDocWordMap = {}  # 字典<key, value> key=cate_doc, value={{word1,tfidf1}, {word2, tfidf2},...}

    for line in open(trainFiles).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        trainWordMap = {}
        m = len(lineSplitBlock)-1
        for i in range(2, m, 2):  # 在每个文档向量中提取(word, tfidf)存入字典
            trainWordMap[lineSplitBlock[i]] = lineSplitBlock[i+1]

        temp_key = lineSplitBlock[0] + '_' + lineSplitBlock[1]  # 在每个文档向量中提取类目cate，文档doc，
        trainDocWordMap[temp_key] = trainWordMap 

    testDocWordMap = {}

    for line in open(testFiles).readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        testWordMap = {} 
        m = len(lineSplitBlock)-1
        for i in range(2, m, 2):
            testWordMap[lineSplitBlock[i]] = lineSplitBlock[i+1]

        temp_key = lineSplitBlock[0] + '_' + lineSplitBlock[1]
        testDocWordMap[temp_key] = testWordMap #<类_文件名，<word, TFIDF>>

    #遍历每一个测试样例计算与所有训练样本的距离，做分类
    count = 0
    rightCount = 0
    KNNResultWriter = open(kNNResultFile,'w')
    for item in testDocWordMap.items():
        classifyResult = KNNComputeCate(item[0], item[1], trainDocWordMap)  # 调用KNNComputeCate做分类

        count += 1
        print 'this is %d round' % count

        classifyRight = item[0].split('_')[0]
        KNNResultWriter.write('%s %s\n' % (classifyRight,classifyResult))
        if classifyRight == classifyResult:
            rightCount += 1
        print '%s %s rightCount:%d' % (classifyRight,classifyResult,rightCount)

    accuracy = float(rightCount)/float(count)
    print 'rightCount : %d , count : %d , accuracy : %.6f' % (rightCount,count,accuracy)
    return accuracy
            


#########################################################
## @param cate_Doc 测试集<类别_文档>
## @param testDic 测试集{{word, TFIDF}}
## @param trainMap 训练集<类_文件名，<word, TFIDF>>
## @return sortedCateSimMap[0][0] 返回与测试文档向量距离和最小的类
#########################################################
def KNNComputeCate(cate_Doc, testDic, trainMap):
    simMap = {} #<类目_文件名,距离> 后面需要将该HashMap按照value排序
    for item in trainMap.items():
        similarity = computeSim(testDic,item[1])  # 调用computeSim()
        simMap[item[0]] = similarity

    sortedSimMap = sorted(simMap.iteritems(), key=itemgetter(1), reverse=True) #<类目_文件名,距离> 按照value排序

    k = 20
    cateSimMap = {} #<类，距离和>
    for i in range(k):
        cate = sortedSimMap[i][0].split('_')[0]
        cateSimMap[cate] = cateSimMap.get(cate,0) + sortedSimMap[i][1]

    sortedCateSimMap = sorted(cateSimMap.iteritems(),key=itemgetter(1),reverse=True)

    return sortedCateSimMap[0][0]   
        
    
#################################################
## @param testDic 一维测试文档向量<<word, tfidf>>
## @param trainDic 一维训练文档向量<<word, tfidf
## @return 返回余弦相似度
def computeSim(testDic, trainDic):
    testList = []  # 测试向量与训练向量共有的词在测试向量中的tfidf值
    trainList = []  # # 测试向量与训练向量共有的词在训练向量中的tfidf值
    
    for word, weight in testDic.items():
        if trainDic.has_key(word):
            testList.append(float(weight)) # float()将字符型数据转换成数值型数据，参与下面运算
            trainList.append(float(trainDic[word]))

    testVect = mat(testList)  # 列表转矩阵，便于下面向量相乘运算和使用Numpy模块的范式函数计算
    trainVect = mat(trainList)
    num = float(testVect * trainVect.T)
    denom = linalg.norm(testVect) * linalg.norm(trainVect)
    #print 'denom:%f' % denom
    return float(num)/(1.0+float(denom))