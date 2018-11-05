# -*- coding: utf-8 -*-
"""
本文档任务如下：
1.先调用VSM，生成训练集和测试集的embeding
2.调用KNN，实现分类，获得测试集和验证集的结果

Created on Mon Oct 29 10:25:42 2018

@author: maqian
"""


import VSM
import KNN

def main(): 
    
    VSM.main_process()

    KNN.data_load()
    

if __name__ == "__main__": 
    main()