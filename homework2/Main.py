# -*- coding: utf-8 -*-
"""
本文档任务如下：
1.先调用Data_Preprocessing进行训练得到参数
2.调用Naive_Bayes_Classifier做预测

Created on Mon Oct 29 10:25:42 2018

@author: Meng chuan
"""

import Data_Preprocessing
import Naive_Bayes_Classifier

def main(): 
    
    Data_Preprocessing.main_process()
    Naive_Bayes_Classifier.data_load()
    

if __name__ == "__main__": 
    main()