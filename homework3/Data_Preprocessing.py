import json
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import numpy as np

def load_data():
    doc_list=[]
    label_list=[]#存储label
    corpus=[]
    with open('Tweets.json') as f:
        for line in f.readlines():
            doc_list.append(json.loads(line))
            label_list.append(json.loads(line)['cluster'])
            corpus.append(json.loads(line)['text'])
    
    label_num=set(label_list)

    #下面使用sklearn调用tf-idf获得每个utterance的embeding
    vectorizer = CountVectorizer() 
    count = vectorizer.fit_transform(corpus)
    transformer = TfidfTransformer()
    tfidf_matrix = transformer.fit_transform(count)
    np.savetxt("embeding/tf-idf.txt", tfidf_matrix.toarray()) 
    
    #将label进行存储
    with open('label.json', 'w') as f:
        json.dump(label_list, f)
        
if __name__ == "__main__": 
    load_data()  