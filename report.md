# Project 2 Report

陈雅正	2014011423

高童 	2014011357

## 一、数据预处理与特征提取

语料库的路径 corpus_dir 在 const.py 中指定。

对于没有 full_text 的文档直接忽略，并且移动 trash_dir 中了。

xml 的解析使用 BeautifulSoup 提取所需元素，包括文本，主题，日期等。

最后所有文档的 full_text 存到 corpus 列表中，文档主题存到了 target_names 里，target 中存主题对应的数字。

语料库的预处理采用 sklearn.feature_extraction.text 的 CountVectorizer。

TFIDF 矩阵的计算使用 sklearn.feature_extraction.text 的 TfidfTransformer。

得到的 x_train_tfidf, target, target_names 用 pickle 存储，计算一次以后都可以直接用 load_data 函数读取。

## 二、基本分类器运用与比较

## 三、ensemble 算法运用与比较

## 四、聚类算法运用与比较

## 组内分工情况

## 遇到的问题及解决

Gradient Boosting Classifier sparse matrix issue using scikit-learn
http://blog.csdn.net/suibianshen2012/article/details/52860522
https://stackoverflow.com/questions/17184079/sklearn-cannot-use-encoded-data-in-random-forest-classifier