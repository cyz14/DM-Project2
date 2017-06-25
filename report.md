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

采用以下分类算法，用十折交叉验证进行比较。以下时间均在同一台计算机上进行计算。

| 分类算法          | Precision | Recall | F1-measure | Time   |
| ------------- | --------- | ------ | ---------- | ------ |
| Log Reg       | 60.8%     | 58.7%  | 54.0%      | 14m2s  |
| Naive Bayes   | 44.9%     | 44.9%  | 33.6%      | 11s    |
| SVM           | 51.8%     | 54.7%  | 50.1%      | 2m7s   |
| Decision Tree | 44.8%     | 45.8%  | 44.8%      | 15m34s |
| MLP           |           |        |            |        |

从表中可以看到，朴素贝叶斯法在时间上具有极大优势，但分类效果最差。逻辑回归和决策树所花时间在一个数量级，但逻辑回归准确率远高于决策树。SVM所花时间不长，但效果较好。由此看来，在这个问题上，SVM是比较容易令人接受的分类算法。

## 三、ensemble 算法运用与比较

## 四、聚类算法运用与比较

## 组内分工情况

## 遇到的问题及解决

Gradient Boosting Classifier sparse matrix issue using scikit-learn
http://blog.csdn.net/suibianshen2012/article/details/52860522
https://stackoverflow.com/questions/17184079/sklearn-cannot-use-encoded-data-in-random-forest-classifier