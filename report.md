# Project 2 Report

陈雅正	2014011423

高童 	2014011357

https://github.com/cyz14/DM-Project2

## 一、数据预处理与特征提取

语料库的路径 corpus_dir 在 const.py 中指定。

对于没有 full_text 的文档直接忽略，并且移动 trash_dir 中了。

xml 的解析使用 BeautifulSoup 提取所需元素，包括文本，主题，日期等。

最后所有文档的 full_text 存到 corpus 列表中，文档主题存到了 target_names 里，target 中存主题对应的数字。

语料库的预处理采用 sklearn.feature_extraction.text 的 CountVectorizer。

TFIDF 矩阵的计算使用 sklearn.feature_extraction.text 的 TfidfTransformer。

得到的 x_train_tfidf, target, target_names 用 pickle 存储，计算一次以后都可以直接用 load_data 函数读取。

## 二、基本分类器运用与比较

采用以下分类算法，用十折交叉验证进行比较。以下分类任务进行时数据集大小均取50000，时间均在同一台计算机上进行计算。

| 分类算法          | Precision | Recall | F1-measure | Time         |
| ------------- | --------- | ------ | ---------- | ------------ |
| Log Reg       | 60.8%     | 58.7%  | 54.0%      | 14m2s        |
| Naive Bayes   | 44.9%     | 44.9%  | 33.6%      | 11s          |
| SVM           | 51.8%     | 54.7%  | 50.1%      | 2m7s         |
| Decision Tree | 44.8%     | 45.8%  | 44.8%      | 15m34s       |
| MLP           | 53.0%     | 55.0%  | 53.0%      | 3h49m (est.) |

从表中可以看到，朴素贝叶斯法在时间上具有极大优势，但分类效果最差。逻辑回归和决策树所花时间在一个数量级，但逻辑回归准确率远高于决策树。SVM所花时间不长，但效果较好。MLP花费的时间很长，以上时间为通过单次训练10倍得到的估计值。由此看来，在这个问题上，SVM是比较容易令人接受的分类算法。

## 三、ensemble 算法运用与比较

采用以下 ensemble 算法，KFold 进行十折交叉验证。数据集大小使用的是 30000。以下时间均在同一台计算机上进行计算。

Bootstrap 自助法是一种选取思想。在 Random Forest Classifier 中有 bootstrap 开启选项。

| 分类算法           | Precision | Recall | F1-measure | Time     |
| -------------- | --------- | ------ | ---------- | -------- |
| Bagging        | 40.7%     | 14.2%  | 14.6%      | 4m37s    |
| AdaBoost       | 17.2%     | 25.9%  | 14.9%      | 2m54s    |
| Random Forest  | 43.4%     | 45.3%  | 40.0%      | 6m24s    |
| Gradient Boost | 92.2%     | 92.2%  | 93.0%      | too long |

注：GradientBoost 在使用 sklearn 的实现的时候速度实在太慢，调整了 n_estimators=10 还是太慢，于是就只用了大小为 1000 的测试集来训练和测试，需要训练 40 分钟左右。因此我们将训练的结果用 pickle 存下，再次测试可以直接 load 训练好的模型。在 1000 的数据集里用 100 的测试集测试10次的平均正确率是 92% 左右。

## 四、聚类算法运用与比较

采用以下聚类算法，由于作业时间有限，数据集大小均取5000，时间均在同一台计算机上进行计算。

在聚类前，通过PCA将数据降到10维；聚类后，用TSNE降到2维，随后作出聚类图像。

| 聚类算法   | NMI   | AMI   | Time  |
| ------ | ----- | ----- | ----- |
| KMeans | 0.192 | 0.136 | 2m45s |
| DBSCAN | 0.146 | 0.045 | 2m26s |

作出的聚类图像位于result文件夹下。从数据可以看出，类与类间区分不明显，不适合采用DBSCAN。

## 组内分工情况

本次实验分工为：

高童负责
    二、基本分类器运用和比较
    四、聚类算法运用和比较，

陈雅正负责
    框架搭建和一、三等其余部分。

## 遇到的问题及解决

Gradient Boosting Classifier sparse matrix issue using scikit-learn
http://blog.csdn.net/suibianshen2012/article/details/52860522
https://stackoverflow.com/questions/17184079/sklearn-cannot-use-encoded-data-in-random-forest-classifier