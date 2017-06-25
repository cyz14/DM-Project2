# DM-Project 2

Chen Yazheng

Gao Tong

## Dependency Packages

* numpy
* beautifulsoup4
* scikit-learn
* xgboost

## 如何运行查看结果

### 1. 预处理

script/const.py 中定义了语料库的位置，如下

```python
corpus_dir = "../samples_50000"
```

预处理需要运行

```sh
python preprocess.py
```

运行结束会输出有效文档个数，并把 TFIDF 矩阵存在 tfidf.pkl 文件中供之后load_data。

### 2. 分类算法比较

classifier.py 使用 optionparser 处理传入的参数

```python
parser.add_option("-c", "--class", default="lgr", help="choose a classifier among lgr, nb, sgd, dct and mlp", action="store", type="string", dest="clf")
```

运行不同的分类器需要指定对应名字

重要：载入data需要按Enter键确认

```sh
# Logistic Regression
python classifier.py -c lgr

# Naive Bayes
python classifier.py -c nb

# SVM
python classifier.py -c sgd

# Decision Tree
python classifier.py -c dct

# MLP
python classifier.py -c mlp
```

采用不同的数据集大小可以用 -s 选项，默认值是 50000.

```sh
parser.add_option("-s", "--size", default=1000, help="config the data set size", action="store", type="int", dest="size")
```

调用时示例

```sh
python classifier.py -c lgr -s 10000
```

### 3. ensemble算法比较

ensemble.py 运行不同的 ensemble 算法需要指定对应名字

```sh
# Bagging
python classifier.py -c bag

# AdaBoost
python classifier.py -c ada

# RandomForest
python classifier.py -c rdf

# GradientBoost
python classifier.py -c grd
```

指定数据集大小
```
python classifier.py -c ada -s 10000
```

其中，在 GradientBoost 中因为速度太慢限制了数据集大小为 1000.

### 4. 聚类算法比较

cluster.py 运行 kmeans 或 dbscan 算法。

```sh
# K-Means
python cluster.py -c kms -s 1000

# DBScan
python cluster.pu -c dbs -s 1000
```

采用PCA进行降维的操作在数据量很大时速度很慢，我们主要用了1000的数据来求比较好的聚类效果。

## Project Structure

### const.py

定义了一些全局常量。

corpus_dir:   语料库地址

trash_dir:    因为内容缺失等原因被移出的文档路径

DATASET_SIZE: 数据规模

### preprocess.py

完成预处理，包括文本处理，tfidf 矩阵的计算，数据集的 dump 和 load。

训练集和测试集的分割在训练时完成。

#### load_data: function

返回预处理得到的 tfidf 矩阵及文档分类数组和文档分类的数字对应的单词组成的列表

### classifier.py

直接运行会尝试载入数据，如果不存在会调用 preprocess.py 预处理，之后就无需再预处理，而是从 pickle 文件直接载入。
完成5种分类器的实现和性能比较。五种分类器：

* Logistic Regression
* Naive Bayes
* SVM
* Decision Tree
* MLP

其中 MLP 的速度相对较慢。

### ensemble.py

ensemble 算法测试

### cluster.py

聚类算法测试