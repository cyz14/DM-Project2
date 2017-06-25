# DM-Project 2

Chen Yazheng

Gao Tong

## Dependency Packages

* numpy
* beautifulsoup4
* scikit-learn
* xgboost

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

使用 optionparser 处理传入的参数

```python
parser.add_option("-c", "--class", default="lgr", help="choose a classifier among lgr, nb, sgd, dct and mlp", action="store", type="string", dest="clf")
```
