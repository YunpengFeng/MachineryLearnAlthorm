from sklearn.feature_extraction import DictVectorizer
# 从文本进行特征提取
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import jieba
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Imputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
import numpy as np

def ChineseVector():
    # 中文分词
    c1 = jieba.cut("初次使用pycharm 的interpreter option为空解决办法")
    c2 = jieba.cut("这里我们说明pycharm中正确的项目删除方式 ")
    c3 = jieba.cut("与其他语言的IDE相比项目删除起来比较困难")
    print(type(c1))
    co1 = list(c1)
    print("c1",co1[0])
    co2 = list(c2)
    co3 = list(c3)
    c11 = ' '.join(co1)
    c22 = ' '.join(co2)
    c33 = ' '.join(co3)
    return c11, c22, c33


def test_sklearn():
    #list =['python programing  is my  is like', 'I also like Java too']
    c1, c2, c3 = ChineseVector()
    #方式一 根据出现的次数
    #cv = CountVectorizer()
    #方式二 词的重要性
    cv = TfidfVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    #sparse矩阵形式输出，不输出0 的位置， 输出的是有值的位置
    print("data:\n",data)
    #数组形式输出
    print("dataarray:\n", data.toarray())
    #输出字典项
    print("dict:\n",cv.get_feature_names())
    return None


def maxmin():
    # 归一化
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    mm = MinMaxScaler(feature_range=(0, 2))
    temp = mm.fit_transform(data)
    print("最大：", mm.data_max_)
    print(temp)

    return None


def stand():
    # 标准化
    data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    standard = StandardScaler()
    temp = standard.fit_transform(data)
    print("样本均值：", standard.mean_)
    print("样本方差：", standard.var_)
    print("样本参数：", standard.get_params())
    print(temp)
    return None


def Im():
    # 缺失值处理
    im = Imputer(missing_values='NaN', strategy='mean', axis=0, verbose=0,copy=True)
    data = [[-1, 2], [np.nan, 6], [0, 10], [1, 18]]
    temp = im.fit_transform(data)
    print(temp)
    return None


def var():
    # 删除低方差的特征
    data = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3], [1, 3, 4, 6]]
    _var = VarianceThreshold(threshold=0.12)
    temp = _var.fit_transform(data)
    print(temp)
    return None


def pca():
    # 使用pca进行降维
    data = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3], [1, 3, 4, 6]]
    _pca = PCA(n_components=0.10)
    temp= _pca.fit_transform(data)
    print(temp)
    return None


if __name__ == "__main__":
    # test_sklearn()
    # maxmin()
    # stand()
    # Im()
    # var()
    pca()
