"""
需求：使用K-近邻算法进行分类
iris数据集的中文名是安德森鸢尾花卉数据集，
英文全称是Anderson’s Iris data set。
iris包含150个样本，对应数据集的每行数据。
每行数据包含每个样本的四个特征和样本的类别信息，
每个样本包含了花萼长度、花萼宽度、花瓣长度、花瓣宽度四个特征（前4列），
我们需要建立一个分类器，分类器可以通过样本的四个特征来判断
样本属于山鸢尾、变色鸢尾还是维吉尼亚鸢尾（这三个名词都是花的品种）。

分析：花萼长度、花萼宽度、花瓣长度、花瓣宽度
特征值：花的品种
"""

from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import  GridSearchCV as girdSCV
# 字典特征抽取
from sklearn.feature_extraction import  DictVectorizer as dictvector
from graphviz import Digraph


def knn():
    # K-近邻算法
    # 读取数据
    data = pd.read_csv("../datasets/iris.csv")
    # 筛选
    # data = data.query("el>5.0&ew<10")
    # 处理数据,将种类转化转化为数值型
    df = pd.DataFrame(data)
    type_mapping ={"Iris-setosa": 1,
                   "Iris-versicolor": 2,
                   "Iris-virginica": 3}
    df["type"] = df["type"].map(type_mapping)
    # df['testlow']= 3233
    print(df.head(20))
    # 取出除特征值和目标值
    y = df['type']
    # 去除type ,其他的都是特征值
    x = df.drop(["type"],axis= 1)

    # 将数据分割成为“训练集”和“测试集”  返回结果 （训练集、测试集的特征值）,(训练集、测试集的目标值)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 特征工程（标准化）
    std= StandardScaler();
    # 对训练集和测试集的特征值标准化
    x_train = std.fit_transform(x_train)
    # 上面已经fit计算了值 ，所以不用fit_transform
    x_test = std.transform(x_test)
    # 进行K-近邻算法 #超参数n_neighbors
    knn = KNeighborsClassifier()
    """
    knn = KNeighborsClassifier(n_neighbors=5,algorithm="brute")
    knn.fit(x_train, y_train)
    # 得出测试数据
    y_predict = knn.predict(x_test)
    #用一组[]会报错 ValueError: Expected 2D array, got 1D array instead:
    # 需要预测的值
    need_predict = [[5.4,3.7,1.5,0.2],[5.0,2.0,3.5,1.0],[622.2,3.4,5.4,2.3]]
    _y_predict = knn.predict(std.transform(need_predict))
    print("_y_predict的类型", type(_y_predict), "\n预测目标种类为", _y_predict)
    print("y_predict的类型", type(y_predict), "\n预测目标", y_predict)
    print("x_test", type(y_predict), "\nx_test", x_test)
    #print("预测数据的实际结果：\n", y_test)
    # 得出准确率
    print("机器学习的准确率是：", knn.score(x_test, y_test))
    """
    # 方式二改进 ：使用网格搜索，其中使用的是交叉验证进行的
    # 构造参数的值进行搜索 cv表示几折验证
    param = {"n_neighbors":[3,4,5,6,7,8,9,10,12,14,13,11,15]}
    scv = girdSCV(knn, param_grid=param, cv=10)
    scv.fit(x_train,y_train)
    # 输出准确率
    print("测试集的准确率", scv.score(x_test, y_test))
    print("选择最好的模型",scv.best_estimator_)
    print("选择最好的结果", scv.best_score_)
    print("每次交叉验证的结果\n", scv.cv_results_)
    return None
"""

"""
# 决策树
def decision():
    # 读取数据
    data = pd.read_csv("../datasets/titanic.csv")
    # 筛选数据,找出特征值和目标值
    x = data[['pclass', 'age', 'sex']]
    y = data['survived']
    #  缺失值处理
    x['age'].fillna(x['age'].mean(),inplace= True)
    # 分割数据
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 处理数据  特征是类别 使用one-hot 编码
    dict = dictvector(sparse=False)
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print("字典：\n",dict.get_feature_names())
    x_test = dict.transform(x_test.to_dict(orient="records"))
    print("训练集数据：\n",x_train)
    dec = DecisionTreeClassifier(max_depth=8)
    dec.fit(x_train,y_train)
    # 预测准确率
    print("预测",dec.predict([[117,0,1,0,1,0]]))
    print("\n预测准确率", dec.score(x_test,y_test))
    # 导出结构树结构
    export_graphviz(dec,out_file='./tree.dot',feature_names=['年龄', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女', '男'])

    return None


if __name__== "__main__":
    # knn()
    decision()