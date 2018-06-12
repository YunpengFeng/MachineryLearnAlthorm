"""
回归和画图11111
1、线性关系 y = kx + b
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import  numpy as np
#  二维画图
def drawing2D():
    plt.figure(figsize=(5, 5))
    plt.scatter([60, 72, 75, 80, 83], [126, 151.2, 157.5, 168, 174.3], [12, 2, 3, 3, 1])
    plt.ylabel('room_price')
    plt.xlabel('room_area')
    plt.show()
    return None


def drawing3D():
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.hot)
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.cm.hot)
    ax.set_zlim(-2, 2)
    # savefig('../figures/plot3d_ex.png',dpi=48)
    plt.show()

    return None


# 需求 根据波士顿地区的一些特征预测房价
from sklearn.datasets import load_boston
from sklearn.linear_model import  LinearRegression, SGDRegressor
from sklearn.model_selection import  train_test_split as tts
from sklearn.preprocessing import  StandardScaler
from sklearn.metrics import  mean_absolute_error as err

def Linear():
    # 获取数据
    lb = load_boston()
    # 分割数据
    x_train, x_test, y_train, y_test = tts(lb.data, lb.target, test_size=0.25)
    # print(y_train, 'weqw\n', y_test)
    # 标准化处理(特征值和目标值都要标准化处理) 要实例化2个的api
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    # 要转化为 二维数组
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # 正规方程
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print("LR各个特征的权重", lr.coef_)
    predict = lr.predict(x_test)
    # 要进行反标准化得到值
    print("LR预测测试集的房子价格", std_y.inverse_transform(predict))
    # 参数是 真实值 和预测值
    print("正规方程的均方误差" , err(std_y.inverse_transform(y_test), std_y.inverse_transform(predict)))
    # 梯度下降
    SGD = SGDRegressor()
    SGD.fit(x_train, y_train)
    print("SGD各个特征的权重", SGD.coef_)
    _predict = SGD.predict(x_test)
    # 要进行反标准化得到值
    print("SGD预测测试集的房子价格", std_y.inverse_transform(_predict))
    print("梯度下降的均方误差", err(std_y.inverse_transform(y_test), std_y.inverse_transform(_predict)))
    return None


if __name__ == "__main__":
    # drawing2D()
    # drawing3D()
    Linear()