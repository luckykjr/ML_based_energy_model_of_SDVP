# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import joblib
import pandas as pd


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, M):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 1  # 初始化Dim（决策变量维数）
        varTypes = [0]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1489.3]  # 决策变量下界
        ub = [1800]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界
        ubin = [1] * Dim  # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    # 定义目标函数
    def aimFunc(self, pop):  # 目标函数
        X = pop.Phen
        x0 = X[:, 0]
        # print(X)
        p = 0.5
        q = 20
        t = 32
        # 模型路径
        path = "./"
        # 加载模型
        data = np.zeros((20, 4))
        data[:, 0] = p
        x1 = data[:, 0]
        data[:, 1] = q
        data[:, 1] = 1000 * data[:, 1] / (x0*0.85)
        x2 = data[:, 1]
        data[:, 3] = x0
        x3 = data[:, 2]
        data[:, 2] = t
        x4 = data[:, 3]
        xx = np.concatenate([[x1], [x2], [x3], [x4]]).T
        # print(xx)
        data_in = pd.DataFrame(xx, columns=['压力p1', '排量V', '温度t', '转速n'])
        # print(data)
        model = joblib.load(path + 'xgb.pkl')
        value = np.expm1(model.predict(data_in))
        # print(value.shape)
        # print(type(value))
        pop.ObjV = value.reshape(-1, 1)  # 计算目标函数值，赋值给pop种群对象的ObjV属性

    def calReferObjV(self):  # Calculate the theoretic global optimal solution here.
        uniformPoint, ans = ea.crtup(self.M, 10000)  # create 10000 uniform points.
        realBestObjV = uniformPoint / 2
        return realBestObjV
