# -*- coding: utf-8 -*-

"""MyProblem.py"""
import numpy as np
import geatpy as ea
import models1
import for_CLass1
import importlib
import matplotlib.pyplot as plt
importlib.reload(models1)
importlib.reload(for_CLass1)
from GA_Improve import *
plt.rcParams["font.family"] = "Times New Roman"

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        self.a = 0
        self.BSS_before = 0
        self.algorithm_before = 0
        self.max_X = 0
        self.max_Y = 0
        self.max_Z = 0
        name = 'BSS Charging Schedule'  # 初始化name（函数名称，可以随意设置）
        M = 3  # 初始化M（目标维数）
        # 最小化该目标
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # path = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\database\\sol_1D.csv'
        # sol_1D = pd.read_csv(path)
        Dim = 1  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        # varTypes = constraint_check_sol2(varTypes, BSSs[0], dict_para_FPC).copy()
        lb = [0] * Dim  # 决策变量下界
        # lb = constraint_check_sol2(lb, BSSs[0], dict_para_FPC).copy()
        ub = [120] * Dim  # 决策变量上界
        # ub = constraint_check_sol2(ub, BSSs[0], dict_para_FPC).copy()
        lbin = [1] * Dim  # 决策变量下边界
        # lbin = constraint_check_sol2(lbin, BSSs[0], dict_para_FPC).copy()
        ubin = [1] * Dim  # 决策变量上边界
        # ubin = constraint_check_sol2(ubin, BSSs[0], dict_para_FPC).copy()
        # 调用父类构造方法完成实例化
        # 决策变量的
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        # 检查约束条件
        # Vars = pop.Phen  # 得到决策变量矩阵
        # ObjV1 = Vars[:, 0]
        # gx = 1 + 9 * np.sum(Vars[:, 1:30], 1)
        # hx = 1 - np.sqrt(ObjV1 / gx)
        # ObjV2 = gx * hx
        TMP = []
        dict  = self.name
        EVs, BSSs = dict['EVs'], dict['BSSs']
        numEV, numBSS, numBat = len(EVs), len(BSSs), int(BSSs[0].availB[0])
        if dict['BSS_j'] != self.BSS_before or self.algorithm_before != dict['algorithm_name']:
            self.a = 0
        self.BSS_before = dict['BSS_j']
        self.algorithm_before = dict['algorithm_name']
        for i in range(len(pop.Phen)):
            _, f1, f2, f3 = cal_all_int(pop[i].Phen[0], BSSs[dict['BSS_j']], dict['dict_para_FPC'])
            tmp = [f1, f2, f3]
            TMP.append(tmp)
        pop.ObjV = np.array(TMP) # 把结果赋值给ObjV
        X = [TMP[i][0] for i in range(len(TMP))]
        Y = [TMP[i][1] for i in range(len(TMP))]
        Z = [TMP[i][2] for i in range(len(TMP))]

        if self.a == 0:
            self.max_X = max(X) + 1
            self.max_Y = max(Y) + 1
            self.max_Z = max(Z) + 1
        # power load and electricity cost
        # Pic_3d
        # if len(X) > 1:
        #     if dict['save_fig']:
        #         ax = plt.figure().add_subplot(111, projection='3d')
        #         ax.scatter(X, Y, Z, c='r', marker='o')
        #         # ax.set_xlim([-50, self.max_X])
        #         # ax.set_ylim([-1, self.max_Y ])
        #         # ax.set_zlim([-1, self.max_Z ])
        #         ax.set_xlabel('Power load')
        #         ax.set_ylabel('Electricity Cost')
        #         ax.set_zlabel('Damage')
        #         plt.title(dict['algorithm_name'] + ' BSS' + str(dict['BSS_j']) +' Repeats:'  + str(dict['repeat-th']) + ' Gen:' + str(self.a))
        #         path =  'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + dict['algorithm_name'] +'\\Picture\\pictures_3d\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(dict['repeat-th']) + '_BSS' + str(dict['BSS_j']) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + dict['algorithm_name']  + '_ref' + str(dict['ref']) +'_gen' + str(self.a) +  '.jpg'
        #         plt.savefig(path)
        #         plt.close()
        #         # Power_load and EC
        #         plt.scatter(X, Y, c='r', marker='o')
        #         # plt.xlim([-50, self.max_X])
        #         # plt.ylim([-1, self.max_Y ])
        #         plt.title(dict['algorithm_name'] + ' BSS' + str(dict['BSS_j']) +' Repeats:'  + str(dict['repeat-th']) + ' Gen:' + str(self.a) + ' Power Load and Electricity Cost')
        #         plt.xlabel('Power load')
        #         plt.ylabel('Electricity Cost')
        #         path =  'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + dict['algorithm_name'] +'\\Picture\\pictures_power_load&EC\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(dict['repeat-th']) + '_BSS' + str(dict['BSS_j']) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + dict['algorithm_name'] + '_ref' + str(dict['ref']) +'_gen' + str(self.a) +  '.jpg'
        #         plt.savefig(path)
        #         plt.close()
        #         # Power_load and EC
        #         plt.scatter(X, Z, c='r', marker='o')
        #         # plt.xlim([-50, self.max_X])
        #         # plt.ylim([-1, self.max_Z ])
        #         plt.title(dict['algorithm_name'] + ' BSS' + str(dict['BSS_j']) +' Repeats:'  + str(dict['repeat-th']) + ' Gen:' + str(self.a) + ' Power Load and Damage')
        #         plt.xlabel('Power load')
        #         plt.ylabel('Damage')
        #         path =  'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + dict['algorithm_name'] +'\\Picture\\pictures_power_load&damage\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(dict['repeat-th'])  + '_BSS' + str(dict['BSS_j']) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + dict['algorithm_name'] + '_ref' + str(dict['ref']) +'_gen' + str(self.a) +  '.jpg'
        #         plt.savefig(path)
        #         plt.close()
        #         # Damage and EC
        #         plt.scatter(Z, Y, c='r', marker='o')
        #         # plt.ylim([-1, self.max_Y])
        #         # plt.xlim([-1, self.max_Z ])
        #         plt.title(dict['algorithm_name'] + ' BSS' + str(dict['BSS_j']) +' Repeats:'  + str(dict['repeat-th']) + ' Gen:' + str(self.a) + ' Damage and Electricity Cost')
        #         plt.ylabel('Electricity Cost')
        #         plt.xlabel('Damage')
        #         path =  'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + dict['algorithm_name'] +'\\Picture\\pictures_EC&damage\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(dict['repeat-th']) + '_BSS' + str(dict['BSS_j']) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + dict['algorithm_name']  + '_ref' + str(dict['ref']) +'_gen' + str(self.a) +  '.jpg'
        #         plt.savefig(path)
        #         plt.close()
        #         self.a = self.a + 1

