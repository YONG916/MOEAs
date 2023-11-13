# Define the EV related Classes
# coding=utf-8
import numpy as np
import time
import math
import datetime
import globalvar1 as gl
import pandas as pd
import copy
import random
import bisect
import matplotlib.pyplot as plt
from for_CLass1 import BSS
from for_CLass1 import EV
from for_CLass1 import Data_analysis
from matplotlib.ticker import MaxNLocator
from models2 import *
from collections import Counter
import pprint
import matplotlib.patches as mpatches
import os.path
plt.rcParams["font.family"] = "Times New Roman"

# datasets的选择
def load_dataset_sol1(caseEV, caseBSS, dicPar):  # Tony 0515
    EV_name = ['EV_A_10.xlsx', 'EV_B_20.xlsx', 'EV_C_50.xlsx']
    BSS_name = ['BSS_3_Bat_3.xlsx', 'BSS_5_Bat_3.xlsx','BSS_5_Bat_5.xlsx', 'BSS_6_Bat_3.xlsx', 'BSS_10_Bat_3.xlsx']

    df_BSS = pd.read_excel('datasets//' + BSS_name[caseBSS])
    df_EV = pd.read_excel('datasets//' + EV_name[caseEV])

    fcnLoc = lambda a: [int(tmp) for tmp in a.split(',')]
    df_EV['location'] = df_EV['location'].apply(fcnLoc)
    df_EV['des'] = df_EV['des'].apply(fcnLoc)
    df_BSS['location'] = df_BSS['location'].apply(fcnLoc)

    # S1.2.3 BSS Class
    BSSs = []
    for iBSS in range(len(df_BSS)):
        tmp_id = df_BSS.loc[iBSS, 'BSS_id']
        BSS1 = BSS(tmp_id, df_BSS.loc[iBSS, 'location'], dicPar)
        BSS1.availB[iBSS] = df_BSS.loc[iBSS, 'init_battery']
        BSSs.append(BSS1)
        BSSs[iBSS].availB = np.ones(dicPar['T_max']) * df_BSS.loc[iBSS, 'init_battery'] #  初始化电池数目
        BSSs[iBSS].schedule = np.ones(dicPar['T_max']) * df_BSS.loc[iBSS, 'init_battery']
        BSSs[iBSS].availChargers = np.ones(dicPar['T_max']) * df_BSS.loc[iBSS, 'init_battery']  # 有多少个电池就有多少个chargers
        BSSs[iBSS].saveBattery_id = (np.ones( df_BSS.loc[iBSS, 'init_battery']) *  (-1)).tolist() # [-1, -1, -1]

    # S1.2.4 EV Class
    EVs = []
    for iEV in range(df_EV.shape[0]):
        tmp_id = df_EV.loc[iEV, 'EV_id']
        tmp_Stamptime = fcnTimeToMin(df_EV.loc[iEV, 'start_time'])
        tmp_time = fcnMinToTime(tmp_Stamptime)
        tmp_start_TS = cal_timeslot(df_EV.loc[iEV, 'start_time'])
        EV1 = EV(df_EV.loc[iEV, 'EV_id'], df_EV.loc[iEV, 'location'], tmp_Stamptime, tmp_time, tmp_start_TS, df_EV.loc[iEV, 'SOC'], df_EV.loc[iEV, 'des'])
        EVs.append(EV1)

    pd.set_option('display.max_columns',None)
    pd.set_option('display.max_rows',None)
    pd.set_option('max_colwidth',200)
    pd.set_option('expand_frame_repr', False)

    return EVs, BSSs, df_EV, df_BSS

# 计算目标函数
def cal_obj1(EVs1, BSSs1, sol, dict_para):
    numEV = len(EVs1)
    EVs, BSSs = copy.deepcopy(EVs1), copy.deepcopy(BSSs1)
    for iEV in range(numEV):
        iBSS = int(sol[iEV])
        EVs[iEV].idx_BSS = iBSS
        EVs[iEV].travel1_Min = travelTime(EVs[iEV].pos, BSSs[iBSS].pos, dict_para['EV_Speed'])
        EVs[iEV].travel2_Min = travelTime(EVs[iEV].des, BSSs[iBSS].pos, dict_para['EV_Speed'])
        EVs[iEV].arr_Min = int(EVs[iEV].start_Min + EVs[iEV].travel1_Min) + 1
        EVs[iEV].arr_Time = fcnMinToTime(EVs[iEV].arr_Min)
        EVs[iEV].arr_SOC = round(EVs[iEV].SOC - cal_dist(EVs[iEV].pos, BSSs[iBSS].pos) * dict_para['EV_Consume'], 2)

    arr_Mins = [EV1.arr_Min for EV1 in EVs]
    iEV_Sort = np.argsort(arr_Mins)
    for iEV in iEV_Sort:
        idx_BSS = EVs[iEV].idx_BSS
        for ii in range(EVs[iEV].arr_Min, dict_para['T_max']):
            # 当前时间，换电站仓位是否为空
            if ii > BSSs[idx_BSS].endingUse_Min:
                # 换电站是否有多余的电池
                if BSSs[idx_BSS].availB[ii] > 0:
                    # if idx_BSS == 3:
                    #     print(iEV)
                    # 操作时间开始前，EV 开始换电的时间(车一开进去，就开始换电)
                    EVs[iEV].swap_start_time = ii
                    # 操作时间之后， 新电池换到车上， 可用电池数目减一,
                    BSSs[idx_BSS].availB[(ii + EVs[iEV].operate_Min) :] -= 1
                    # 换的是哪个EV换下来的电池
                    EVs[iEV].swap_battery_id = BSSs[idx_BSS].saveBattery_id[0] # 新电池换上去
                    # saveBattery_id 减去 1, 因为当前电池id被换走
                    del(BSSs[idx_BSS].saveBattery_id[0])
                    # 旧的电池开始充电，计算电池充电时间
                    time_charge = int((dict_para['EV_Capacity']  * 60 * (1 - 0.01 * EVs[iEV].arr_SOC)) / dict_para['charge_Power']) + 1
                    # 旧的电池充满电，可用电池数目加一
                    BSSs[idx_BSS].availB[(ii + time_charge + EVs[iEV].operate_Min):] += 1
                    # 旧的电池充满电, saveBattery_id加上EV换下来的电池 对应的id
                    BSSs[idx_BSS].saveBattery_id.append(iEV)
                    # 操作时间之后，记录电池开始充电的时间
                    EVs[iEV].Battery_startCharge_Min = ii + EVs[iEV].operate_Min
                    EVs[iEV].Battery_startCharge_Timeslot = cal_MinToTimeslot(EVs[iEV].Battery_startCharge_Min, dict_para)
                    # 记录电池结束充电的时间
                    EVs[iEV].Battery_endCharge_Min = EVs[iEV].Battery_startCharge_Min  + time_charge -  1
                    EVs[iEV].Battery_endCharge_Timeslot = cal_MinToTimeslot(EVs[iEV].Battery_endCharge_Min, dict_para)
                    # 等待时间（包含5min操作时间）
                    EVs[iEV].wait_Min = ii - EVs[iEV].arr_Min + EVs[iEV].operate_Min
                    # EV离开BSS 的时间
                    EVs[iEV].depart_Min = EVs[iEV].arr_Min + EVs[iEV].wait_Min
                    EVs[iEV].depart_Time = fcnMinToTime(EVs[iEV].depart_Min)
                    # EV到达目的地的时间
                    EVs[iEV].finish_Min = EVs[iEV].depart_Min + EVs[iEV].travel2_Min
                    EVs[iEV].finish_Time = fcnMinToTime(EVs[iEV].finish_Min)
                    # EV 直达的时间
                    EVs[iEV].travel_Direct_Min = travelTime(EVs[iEV].pos, EVs[iEV].des, dict_para['EV_Speed'])
                    # EV 的目标函数
                    EVs[iEV].obj1 = EVs[iEV].travel1_Min + EVs[iEV].travel2_Min + EVs[iEV].wait_Min - EVs[iEV].travel_Direct_Min
                    # EV离开的时间 = 换电仓结束使用的时间
                    BSSs[idx_BSS].endingUse_Min = EVs[iEV].depart_Min

                    break

    obj1 = round(sum([EV1.obj1 for EV1 in EVs]), 2) / numEV
    return obj1, EVs, BSSs




# 1.运行算法
def Run_NIR(EVs1, BSSs1, dict_para):
    print('\nNIR Repetition: 0')
    EVs, BSSs = EVs1.copy(), BSSs1.copy()
    sol_NIR = fcn_sol_NIR(EVs, BSSs, dict_para)
    obj1_NIR, EVs_NIR, BSSs_NIR= cal_obj1(EVs, BSSs, sol_NIR, dict_para)
    EVs_NIR = cal_Battery_chargingTimeslot(EVs_NIR, dict_para)
    resNIR_save = {'save_obj': [obj1_NIR], 'save_sol': [sol_NIR], 'save_iter': [], 'save_time': [], 'save_EVs':[EVs_NIR], 'algorithm_type':'NIR', 'save_BSSs':[BSSs_NIR], 'dict_para':dict_para}
    return  resNIR_save

def Run_GA(EVs1, BSSs1, dict_para_GA):
    EVs, BSSs = EVs1.copy(), BSSs1.copy()
    dict_para = dict_para_GA['dict_para']
    save_obj, save_sol, save_iter, save_time, save_EVs, save_BSSs = [], [], [], [], [], []
    print('\n')
    for ii in range(dict_para['num_repeats']):
        print('GA Repetition: %d'%ii)
        startTime_GA = time.time()
        sol_GA, iter_GA = fcn_sol_GA(EVs, BSSs, dict_para_GA)
        obj1_GA, EVs_GA, BSSs_GA = cal_obj1(EVs, BSSs, sol_GA, dict_para)
        endTime_GA = time.time()
        # Save
        EVs_GA = cal_Battery_chargingTimeslot(EVs_GA, dict_para)
        save_obj.append(obj1_GA)
        save_sol.append(sol_GA)
        save_iter.append(iter_GA)
        save_time.append(endTime_GA - startTime_GA)
        save_EVs.append(EVs_GA)
        save_BSSs.append(BSSs_GA)

    resGA_save = {'save_obj': save_obj, 'save_sol': save_sol, 'save_iter': save_iter, 'save_time': save_time, 'save_EVs':save_EVs, 'algorithm_type':'GA', 'save_BSSs':save_BSSs, 'dict_para':dict_para}
    '==========================保存到表格中================================'
    Save_results(resGA_save)
    '=========================================================='
    return  resGA_save


def cal_Battery_chargingTimeslot(EVs1, dict_para):
    time_slot = dict_para['time_slot']
    EVs = EVs1.copy()

    for iEV in range(len(EVs)):
        # Battery 起始充电的时间
        time_tmp = EVs[iEV].Battery_startCharge_Min
        while time_tmp % time_slot != 0:
            time_tmp = time_tmp + 1
        # 29 变成 30了，
        # print(EVs[iEV].Battery_startCharge_Min, time_tmp)
        EVs[iEV].Battery_startCharge_Timeslot = int(time_tmp / time_slot)
        # battery 结束充电的时间
        time_tmp = EVs[iEV].Battery_endCharge_Min
        while time_tmp % time_slot != 0:
            time_tmp = time_tmp - 1
        # 29 变成 25 了，
        EVs[iEV].Battery_endCharge_Timeslot = int(time_tmp / time_slot)
        # print(EVs[iEV].Battery_endCharge_Min, time_tmp)
    return EVs

def Run_saved_result(i_alg, dict_para):
    path = 'Result\\Solution1\\dict_alg_result\\EVs' + str(len(dict_para['EVs'])) + '\\dict_' + i_alg + '\\' + i_alg  + '.csv'
    path_BSS = 'Result\\Solution1\\dict_alg_result\\EVs' + str(len(dict_para['EVs'])) + '\\dict_' + i_alg + '\\' + i_alg + '_BSS'  + '.csv'
    df, df_BSS = pd.read_csv(path, index_col= 0), pd.read_csv(path_BSS, index_col= 0)
    df, df_BSS = df.reset_index(drop=True), df_BSS.reset_index(drop=True)
    if len(df) >= dict_para['num_repeats']:
        # 为了让我的数据看上去更好看？
        if i_alg == 'ATS':
            # 选择最小的前三位
            # id = np.argsort(df['save_obj'].values)[:dict_para['num_repeats']] # This is a list
            # df, df_BSS = df.iloc[id].reset_index(drop = True), df_BSS.iloc[id].reset_index(drop = True)
            # 只选择最后一次的运行结果
            a1,a2 = -17,-2
            df, df_BSS = df.iloc[a1:a2].reset_index(drop = True), df_BSS.iloc[a1:a2].reset_index(drop = True)
        elif i_alg == 'TS':
            # 选择最大的前三位
            id = (np.argsort((-1) * df['save_obj'].values)[:dict_para['num_repeats']]).tolist() # This is a list
            # id.append(np.argmin(df['save_obj'].values))
            id = np.array(id)
            # 随机选择3位
            # id =
            df, df_BSS = df.iloc[id].reset_index(drop = True), df_BSS.iloc[id].reset_index(drop = True)
        elif i_alg == 'ISA':
            # 选择最大的前三位
            id = (np.argsort((-1) * df['save_obj'].values)[:dict_para['num_repeats']]).tolist() # This is a list
            # id.append(np.argmin(df['save_obj'].values))
            id = np.array(id)
            # 随机选择3位
            # id =
            df, df_BSS = df.iloc[id].reset_index(drop = True), df_BSS.iloc[id].reset_index(drop = True)

        else:
            df, df_BSS = df.iloc[:dict_para['num_repeats']], df_BSS.iloc[:dict_para['num_repeats']]
        save_EVs, save_BSSs = Get_saveEVs_saveBSSs(df,df_BSS, dict_para)
        save_sol, save_iter = Get_saveSol_saveIter(df)
        res_save = {'save_obj': df['save_obj'].values.tolist(), 'save_sol': save_sol, 'save_iter': save_iter, 'save_time': df['save_time'].values.tolist(), 'save_EVs':save_EVs, 'algorithm_type':i_alg, 'save_BSSs':save_BSSs, 'dict_para':dict_para}
        return res_save
    else:
        print('='*50)
        print('No enough solutions! ' + i_alg + ' Only have ' + str(len(df)) + ' solutions!')
        print('=' * 50)


def out_data_analysis_sol1(res_save):
    # 初始化
    # ’res_save‘ is a list, Represents each algorithm result
    dict_para = res_save[0]['dict_para']
    Print_csv, num_repeats = dict_para['Print_csv'], dict_para['num_repeats']
    numEV, numBSS, numBat = len(res_save[0]['save_EVs'][0]), len(res_save[0]['save_BSSs'][0]), int(res_save[0]['save_BSSs'][0][0].availB[0])
    Algorithm = []
    name = ''
    for i in range(len(res_save)):
        resGA_save = res_save[i].copy()
        name = name  + '_' + resGA_save['algorithm_type']
        # NIR and GS:
        if resGA_save['algorithm_type'] == 'NIR' or resGA_save['algorithm_type'] == 'GS':
            if resGA_save['save_EVs'][0] == '/':
                Algorithm1 = Data_analysis(resGA_save['algorithm_type'], '运行时间过长！', '/', '/', '/', '/', '/', '/', resGA_save['save_sol'][0])
                # 为 Solution2 保存数据
                path_best_sol1 = 'Result\\Solution1\\solution1_for_solution2\\' + 'EVs' + str(numEV) + '_BSSs' + str(
                    numBSS) + '_Bats' + str(numBat)  + '_' + resGA_save['algorithm_type'] + '.csv'
                # GS 算法的长度
                if numBSS <= 2:
                    np.savetxt(path_best_sol1, resGA_save['save_sol'][0], delimiter=',')

            else:
                Algorithm1 = Data_analysis(resGA_save['algorithm_type'], 1, resGA_save['save_obj'][0], '/', '/', '/','/', '/', resGA_save['save_sol'][0])
                # 为 Solution2 保存数据
                path_best_sol1 = 'Result\\Solution1\\solution1_for_solution2\\' + 'EVs' + str(numEV) + '_BSSs' + str(
                    numBSS) + '_Bats' + str(numBat)  + '_' + resGA_save['algorithm_type'] + '.csv'
                np.savetxt(path_best_sol1, resGA_save['save_sol'][0], delimiter=',')

                df_sol_GA = output_df_sol(resGA_save['save_EVs'][0])
            Algorithm.append(Algorithm1)
        # Other Algorithm
        else:
            aver_GA, SD_GA, max_value, min_value, median_value = aver_SD(resGA_save['save_obj'])
            # 最优解的id
            best_sol_id = np.argmin(resGA_save['save_obj'])
            # 最优解的运行时间、Solution、EVs
            best_run_time, best_solution, best_EVs = resGA_save['save_time'][best_sol_id], resGA_save['save_sol'][best_sol_id], resGA_save['save_EVs'][best_sol_id]
            # 为 Solution2 保存数据
            path_best_sol1 = 'Result\\Solution1\\solution1_for_solution2\\' + 'EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat)  + '_' + resGA_save['algorithm_type'] + '.csv'
            np.savetxt(path_best_sol1, best_solution, delimiter=',')
            # EV 读入 df ,再读入到表格中
            df_sol_GA = output_df_sol(best_EVs)
            # 读入Class
            Algorithm1 = Data_analysis(resGA_save['algorithm_type'],num_repeats, aver_GA,best_run_time,SD_GA,max_value, min_value, median_value, best_solution)
            Algorithm.append(Algorithm1)
        # 打印结果在控制台
        if dict_para['show_class']:
            print('\n')
            pprint.pprint(Algorithm1.__dict__)
        # 保存 EV 的数据
        if Print_csv:
            file_path = 'Result//Solution1//res//'  + dict_para['time_stamp'] + 'Repeats' + str(num_repeats)  + '_' +  resGA_save['algorithm_type'] + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '.csv'
            df_sol_GA.to_csv(file_path, index=False)
    # algorithm result  读入 df ,再读入到表格中
    df_sol_algorithm = output_df_sol(Algorithm)
    # 保存 algorithm 的数据
    if Print_csv:
        file_path = 'Result//Solution1//algor_res//'  + dict_para['time_stamp'] + 'Repeats' + str(num_repeats) +  '_' + name  + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '.csv'
        df_sol_algorithm.to_csv(file_path, index=False)


def Run_MOPSO_eachBSS_int(dict_para_PSO):
    resGA_save, resGA1_save, resGA_save_eachBSS = [], {}, []
    # GA1, GA2, GA3, GA4, GA5, GA6, GA7  = dict_para_GA['GA_type']
    # 最多车的站,z作为算法的迭代结果输出
    # id_BSS = max(set(dict['sol1']), key= dict['sol1'].count)
    # print('\n最多车BSS%d'%id_BSS)
    sol, save_BSSs, save_EVs, save_obj, save_sol, save_iter, save_iter_load, save_iter_EC, save_iter_damage, save_time, save_load, save_EC, save_damage = [], [], [], [], [], [], [], [], [], [], [], [], []
    EVs, BSSs, Bats = dict_para_PSO['EVs'], dict_para_PSO['BSSs'], dict_para_PSO['Bats']
    numEV, numBSS, numBat = len(EVs), len(BSSs), int(BSSs[0].availB[0])
    dict_para =  dict_para_PSO['dict_para']
    for ii in range(dict_para_PSO['num_repeats']):
        print('=' * 50, '\n', 'Repeat:', ii)
        # 下面只是BSS0的帕累托前沿
        for iBSS in dict_para['BSS_set']:
            if BSSs[iBSS].Battery_ID != []:

                if dict_para['Read_PF']:
                    print('=' * 50)
                    print('Read True Pareto Front!')
                    path = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\PF\\EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_sol1' + dict_para['sol1_algorithm_type'] + '_BSS' + str(iBSS) + '.csv'
                    dict_para_PSO['PF'] = pd.read_csv(path).values[:, -3:]
                    print('=' * 50)

                print('MOPSO: BSS%d'%iBSS)
                # 改变problem 的参数
                dict_para_PSO['repeat-th'] = ii
                # 输出非支配解集
                start_time = time.time()
                # 生成非支配解 W
                dict_result = fcn_sol2_MOPSO_int(iBSS, dict_para_PSO)
                # dict_result = fcn_sol2_MOPSO_int(iBSS, dict_para_PSO)
                end_time = time.time()
                print('running time:',(end_time - start_time))
                dict_result['Run_time'] = end_time - start_time
                Save_csv(dict_result, dict_para_PSO)


def fcn_sol2_MOPSO_int(id_BSS_j, dict_para_PSO):
    # 外部档案规模是NP
    # 1.读取数据
    EVs, BSSs, Bats = dict_para_PSO['EVs'], dict_para_PSO['BSSs'], dict_para_PSO['Bats']
    dict_para = dict_para_PSO['dict_para']
    PF = dict_para_PSO['PF']
    time_stamp, Gen, NP ,c1, c2, w = dict_para['time_stamp'], dict_para_PSO['Gen'], dict_para_PSO['NP'], dict_para_PSO['c1'], dict_para_PSO['c2'], dict_para_PSO['w']
    x_obj, x_obj_load, x_obj_EC, x_obj_damage = [], [], [], []
    dict_para_PSO['BSS_j'] = BSSs[id_BSS_j]
    BSS_j = BSSs[id_BSS_j]
    iter_best, x, W, HV, IGD, GD, Spacing, numW_iter = [], [], [], [], [], [], [], []
    # 2. 初始化粒子的速度与位置
    V_max, V_min = int(dict_para['max_power_load'] / dict_para['N']) / 2, -1 * int(dict_para['max_power_load'] / dict_para['N']) / 2
    X_max, X_min = int(dict_para['max_power_load'] / dict_para['N']), 0
    dict_para_PSO['V_max'], dict_para_PSO['V_min'], dict_para_PSO['X_max'], dict_para_PSO['X_min'] = V_max, V_min, X_max, X_min
    #  随机初始化种群个体(限定位置和速度)
    for i in range(NP):
        sol_1D = create_sol_1D(BSS_j, dict_para_PSO)
        # 计算每个子目标的目标函数值
        x_obj_tmp, x_obj_load_tmp , x_obj_EC_tmp, x_obj_damage_tmp = cal_all_int(sol_1D, BSS_j, dict_para_PSO)
        x.append(sol_1D)
        x_obj.append(x_obj_tmp)
        x_obj_load.append(x_obj_load_tmp)
        x_obj_EC.append(x_obj_EC_tmp)
        x_obj_damage.append(x_obj_damage_tmp)
    x ,x_obj, x_obj_load, x_obj_EC, x_obj_damage = np.array(x, dtype=float), np.array(x_obj, dtype=float), np.array(x_obj_load, dtype=float), np.array(x_obj_EC, dtype=float),np.array(x_obj_damage, dtype=float)
    v, v_archive, gbest = x.copy(), x.copy(), x.copy()
    # 把子目标函数合并到解的后面
    x = np.c_[x, x_obj_load]
    x = np.c_[x, x_obj_EC]
    x = np.c_[x, x_obj_damage]
    pbest = x.copy() # 局部最优
    NP = len(x)
    # 开始迭代
    for gen in range(Gen):
        # 扰动发生的概率
        numW_iter.append(len(W))
        print('Gen:', gen, '当前非支配解的个数len(W):', len(W))
        # 构建非支配解集(不断比较目标函数值)
        # 竞争机制：通过竞争机制快速比较种群个体之间的支配关系，采用外部档案集来存储找到的非支配解
        N, v_N = cal_non_dominated_solution(x, v)
        if gen != 0:
            # W, v_archive = cal_crowded_distance(v_N,N,v_archive,W)
            W = cal_crowded_distance1(N, W)
        else:
            W, v_archive = N.copy(), v_N.copy()
        # # Archive 产生变异
        # W, v_archive = Archive_Mutation(gen, W, v_archive, dict_para_PSO)
        #  限制外部档案规模，截断外部档案集
        if len(W) > NP:
            W = W[:NP, :]
            v_archive = v_archive[:NP, :]
        # 寻找pbest:在种群里面寻找支配度最高的
        pbest = cal_pbest(x, pbest, dict_para_PSO)
        dict_para_PSO['W'], dict_para_PSO['pbest'], dict_para_PSO['gbest']  = W, pbest, gbest
        x, v = Particle_update(gen,x,v,dict_para_PSO)
        # 计算各个评价指标
        # 计算一下HV
        if dict_para['Read_PF']:
            HV.append(ea.indicator.HV(W[:NP, -3:], PF))
            GD.append(ea.indicator.GD(W[:NP, -3:], PF))
            IGD.append(ea.indicator.IGD(W[:NP, -3:], PF))
        Spacing.append(ea.indicator.Spacing(W[:NP, -3:]))


    dict_result =  {'id_BSS':id_BSS_j, 'Alg_name':'MOPSO', 'W':W, 'Feasible_Solution': numW_iter, 'HV':HV, 'IGD':IGD, 'GD':GD, 'SP':Spacing}
    return dict_result

def Run_NSGA2_eachBSS_int(problem):
    resGA_save, resGA1_save, resGA_save_eachBSS = [], {}, []
    # GA1, GA2, GA3, GA4, GA5, GA6, GA7  = dict_para_GA['GA_type']
    dict = problem.name
    problem.name['algorithm_name'] = 'NSGA2'
    dict_para = dict['dict_para']
    num_repeats = dict['num_repeats']
    numEV, numBSS = len(dict_para['EVs']), len(dict_para['BSSs'])
    numBat = int(dict_para['BSSs'][0].availB[0])
    EVs, BSSs, Bats = dict['EVs'],dict['BSSs'], dict['Bats']
    # 最多车的站,z作为算法的迭代结果输出
    # id_BSS = max(set(dict['sol1']), key= dict['sol1'].count)
    # print('\n最多车BSS%d'%id_BSS)
    sol, save_BSSs, save_EVs, save_obj, save_sol, save_iter, save_iter_load, save_iter_EC, save_iter_damage, save_time, save_load, save_EC, save_damage = [], [], [], [], [], [], [], [], [], [], [], [], []
    for ii in range(num_repeats):
        print('=' * 50, '\n', 'Repeat:', ii)
        for iBSS in dict_para['BSS_set']:
            if BSSs[iBSS].Battery_ID != []:
                print('NSGA2: BSS%d'%iBSS)
                # NSGA2

                if dict_para['Read_PF']:
                    print('=' * 50)
                    print('Read True Pareto Front!')
                    path = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\PF\\EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_sol1' + dict_para['sol1_algorithm_type'] + '_BSS' + str(iBSS) + '.csv'
                    problem.name['PF'] = pd.read_csv(path).values[:, -3:]
                    print('=' * 50)

                # 改变problem 的参数
                problem.name['BSS_j'] = iBSS
                problem.name['repeat-th'] = ii
                SOL, OBJ = creat_population(problem)
                problem.varTypes = np.zeros(len(SOL[0]))
                problem.borders = np.ones([2, len(SOL[0])])
                problem.ranges = np.vstack([np.zeros(len(SOL[0])), np.ones(len(SOL[0])) * 120])
                # 重置Field
                Field = ea.crtfld(problem.name['Encoding'], problem.varTypes, problem.ranges,problem.borders)  # 创建区域描述器
                # 生成新的种群
                population = ea.Population(dict['Encoding'], Field, dict['NP'], SOL, OBJ, problem)
                # 运行算法
                myAlgorithm = ea.moea_NSGA2_templet(problem, population)
                myAlgorithm.MAXGEN = dict['Gen']  # 最大进化代数
                myAlgorithm.logTras = 1  # 设置每多少代记录日志，若设置成0则表示不记录日志
                myAlgorithm.verbose = True  # 设置是否打印输出日志信息
                myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
                start_time = time.time()
                [NDSet, _] = myAlgorithm.run()
                end_time = time.time()
                print('running time:',(end_time - start_time))
                # #encode
                if dict['save_csv']:
                    df_obj = pd.DataFrame({'Power load': NDSet.ObjV[:, 0], 'Electricity Cost': NDSet.ObjV[:, 1], 'Damage': NDSet.ObjV[:, 2]})
                    df_sol_set = pd.DataFrame(NDSet.Chrom)
                    path_obj = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + problem.name['algorithm_name'] + '\\Non_dominated_solution_set\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(ii) + '_BSS' + str(dict['BSS_j']) +'_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + problem.name['algorithm_name'] + '_ref' + str(problem.name['ref']) + '_encode' + '.csv'
                    path_sol = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + problem.name['algorithm_name'] + '\\res_f1_f2_f3\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(ii) + '_BSS' + str(dict['BSS_j']) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + problem.name['algorithm_name'] + '_ref' + str(problem.name['ref']) + '_encode' + '.csv'
                    df_obj.to_csv(path_sol)
                    df_sol_set.to_csv(path_obj)
                    # 保存运行时间
                    path = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + problem.name['algorithm_name'] + '\\Evaluation\\Run_time\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(ii) + '_BSS' + str(dict['BSS_j']) +'_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + problem.name['algorithm_name'] + '_ref' + str(problem.name['ref']) + '_encode' + '.csv'
                    df = pd.DataFrame([end_time - start_time])
                    df.to_csv(path)


def Run_RVEA_eachBSS_int(problem):
    resGA_save, resGA1_save, resGA_save_eachBSS = [], {}, []
    # GA1, GA2, GA3, GA4, GA5, GA6, GA7  = dict_para_GA['GA_type']
    dict = problem.name
    problem.name['algorithm_name'] = 'RVEA'
    dict_para = dict['dict_para']
    num_repeats = dict['num_repeats']
    numEV, numBSS = len(dict_para['EVs']), len(dict_para['BSSs'])
    numBat = int(dict_para['BSSs'][0].availB[0])
    EVs, BSSs, Bats = dict['EVs'],dict['BSSs'], dict['Bats']
    # 最多车的站,z作为算法的迭代结果输出
    # id_BSS = max(set(dict['sol1']), key= dict['sol1'].count)
    # print('\n最多车BSS%d'%id_BSS)
    sol, save_BSSs, save_EVs, save_obj, save_sol, save_iter, save_iter_load, save_iter_EC, save_iter_damage, save_time, save_load, save_EC, save_damage = [], [], [], [], [], [], [], [], [], [], [], [], []
    for ii in range(num_repeats):
        print('=' * 50, '\n', 'Repeat:', ii)
        for iBSS in dict_para['BSS_set']:
            if BSSs[iBSS].Battery_ID != []:
                print('RVEA: BSS%d'%iBSS)
                # RVEA

                if dict_para['Read_PF']:
                    print('=' * 50)
                    print('Read True Pareto Front!')
                    path = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\PF\\EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_sol1' + dict_para['sol1_algorithm_type'] + '_BSS' + str(iBSS) + '.csv'
                    problem.name['PF'] = pd.read_csv(path).values[:,-3:]
                    print('=' * 50)

                # 改变problem 的参数
                problem.name['BSS_j'] = iBSS
                problem.name['repeat-th'] = ii
                SOL, OBJ = creat_population(problem)
                problem.varTypes = np.zeros(len(SOL[0]))
                problem.borders = np.ones([2, len(SOL[0])])
                problem.ranges = np.vstack([np.zeros(len(SOL[0])), np.ones(len(SOL[0])) * 120])
                # 重置Field
                Field = ea.crtfld(problem.name['Encoding'], problem.varTypes, problem.ranges,problem.borders)  # 创建区域描述器
                # 生成新的种群
                population = ea.Population(dict['Encoding'], Field, dict['NP'], SOL, OBJ, problem)
                myAlgorithm = ea.moea_RVEA_templet(problem, population)
                myAlgorithm.MAXGEN = dict['Gen']  # 最大进化代数
                myAlgorithm.logTras = 1  # 设置每多少代记录日志，若设置成0则表示不记录日志
                myAlgorithm.verbose = True  # 设置是否打印输出日志信息
                myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
                start_time = time.time()
                [NDSet, _] = myAlgorithm.run()
                end_time = time.time()
                print('running time:', (end_time - start_time))
                #encode
                if dict['save_csv']:
                    df_obj = pd.DataFrame({'Power load': NDSet.ObjV[:, 0], 'Electricity Cost': NDSet.ObjV[:, 1], 'Damage': NDSet.ObjV[:, 2]})
                    df_sol_set = pd.DataFrame(NDSet.Chrom)
                    path_obj = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + problem.name['algorithm_name'] + '\\Non_dominated_solution_set\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(ii) + '_BSS' + str(dict['BSS_j']) +'_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + problem.name['algorithm_name'] + '_ref' + str(problem.name['ref']) + '_encode' + '.csv'
                    path_sol = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + problem.name['algorithm_name'] + '\\res_f1_f2_f3\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(ii) + '_BSS' + str(dict['BSS_j']) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + problem.name['algorithm_name'] + '_ref' + str(problem.name['ref']) + '_encode' + '.csv'
                    df_obj.to_csv(path_sol)
                    df_sol_set.to_csv(path_obj)
                    # 保存运行时间
                    path = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + problem.name['algorithm_name'] + '\\Evaluation\\Run_time\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(ii) + '_BSS' + str(dict['BSS_j']) +'_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + problem.name['algorithm_name'] + '_ref' + str(problem.name['ref']) + '_encode' + '.csv'
                    df = pd.DataFrame([end_time - start_time])
                    df.to_csv(path)

def Run_NSGA3_eachBSS_int(problem):
    resGA_save, resGA1_save, resGA_save_eachBSS = [], {}, []
    # GA1, GA2, GA3, GA4, GA5, GA6, GA7  = dict_para_GA['GA_type']
    dict = problem.name
    problem.name['algorithm_name'] = 'NSGA3'
    dict_para = dict['dict_para']
    num_repeats = dict['num_repeats']
    numEV, numBSS = len(dict_para['EVs']), len(dict_para['BSSs'])
    numBat = int(dict_para['BSSs'][0].availB[0])
    EVs, BSSs, Bats = dict['EVs'],dict['BSSs'], dict['Bats']
    # 最多车的站,z作为算法的迭代结果输出
    # id_BSS = max(set(dict['sol1']), key= dict['sol1'].count)
    # print('\n最多车BSS%d'%id_BSS)
    sol, save_BSSs, save_EVs, save_obj, save_sol, save_iter, save_iter_load, save_iter_EC, save_iter_damage, save_time, save_load, save_EC, save_damage = [], [], [], [], [], [], [], [], [], [], [], [], []
    for ii in range(num_repeats):
        print('=' * 50, '\n', 'Repeat:', ii)
        for iBSS in dict_para['BSS_set']:
            if BSSs[iBSS].Battery_ID != []:
                print('NSGA3: BSS%d'%iBSS)
                # NSGA3

                if dict_para['Read_PF']:
                    print('=' * 50)
                    print('Read True Pareto Front!')
                    path = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\PF\\EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_sol1' + dict_para['sol1_algorithm_type'] + '_BSS' + str(iBSS) + '.csv'
                    problem.name['PF'] = pd.read_csv(path).values[:,-3:]
                    print('=' * 50)

                # 改变problem 的参数
                problem.name['BSS_j'] = iBSS
                problem.name['repeat-th'] = ii
                SOL, OBJ = creat_population(problem)
                problem.varTypes = np.zeros(len(SOL[0]))
                problem.borders = np.ones([2, len(SOL[0])])
                problem.ranges = np.vstack([np.zeros(len(SOL[0])), np.ones(len(SOL[0])) * 120])
                # 重置Field
                Field = ea.crtfld(problem.name['Encoding'], problem.varTypes, problem.ranges,problem.borders)  # 创建区域描述器
                # 生成新的种群
                population = ea.Population(dict['Encoding'], Field, dict['NP'], SOL, OBJ, problem)
                myAlgorithm = ea.moea_NSGA3_templet(problem, population)
                myAlgorithm.MAXGEN = dict['Gen']  # 最大进化代数
                myAlgorithm.logTras = 1  # 设置每多少代记录日志，若设置成0则表示不记录日志
                myAlgorithm.verbose = True  # 设置是否打印输出日志信息
                myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
                start_time = time.time()
                [NDSet, _] = myAlgorithm.run()
                end_time = time.time()
                print('running time:', (end_time - start_time))
                #encode
                if dict['save_csv']:
                    df_obj = pd.DataFrame({'Power load': NDSet.ObjV[:, 0], 'Electricity Cost': NDSet.ObjV[:, 1], 'Damage': NDSet.ObjV[:, 2]})
                    df_sol_set = pd.DataFrame(NDSet.Chrom)
                    path_obj = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + problem.name['algorithm_name'] + '\\Non_dominated_solution_set\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(ii) + '_BSS' + str(dict['BSS_j']) +'_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + problem.name['algorithm_name'] + '_ref' + str(problem.name['ref']) + '_encode' + '.csv'
                    path_sol = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + problem.name['algorithm_name'] + '\\res_f1_f2_f3\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(ii) + '_BSS' + str(dict['BSS_j']) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + problem.name['algorithm_name'] + '_ref' + str(problem.name['ref']) + '_encode' + '.csv'
                    df_obj.to_csv(path_sol)
                    df_sol_set.to_csv(path_obj)
                    # 保存运行时间
                    path = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + problem.name['algorithm_name'] + '\\Evaluation\\Run_time\\Gen' + str(dict['Gen']) + '_NP' + str(dict['NP'])  + '_Repeats' + str(ii) + '_BSS' + str(dict['BSS_j']) +'_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + problem.name['algorithm_name'] + '_ref' + str(problem.name['ref']) + '_encode' + '.csv'
                    df = pd.DataFrame([end_time - start_time])
                    df.to_csv(path)




def Save_csv(dict_result,  dict_para_PSO):
    # Encode
    W, Alg_name, id_BSS = dict_result['W'], dict_result['Alg_name'], dict_result['id_BSS']
    File_name = 'Result\\Solution2\\Total_BSS\\Whole_factor\\pareto_result\\' + Alg_name + '\\'
    EVs, BSSs, Bats = dict_para_PSO['EVs'], dict_para_PSO['BSSs'], dict_para_PSO['Bats']
    numEV, numBSS, numBat = len(EVs), len(BSSs), int(BSSs[0].availB[0])
    HV, IGD, GD, Spacing, Run_time, numFeasible_sol = pd.DataFrame(dict_result['HV']), pd.DataFrame(dict_result['IGD']), pd.DataFrame(dict_result['GD']), pd.DataFrame(dict_result['SP']), pd.DataFrame([dict_result['Run_time']]), pd.DataFrame(dict_result['Feasible_Solution'])
    dict_para = dict_para_PSO['dict_para']
    if dict_para_PSO['save_csv']:
        #  1.保存帕累托解集
        df_obj = pd.DataFrame({'Power load': W[:, -3], 'Electricity Cost': W[:, -2], 'Damage': W[:, -1]})
        df_sol_set = pd.DataFrame(W[:, :-3])
        path_obj =  File_name + 'Non_dominated_solution_set\\Gen' + str(dict_para_PSO['Gen']) + '_NP' + str(dict_para_PSO['NP']) + '_Repeats' + str(dict_para_PSO['repeat-th'] ) + '_BSS' + str(id_BSS) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + Alg_name + '_ref' + str(0) + '_encode' + '.csv'
        path_sol = File_name + 'res_f1_f2_f3\\Gen' + str(dict_para_PSO['Gen']) + '_NP' + str(dict_para_PSO['NP']) + '_Repeats' + str(dict_para_PSO['repeat-th'] ) + '_BSS' + str(id_BSS) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + Alg_name + '_ref' + str(0) + '_encode' + '.csv'
        df_obj.to_csv(path_sol)
        df_sol_set.to_csv(path_obj)
        if dict_para['Read_PF']:
            # 2.保存HV
            path = File_name + 'Evaluation\\HV\\Gen' + str(dict_para_PSO['Gen']) + '_NP' + str(dict_para_PSO['NP']) + '_Repeats' + str(dict_para_PSO['repeat-th'] ) + '_BSS' + str(id_BSS) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + Alg_name + '_ref' + str(0) + '_encode' + '.csv'
            HV.to_csv(path)
            # 3.保存GD
            path = File_name + 'Evaluation\\GD\\Gen' + str(dict_para_PSO['Gen']) + '_NP' + str(dict_para_PSO['NP']) + '_Repeats' + str(dict_para_PSO['repeat-th'] ) + '_BSS' + str(id_BSS) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + Alg_name + '_ref' + str(0) + '_encode' + '.csv'
            GD.to_csv(path)
            # 4.保存IGD
            path = File_name + 'Evaluation\\IGD\\Gen' + str(dict_para_PSO['Gen']) + '_NP' + str(dict_para_PSO['NP']) + '_Repeats' + str(dict_para_PSO['repeat-th'] ) + '_BSS' + str(id_BSS) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + Alg_name + '_ref' + str(0) + '_encode' + '.csv'
            IGD.to_csv(path)
        # 5.保存Spacing
        path = File_name + 'Evaluation\\SP\\Gen' + str(dict_para_PSO['Gen']) + '_NP' + str(dict_para_PSO['NP']) + '_Repeats' + str(dict_para_PSO['repeat-th'] ) + '_BSS' + str(id_BSS) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + Alg_name + '_ref' + str(0) + '_encode' + '.csv'
        Spacing.to_csv(path)
        # 6.保存运行时间
        path = File_name + 'Evaluation\\Run_time\\Gen' + str(dict_para_PSO['Gen']) + '_NP' + str(dict_para_PSO['NP']) + '_Repeats' + str(dict_para_PSO['repeat-th'] ) + '_BSS' + str(id_BSS) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + Alg_name + '_ref' + str(0) + '_encode' + '.csv'
        Run_time.to_csv(path)
        # 7.保存可行解
        path = File_name + 'Evaluation\\numFeasible_sol\\Gen' + str(dict_para_PSO['Gen']) + '_NP' + str(dict_para_PSO['NP']) + '_Repeats' + str(dict_para_PSO['repeat-th'] ) + '_BSS' + str(id_BSS) + '_EVs' + str(numEV) + '_BSSs' + str(numBSS) + '_Bats' + str(numBat) + '_' + Alg_name + '_ref' + str(0) + '_encode' + '.csv'
        numFeasible_sol.to_csv(path)

def creat_population(problem):
    dict = problem.name
    OBJ, SOL_1D = [], []
    BSSs = dict['BSSs']
    dict_para_FPC = dict['dict_para_FPC']
    print('=' * 50)
    print('种群初始化开始\n', '...')
    for i in range(dict['NP']):
        sol_1D = np.array(create_sol_1D_int(BSSs[dict['BSS_j']], dict_para_FPC))
        # population[i].Chrom = sol_1D.copy()
        # population[i].Phen = sol_1D.copy()
        _, f1, f2, f3 = cal_all_int(sol_1D , BSSs[dict['BSS_j']], dict_para_FPC)
        tmp = np.array([f1, f2, f3])
        OBJ.append(tmp)
        SOL_1D.append(sol_1D)
    OBJ = np.array(OBJ)
    SOL_1D = np.array(SOL_1D)
    print('种群初始化完成')
    print('=' * 50)
    return SOL_1D, OBJ