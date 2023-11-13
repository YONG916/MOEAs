"""
Date: Created on 2023/11/12
Name: MOEAs for BSS Charging Schedule Problem
Author: Yong Su
"""
import models1
import for_CLass1
import importlib
importlib.reload(models1)
importlib.reload(for_CLass1)
from models1 import *

#%% md
# S0.1.运行哪种算法
GA = 1


# S0.2.是否进行预选择
FCFS_preSelection = 0
ACS_preSelection = 0

T_max = int(fcnTimeToMin('20:00:00')) # 最大运行时间

# S0.4.数据集的选择[-15:]
caseEV = 1
caseBSS = 1

# S0.5.算法的运行次数
num_repeats = 15

# S0.6.数据导入
dicPar = {'T_max': T_max, 'numCharger': 3, 'star_Time': '8:00'}  # dict
EVs, BSSs, df_EV, df_BSS = load_dataset_sol1(caseEV, caseBSS, dicPar)

# S0.7.预处理
dict_para_pre = { 'EV_Speed': 50, 'T_max': T_max, 'selection_par':3, 'K':3}
pre_id = pre_selection(EVs, BSSs, dict_para_pre)


# 获得当前时间时间戳
now = int(time.time())
# 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
timeArray = time.localtime(now)
other_day = time.strftime("%Y-%m-%d", timeArray)
other_hour = time.strftime("%H", timeArray)
# other_min = time.strftime("%M", timeArray) + other_min + 'm_'
time_stamp = other_day + '_' +  other_hour + 'h_'

# S0.8.Initialization
dict_para = {'EV_Capacity': 50,
             'EV_Speed': 50,
             'EV_Consume': 0.25,
            'T_max': T_max,
             'time_slot' : 5,
             'Print_pic':Print_pic,
             'Print_csv':Print_csv,
             'Print_solution':1,
             'FCFS_preSelection': 0,
             'ACS_preSelection':0,
             'pre_id':[],
             'num_chargers':0 ,
             'FCFS':1,
             'max_power_load': 120,
              'normal_power': 180,
             'charge_Power': 75,
             'EP': [0.1,0.1,0.1,
                    0.2, 0.2,0.2,
                    0.4, 0.4, 0.4,0.4,
                    0.2, 0.2, 0.2,

                    0.1, 0.1,0.1,
                    0.2,0.2,0.2,
                    0.4,0.4,0.4, 0.4,
                    0.2,0.2,0.2,

                    0.1, 0.1, 0.1,
                    0.2, 0.2, 0.2,
                    0.4, 0.4, 0.4, 0.4,
                    0.2, 0.2, 0.2,
                    0.1,0.1, 0.1],
             'sol_FCFS':[],
             'w':[1, 1, 1],
             'NP':100,
             'Gen':1000, # 500最好
             'numTime':0,
             'Run time':[1, 1, 1, 1, 1, 1, 1],
             'Elite_choice':1,
             'Elite_choice_Percentage':0.5,
             'BSSs':BSSs,
             'EVs':EVs,
             'num_repeats':num_repeats,
             'show_class':1,
             'Gante_order':1,
             'time_stamp': time_stamp,
             'Chinese_version':1
             }

res_save = []
# S1.1.NIR and GA
resNIR_save = Run_NIR(EVs, BSSs, dict_para)
res_save.append(resNIR_save)
sol_NIR = resNIR_save['save_sol'][0]
# Pic_Gante_Battery(res_save)

if GA:
    resGA_save = Run_GA(EVs, BSSs, dict_para_GA)
    res_save.append(resGA_save)
else:
    resGA_save = Run_saved_result('GA', dict_para)
    res_save.append(resGA_save)

out_data_analysis_sol1(res_save)




""""===============================================2. 多目标优化（帕累托前沿）离散的Solution=============================================================================================================="""
# 1. 选择算法
NSGA2 = 0
NSGA3 = 0
RVEA = 0
MOPSO = 0
# 是否读入近似的Pareto Front
dict_para['Read_PF'] = 0
'==========================Create Pareto Front============================================='
# alg_name = ['NSGA2', 'NSGA3', 'RVEA', 'MOPSO']
# Create_PF(alg_name, dict_pareto_PSO)
'=========================Should be deleted========================================'
# 输入参数
dict_pareto_PSO['Gen'], dict_pareto_PSO['NP'] = 300, 1000
problem.name['Gen'], problem.name['NP'] = 300, 1000
dict_para['dict_pareto'] = problem.name
# 开始读入近似Pareto Front
dict_para['Read_PF'] = 1

if MOPSO:
    # Basic MOPSO: int Solution
    Run_MOPSO_eachBSS_int(dict_pareto_PSO)

if NSGA2:
    Run_NSGA2_eachBSS_int(problem)

if RVEA:
    Run_RVEA_eachBSS_int(problem)

if NSGA3:
    Run_NSGA3_eachBSS_int(problem)










