"""
--coding: utf-8
--WT
--10/22/2021
"""

# Gearbox fault in this script includes(205017,205018,205027,205030,205031,205032,205035,205036,205037)
# 205032,205035,205036,205037）

import pandas as pd
import numpy as np
# import sql_config as sc
import math
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from collections import Counter
# from xgboost import XGBClassifier
from Relibility_Prediction import DAE_NN
from drif_spot import ESPOT


# Part 1: collect Gearbox fault feature from SCADA
# class data_extract:
#     def __init__(self, table):
#         self.table = table
#
#     # extract 2020 and 2021 year data with interval 10min
#     def run_sql(self):
#         conn_0 = sc.conn_0()
#         conn_1 = sc.conn_1()
#         sql_0 = "select * from (select distinct wtid,real_time,grTemp1GearOil_1sec,grTemp1GearOil_10min," \
#                 "grTempGearBearDE_1sec,grTempGearBearDE_10min,grTempGearBearNDE_1sec,grTempGearBearNDE_10min," \
#                 "grTempGearOilHeaterSystem_10min,grTempWaterGearCooling_10min,grGearWaterPumpRunningHours," \
#                 "grGearFanRunningHours,grGearOilPumpHSRunningHours,grGearOilPumpLSRunningHours,giFaultInformation," \
#                 "giTurbineOperationMode,grRotorSpeedPDM,grCAN_ReactivePower,grWindSpeed_10min,grGenPowerForProcess_10min," \
#                 "cast(DATE_FORMAT( real_time , '%H' ) as signed) as hou," \
#                 "cast(DATE_FORMAT( real_time ,'%i' ) as signed) as min," \
#                 "cast(DATE_FORMAT( real_time , '%Y-%m-%d ' ) as date) as dat from {} " \
#                 "where real_time >= '2020-12-15 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 10) = 0 group by b.min, b.hou, b.dat".format(self.table)
#         sql_1 = "select * from (select distinct wtid,real_time,grTemp1GearOil_1sec,grTemp1GearOil_10min," \
#                 "grTempGearBearDE_1sec,grTempGearBearDE_10min,grTempGearBearNDE_1sec,grTempGearBearNDE_10min," \
#                 "grTempGearOilHeaterSystem_10min,grTempWaterGearCooling_10min,grGearWaterPumpRunningHours," \
#                 "grGearFanRunningHours,grGearOilPumpHSRunningHours,grGearOilPumpLSRunningHours,giFaultInformation," \
#                 "giTurbineOperationMode,grRotorSpeedPDM,grCAN_ReactivePower,grWindSpeed_10min,grGenPowerForProcess_10min," \
#                 "cast(DATE_FORMAT( real_time , '%H' ) as signed) as hou," \
#                 "cast(DATE_FORMAT( real_time ,'%i' ) as signed) as min," \
#                 "cast(DATE_FORMAT( real_time , '%Y-%m-%d ' ) as date) as dat from {} " \
#                 "where real_time <= '2021-01-05 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 10) = 0 group by b.min, b.hou, b.dat".format(self.table)
#         data_0 = pd.read_sql(sql=sql_0, con=conn_0)   # 取2020年SCADA数据
#         data_1 = pd.read_sql(sql=sql_1, con=conn_1)   # 取2021年SCADA数据
#         return data_0, data_1
#
#     # combine 2020 year data with 2021 year data
#     def get_data(self):
#         subdata_0, subdata_1 = self.run_sql()
#         data = pd.concat([subdata_0, subdata_1])
#         data = data.drop(columns=['hou', 'min', 'dat'])
#         data = data.dropna().reset_index(drop=True)
#         return data
#
#     # get fault data
#     def fault_data(self):
#         fconn = sc.fault_conn()
#         str = self.table[1:6]
#         sql1 = "select WTGS_CODE,FAULT_CODE,HAPPEN_TIME,END_TIME from iot_wind.tb_wind_fault where FARM_CODE = {} " \
#                "AND HAPPEN_TIME >= '2020-12-30 00:00:00' and HAPPEN_TIME <= '2021-12-31 00:00:00' and fault_code in" \
#                "(205001,205002,205004,205005,205006,205007,205009,205010,205015,205016,205017,205018,205027,205028," \
#                "205029,205030,205031,205032,205033,205034,205035,205036,205037,205039,205040,205041,205042,205043," \
#                "205044,205045,205046,205047,205048,205049,205050,205051,205100)" \
#                " order by WTGS_CODE, HAPPEN_TIME".format(str)
#         fault = pd.read_sql(sql1, con=fconn)
#         fconn.close()
#         return fault
#
#     # label data and give fault code, 0:normal, 1:fault.
#     def label(self):
#         merge = pd.merge(self.get_data(), self.fault_data(), how='left', left_on='wtid', right_on='WTGS_CODE')
#         merge_pur = merge.loc[(merge['real_time'] > merge['HAPPEN_TIME']) & (merge['real_time'] < merge['END_TIME'])]
#         merge_code = merge_pur[['wtid', 'real_time', 'FAULT_CODE']]
#
#         label_data = pd.merge(self.get_data(), merge_code, on=['wtid', 'real_time'], how='left')
#         label_data['FAULT_CODE'] = label_data['FAULT_CODE'].apply(float)
#         label_data['label'] = label_data.FAULT_CODE.apply(lambda x: 0 if math.isnan(x) == True else 1)
#         # label_data.to_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Gearbox_fault_data\\205018\\turbine73-8(Dec).xlsx") #这一行是存excel
#         print('\n\n**************标签数据结束****************************')
#         return label_data
#


# step2: put more fault data to do DAE algorithm
def merge_data():
    d1 = pd.read_excel(r"../data/Gearbox_MY.xlsx", index_col=0)
    test = d1.copy()
    print('\n\n**************读取数据完毕****************************')
    multi_data = test.reset_index(drop=True)
    multi_data['FAULT_CODE'] = multi_data['FAULT_CODE'].apply(str)
    fault_data = multi_data.loc[multi_data['label'] == 1]
    # print(f'data_shape = {multi_data.shape}\nNP_ratio = {Counter(multi_data["label"])}')
    fea_data = (multi_data.drop(columns=['wtid', 'FAULT_CODE'])).reset_index(drop=True)
    return fea_data, multi_data, fault_data

# step3: set anomaly rules as many types as possible
#进行自适应阈值计算
class anomaly:
    def __init__(self, multi_data, length):
        self.multi_data = multi_data
        self.length = length

    def threshold(self, variable):
        clean_var = self.multi_data[['real_time',variable,'label']]
        Encoder = DAE_NN(clean_var)
        score, score_var = Encoder.DAE_model()

        Reli_var = pd.DataFrame()
        Reli_var['Relibility'] = score_var[variable]
        se = Reli_var['Relibility'].values
        lie = Reli_var['Relibility'].tail(self.length).values

        final = Reli_var.tail(self.length).reset_index(drop=True)
        final['predict'] = 0

        q = 1e-2  # risk parameter
        d = 6  # depth
        s = ESPOT(q, d)
        s.fit(se, lie)  # data import
        s.initialize()  # initialization step
        results = s.run()  # run
        del results['upper_thresholds']
        # plt.figure(figsize=(10, 6))
        # s.plot(results)
        # plt.xlabel('datetime_index')
        # plt.ylabel('Relibility'+ '_'+ variable)

        final.loc[results['alarms'], 'predict'] = 1

        final = pd.concat([(self.multi_data[['wtid', 'real_time']].tail(self.length)).reset_index(drop=True), final],
                          axis=1)
        final['count'] = final.groupby(['wtid'], as_index=False, group_keys=False).predict.cumsum()
        def condition(data):
            if data['count'] == 0:
                return '良好'
            elif data['count'] <= 15 and data['predict'] == 1:
                return '一般'
            elif data['count'] <= 20 and data['predict'] == 1:
                return '较差'
            elif data['count'] > 25 and data['predict'] == 1:
                return '差'
            else:
                return '良好'

        final['level'] = final.apply(condition, axis=1)
        final = final.drop(['predict', 'count'], axis=1)
        return final, results

    def fault_Tempoil(self):  # 205017&205018  Tempurature of Gearboxoil may too high or too low
        tempoil, results1 = self.threshold('grTemp1GearOil_1sec')
        tempoil['component'] = '齿轮箱'
        tempoil['Parameter'] = '齿轮箱油温'
        tempoil['fault'] = '机舱齿轮箱油温异常'
        print('\n\n**************机舱齿轮箱油温异常---异常诊断结束****************************')
        return tempoil,results1

    def fault_TempBearingDE(self):  # 205030&205031  Tempurature of bearingDE may too high or too low
        TempBearingDE, results2 = self.threshold('grTempGearBearDE_1sec')
        TempBearingDE['component'] = '齿轮箱'
        TempBearingDE['Parameter'] = '齿轮箱驱动端轴承温度'
        TempBearingDE['fault'] = '齿轮箱驱动端轴承温度异常'
        print('\n\n**************齿轮箱驱动端轴承温度异常---异常诊断结束****************************')
        return TempBearingDE,results2

    def fault_TempBearingNDE(self):  # 205030&205031  Tempurature of bearingDE may too high or too low
        TempBearingNDE,results3 = self.threshold('grTempGearBearNDE_1sec')
        TempBearingNDE['component'] = '齿轮箱'
        TempBearingNDE['Parameter'] = '齿轮箱非驱动端轴承温度'
        TempBearingNDE['fault'] = '齿轮箱非驱动端轴承温度异常'
        print('\n\n**************齿轮箱非驱动端轴承温度异常---异常诊断结束****************************')
        return TempBearingNDE, results3

    def Reli_day(self, data, result):
        data['Threshold'] = pd.DataFrame(result['lower_thresholds'])
        data['real_time'] = data.real_time.apply(lambda x: x.strftime('%Y-%m-%d'))
        data0 = data.drop(columns=['Relibility', 'Threshold']).drop_duplicates(['wtid','real_time', 'level'])
        data0['priority'] = np.where(data0['level'] == '差', 4,
                                     np.where(data0['level'] == '较差', 3, np.where(data0['level'] == '一般', 2, 1)))
        data1 = data0.groupby(['wtid', 'real_time'], as_index=False, group_keys=False).apply(
            lambda x: x.sort_values(['priority'], ascending=False).head(1)).drop(columns=['priority'])
        data1['Reliability'] = data['Relibility']
        data1['Threshold'] = data['Threshold']
        data1 = data1.reset_index(drop=True)
        return data1

    def run(self):
        try:
            tempoil, results1 = self.fault_Tempoil()
            tempoil0 = self.Reli_day( tempoil, results1)
        except Exception:
            tempoil0 = pd.DataFrame()
            print('\n\n**************机舱齿轮箱油温异常诊断错误****************************')
        try:
            TempBearingDE, results2 = self.fault_TempBearingDE()
            TempBearingDE0 = self.Reli_day( TempBearingDE, results2)
        except Exception:
            TempBearingDE0 = pd.DataFrame()
            print('\n\n**************齿轮箱驱动端轴承温度异常诊断错误****************************')
        try:
            TempBearingNDE, results3 = self.fault_TempBearingNDE()
            TempBearingNDE0 = self.Reli_day(TempBearingNDE, results3)
        except Exception:
            TempBearingNDE0 = pd.DataFrame()
            print('\n\n**************齿轮箱非驱动端轴承温度异常诊断错误****************************')
        # try:
        #     TempOilHeater = self.fault_TempOilHeater()
        # except Exception:
        #     print('\n\n**************齿轮箱油加热油温温度异常诊断错误****************************')
        data = pd.concat([tempoil0,TempBearingDE0,TempBearingNDE0]).reset_index(drop=True)
        data['farm_code'] = data['wtid'].apply(lambda x: int(str(x)[:5]))
        return data

    def Reli_Gearbox(self, data):
        data_select = data[['wtid', 'real_time', 'Reliability', 'level']].drop_duplicates()
        mean = data_select.groupby(['wtid', 'real_time'], as_index=False, group_keys=False).Reliability.mean()

        data_level = data_select[['wtid', 'real_time', 'level']]
        select_level = pd.crosstab([data_level['wtid'], data_level['real_time']], data_level['level']).reset_index()

        select_level['bad'] = select_level[list(set(select_level.columns.tolist()) - {'wtid', 'real_time', '良好'})].sum(axis=1).astype(np.int64)
        select_level['level'] = select_level.apply(lambda x: '良好' if x['bad'] == 0 else ('一般' if x['bad'] == 1 else ('较差' if x['bad'] == 2 else '差')), axis=1)
        select_level = select_level[['wtid', 'real_time', 'level']]
        score_day = pd.merge(mean, select_level, on=['wtid', 'real_time'], how='left')
        score_day = score_day.rename({'Reliability': 'gearbox_reliability', 'level': 'gearbox_level'}, axis=1)
        return score_day















