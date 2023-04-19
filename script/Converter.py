"""
--coding: utf-8
--WT
--10/22/2021
"""
# Converter fault in this script includes(305003，305027，3050111，3050132)

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


# # Part 1: collect generator fault feature from SCADA
# class data_extract:
#     def __init__(self, table):
#         self.table = table
#
#     # extract 2020 and 2021 year data with interval 10min
#     def run_sql(self):
#         conn_0 = sc.conn_0()
#         conn_1 = sc.conn_1()
#         sql_0 = "select * from (select distinct wtid,real_time,grGenSpeedForProcess,grGenSpeedForProcess_10min," \
#                 "giTurbineOperationMode,grA_CAN_GridCurrent," \
#                 "grB_CAN_GridCurrent,grC_CAN_GridCurrent,grGridFrequencyForProcess," \
#                 "grUL1_690V_KL3403,grIL1_690V_KL3403,grUL2_690V_KL3403,grIL2_690V_KL3403,grUL3_690V_KL3403," \
#                 "grIL3_690V_KL3403,giFaultInformation,grCAN_ReactivePower,grWindSpeed_10min," \
#                 "cast(DATE_FORMAT( real_time , '%H' ) as signed) as hou," \
#                 "cast(DATE_FORMAT( real_time ,'%i' ) as signed) as min," \
#                 "cast(DATE_FORMAT( real_time , '%Y-%m-%d ' ) as date) as dat from {} " \
#                 "where real_time >= '2020-12-31 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 10) = 0 group by b.min, b.hou, b.dat".format(self.table)
#         sql_1 = "select * from (select distinct wtid,real_time,grGenSpeedForProcess,grGenSpeedForProcess_10min," \
#                 "giTurbineOperationMode,grA_CAN_GridCurrent," \
#                 "grB_CAN_GridCurrent,grC_CAN_GridCurrent,grGridFrequencyForProcess," \
#                 "grUL1_690V_KL3403,grIL1_690V_KL3403,grUL2_690V_KL3403,grIL2_690V_KL3403,grUL3_690V_KL3403," \
#                 "grIL3_690V_KL3403,giFaultInformation,grCAN_ReactivePower,grWindSpeed_10min," \
#                 "cast(DATE_FORMAT( real_time , '%H' ) as signed) as hou," \
#                 "cast(DATE_FORMAT( real_time ,'%i' ) as signed) as min," \
#                 "cast(DATE_FORMAT( real_time , '%Y-%m-%d ' ) as date) as dat from {} " \
#                 "where real_time >= '2021-01-01 00:00:00' and real_time <= '2021-10-01 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 3) = 0 group by b.min, b.hou, b.dat".format(self.table)
#         data_0 = pd.read_sql(sql=sql_0, con=conn_0)
#         data_1 = pd.read_sql(sql=sql_1, con=conn_1)
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
#                "AND HAPPEN_TIME >= '2021-03-07 00:00:00' and HAPPEN_TIME <= '2021-03-10 00:00:00' and fault_code in" \
#                "(305001,305003,305006,305007,305027,305083,305111,305132,305134,305135,305155)" \
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
#         label_data.to_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Converter_fault_data\\edition1.xlsx")
#         print('\n\n**************标签数据结束****************************')
#         return label_data
#
#


# step2: put more fault data to do DAE algorithm
def merge_data():
    d1 = pd.read_excel(r"../data/Converter_MY.xlsx", index_col=0)

    test = d1.copy()
    print('\n\n**************读取数据完毕****************************')
    multi_data = test.reset_index(drop=True)
    multi_data['FAULT_CODE'] = multi_data['FAULT_CODE'].apply(str)
    fault_data = multi_data.loc[multi_data['label'] == 1]
    # print(f'data_shape = {multi_data.shape}\nNP_ratio = {Counter(multi_data["label"])}')
    fea_data = (multi_data.drop(columns=['wtid', 'FAULT_CODE'])).reset_index(drop=True)
    return fea_data, multi_data, fault_data

# step3: set anomaly rules as many types as possible
class anomaly:
    def __init__(self, multi_data, length):
        self.multi_data = multi_data
        self.length = length

    def threshold(self, variable):
        clean_var = self.multi_data[['real_time', variable, 'label']]
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
        # plt.ylabel('Relibility' + '_' + variable)

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
        return final,results

    # def fault_AGridCurrent(self):  # AgridCurrent may happen error
    #     AGridCurrent,results1 = self.threshold(self.score_AGridCurrent,'grA_CAN_GridCurrent')
    #     AGridCurrent['component'] = '变流器'
    #     AGridCurrent['Parameter'] = '网侧A相电流'
    #     AGridCurrent['fault'] = '网侧A相电流异常'
    #     print('\n\n**************网侧A相电流异常---异常诊断结束****************************')
    #     return AGridCurrent,results1
    #
    # def fault_BGridCurrent(self):  # BgridCurrent may happen error
    #     BGridCurrent,results2 = self.threshold(self.score_AGridCurrent,'grB_CAN_GridCurrent')
    #     BGridCurrent['component'] = '变流器'
    #     BGridCurrent['Parameter'] = '网侧B相电流'
    #     BGridCurrent['fault'] = '网侧B相电流异常'
    #     print('\n\n**************网侧B相电流异常---异常诊断结束****************************')
    #     return BGridCurrent,results2
    #
    # def fault_CGridCurrent(self):  # CgridCurrent may happen error
    #     CGridCurrent,results3 = self.threshold(self.score_AGridCurrent,'grC_CAN_GridCurrent')
    #     CGridCurrent['component'] = '变流器'
    #     CGridCurrent['Parameter'] = '网侧C相电流'
    #     CGridCurrent['fault'] = '网侧C相电流异常'
    #     print('\n\n**************网侧C相电流异常---异常诊断结束****************************')
    #     return CGridCurrent,results3

    def fault_UL1Voltage(self):  # Voltage of L1 may happen error
        UL1Voltage ,results4= self.threshold('grUL1_690V_KL3403')
        UL1Voltage['component'] = '变流器'
        UL1Voltage['Parameter'] = '电网L1电压'
        UL1Voltage['fault'] = '电网L1电压异常'
        print('\n\n**************电网L1电压异常---异常诊断结束****************************')
        return UL1Voltage,results4

    def fault_UL2Voltage(self):  # Voltage of L2 may happen error
        UL2Voltage,results5 = self.threshold('grUL2_690V_KL3403')
        UL2Voltage['component'] = '变流器'
        UL2Voltage['Parameter'] = '电网L2电压'
        UL2Voltage['fault'] = '电网L2电压异常'
        print('\n\n**************电网L2电压异常---异常诊断结束****************************')
        return UL2Voltage,results5

    def fault_UL3Voltage(self):  # Voltage of L3 may happen error
        UL3Voltage ,results6= self.threshold('grUL3_690V_KL3403')
        UL3Voltage['component'] = '变流器'
        UL3Voltage['Parameter'] = '电网L3电压'
        UL3Voltage['fault'] = '电网L3电压异常'
        print('\n\n**************电网L3电压异常---异常诊断结束****************************')
        return UL3Voltage,results6

    def fault_IL1Current(self):  # Electric current of L1 may happen error
        IL1Current,results7 = self.threshold('grIL1_690V_KL3403')
        IL1Current['component'] = '变流器'
        IL1Current['Parameter'] = '电网L1电流'
        IL1Current['fault'] = '电网L1电流异常'
        print('\n\n**************电网L1电流异常---异常诊断结束****************************')
        return IL1Current,results7

    def fault_IL2Current(self):  # Electric current of L2 may happen error
        IL2Current,results8 = self.threshold('grIL2_690V_KL3403')
        IL2Current['component'] = '变流器'
        IL2Current['Parameter'] = '电网L2电流'
        IL2Current['fault'] = '电网L2电流异常'
        print('\n\n**************电网L2电流异常---异常诊断结束****************************')
        return IL2Current,results8

    def fault_IL3Current(self):  # Electric current of L3 may happen error
        IL3Current,results9 = self.threshold('grIL3_690V_KL3403')
        IL3Current['component'] = '变流器'
        IL3Current['Parameter'] = '电网L3电流'
        IL3Current['fault'] = '电网L3电流异常'
        print('\n\n**************电网L3电流异常---异常诊断结束****************************')
        return IL3Current,results9

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
        # try:
        #     AGridCurrent,results1= self.fault_AGridCurrent()
        #     AGridCurrent0 = self.Reli_day( AGridCurrent,results1)
        # except Exception:
        #     AGridCurrent0 = pd.DataFrame()
        #     print('\n\n**************网侧A相电流异常诊断错误****************************')
        # try:
        #     BGridCurrent,results2 = self.fault_BGridCurrent()
        #     BGridCurrent0 = self.Reli_day( BGridCurrent,results2)
        # except Exception:
        #     BGridCurrent0 = pd.DataFrame()
        #     print('\n\n**************网侧B相电流异常诊断错误****************************')
        # try:
        #     CGridCurrent,results3= self.fault_CGridCurrent()
        #     CGridCurrent0 = self.Reli_day(CGridCurrent,results3)
        # except Exception:
        #     CGridCurrent0 = pd.DataFrame()
        #     print('\n\n**************网侧C相电流异常诊断错误****************************')
        try:
            UL1Voltage,results4 = self.fault_UL1Voltage()
            UL1Voltage0 = self.Reli_day( UL1Voltage,results4)
        except Exception:
            UL1Voltage0= pd.DataFrame()
            print('\n\n**************电网L1电压异常诊断错误****************************')
        try:
            UL2Voltage,results5 = self.fault_UL2Voltage()
            UL2Voltage0 = self.Reli_day(UL2Voltage,results5)
        except Exception:
            UL2Voltage0 = pd.DataFrame()
            print('\n\n**************电网L2电压异常诊断错误****************************')
        try:
            UL3Voltage,results6 = self.fault_UL3Voltage()
            UL3Voltage0 = self.Reli_day(UL3Voltage,results6)
        except Exception:
            UL3Voltage0 = pd.DataFrame()
            print('\n\n**************电网L3电压异常诊断错误****************************')
        try:
            IL1Current ,results7= self.fault_IL1Current()
            IL1Current0 = self.Reli_day(IL1Current ,results7)
        except Exception:
            IL1Current0 = pd.DataFrame()
            print('\n\n**************电网L1电流异常诊断错误****************************')
        try:
            IL2Current ,results8= self.fault_IL2Current()
            IL2Current0 = self.Reli_day(IL2Current ,results8)
        except Exception:
            IL2Current0 = pd.DataFrame()
            print('\n\n**************电网L2电流异常诊断错误****************************')
        try:
            IL3Current,results9 = self.fault_IL3Current()
            IL3Current0 = self.Reli_day(IL3Current,results9)
        except Exception:
            IL3Current0 = pd.DataFrame()
            print('\n\n**************电网L3电流异常诊断错误****************************')
        data = pd.concat([UL1Voltage0,UL2Voltage0,UL3Voltage0,IL1Current0,IL2Current0,IL3Current0]).reset_index(drop=True)
        data['farm_code'] = data['wtid'].apply(lambda x: int(str(x)[:5]))
        return  data

    def Reli_Converter(self, data):
        data_select = data[['wtid', 'real_time', 'Reliability', 'level']].drop_duplicates()
        mean = data_select.groupby(['wtid', 'real_time'], as_index=False, group_keys=False).Reliability.mean()

        data_level = data_select[['wtid', 'real_time', 'level']]
        select_level = pd.crosstab([data_level['wtid'], data_level['real_time']], data_level['level']).reset_index()

        select_level['bad'] = select_level[list(set(select_level.columns.tolist()) - {'wtid', 'real_time', '良好'})].sum(axis=1).astype(np.int64)
        select_level['level'] = select_level.apply(lambda x: '良好' if x['bad'] == 0 else ('一般' if x['bad'] == 1 else ('较差' if x['bad'] == 2 else '差')), axis=1)
        select_level = select_level[['wtid', 'real_time', 'level']]
        score_day = pd.merge(mean, select_level, on=['wtid', 'real_time'], how='left')
        score_day = score_day.rename({'Reliability': 'converter_reliability', 'level': 'converter_level'}, axis=1)
        return score_day








