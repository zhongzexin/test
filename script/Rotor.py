"""
--coding: utf-8
--WT
--10/22/2021
"""

# Rotor fault in this script includes(204002,204033)
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

# Part 1: collect generator fault feature from SCADA
# class data_extract:
#     def __init__(self, table):
#         self.table = table
#
#     # extract 2020 and 2021 year data with interval 10min
#     def run_sql(self):
#         conn_0 = sc.conn_0()
#         conn_1 = sc.conn_1()
#         sql_0 = "select * from (select distinct wtid,real_time,grTempRotorBearA_1sec,grTempRotorBearA_10min," \
#                 "grTempRotorBearB_1sec,grTempRotorBearB_10min,grRotorSpeedPDM,grTempOutdoor_1sec,grTempOutdoor_10min," \
#                 "grGenSpeedForProcess,grTempNacelle_1sec,grTempNacelle_10min,grWindSpeed_10min,giTurbineOperationMode," \
#                 "giFaultInformation,grCAN_ReactivePower,"\
#                 "cast(DATE_FORMAT( real_time , '%H' ) as signed) as hou," \
#                 "cast(DATE_FORMAT( real_time ,'%i' ) as signed) as min," \
#                 "cast(DATE_FORMAT( real_time , '%Y-%m-%d ' ) as date) as dat from {} " \
#                 "where real_time <= '2020-01-01 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 5) = 0 group by b.min, b.hou, b.dat".format(self.table)
#         sql_1 = "select * from (select distinct wtid,real_time,grTempRotorBearA_1sec,grTempRotorBearA_10min," \
#                 "grTempRotorBearB_1sec,grTempRotorBearB_10min,grRotorSpeedPDM,grTempOutdoor_1sec,grTempOutdoor_10min," \
#                 "grGenSpeedForProcess,grTempNacelle_1sec,grTempNacelle_10min,grWindSpeed_10min,giTurbineOperationMode," \
#                 "giFaultInformation,grCAN_ReactivePower,"\
#                 "cast(DATE_FORMAT( real_time , '%H' ) as signed) as hou," \
#                 "cast(DATE_FORMAT( real_time ,'%i' ) as signed) as min," \
#                 "cast(DATE_FORMAT( real_time , '%Y-%m-%d ' ) as date) as dat from {} " \
#                 "where real_time >= '2021-06-01 00:00:00' and real_time <= '2021-07-01 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 5) = 0 group by b.min, b.hou, b.dat".format(self.table)
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
#                "AND HAPPEN_TIME >= '2021-03-01 00:00:00' and HAPPEN_TIME <= '2021-03-01 00:00:00' and fault_code in" \
#                "(204001,204002,204006,204007,384,385,386,204012,204016,204024,204025,204026,204027,204028,204029," \
#                "204030,204031,204032,204033,204034,204035,204036,204037,204101)" \
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
#         label_data.to_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Rotor_fault_data\\tempbearingA\\10006.xlsx")
#         print('\n\n**************标签数据结束****************************')
#         return label_data



# step2: put more fault data to do DAE algorithm
def merge_data():
    d1 = pd.read_excel(r"../data/Rotor_MY.xlsx", index_col=0)

    test = d1.copy()
    print('\n\n**************读取数据完毕****************************')
    multi_data = test.reset_index(drop=True)
    multi_data['FAULT_CODE'] = multi_data['FAULT_CODE'].apply(str)
    fault_data = multi_data.loc[multi_data['label'] == 1]
    # print(f'data_shape = {multi_data.shape}\nNP_ratio = {Counter(multi_data["label"])}')
    fea_data = (multi_data.drop(columns=['wtid', 'FAULT_CODE'])).reset_index(drop=True)   # show more format of priori data
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

    # def fault_RotorSpeed(self):  # 204001&204002&204017 speed of rotor may have problem
    #     RotorSpeed, results1 = self.threshold('grRotorSpeedPDM')
    #     RotorSpeed['component'] = '主轴'
    #     RotorSpeed['Parameter'] = '主轴速度'
    #     RotorSpeed['fault'] = '主轴超速'
    #     print('\n\n**************主轴超速---异常诊断结束****************************')
    #     return RotorSpeed, results1

    # def fault_TempBearA(self):  # 204026&204027&204028&204029 Tempurature of rotor bearing A may have problem
    #     TempBearA , results2= self.threshold('grTempRotorBearA_1sec')
    #     TempBearA['component'] = '主轴'
    #     TempBearA['Parameter'] = '主轴轴承A温度'
    #     TempBearA['fault'] = '主轴轴承A温度异常'
    #     print('\n\n**************主轴轴承A温度异常---异常诊断结束****************************')
    #     return TempBearA, results2

    def fault_TempBearB(self):  # 204026&204027&204028&204029 Tempurature of rotor bearing B may have problem
        TempBearB , results3= self.threshold('grTempRotorBearB_1sec')
        TempBearB['component'] = '主轴'
        TempBearB['Parameter'] = '主轴轴承B温度'
        TempBearB['fault'] = '主轴轴承B温度异常'
        print('\n\n**************主轴轴承B温度异常---异常诊断结束****************************')
        return TempBearB, results3

    def Reli_day(self,data,result):
        data['Threshold'] = pd.DataFrame(result['lower_thresholds'])
        data['real_time'] = data.real_time.apply(lambda x: x.strftime('%Y-%m-%d'))
        data0 = data.drop(columns=['Relibility', 'Threshold']).drop_duplicates(['wtid','real_time', 'level'])
        data0['priority'] = np.where(data0['level'] == '差', 4,
                                     np.where(data0['level'] == '较差', 3, np.where(data0['level'] == '一般', 2, 1)))
        data1 = data0.groupby(['wtid', 'real_time'], as_index=False, group_keys=False).apply(
            lambda x: x.sort_values(['priority'], ascending=False).head(1)).drop(columns=['priority'])
        data1['Reliability'] = data['Relibility']
        data1['Threshold'] = data['Threshold']
        data1 =  data1.reset_index(drop=True)
        return data1

    def run(self):
        # try:
        #     RotorSpeed,results1 = self.fault_RotorSpeed()
        #     RotorSpeed0 = self.Reli_day(RotorSpeed,results1)
        # except Exception:
        #     RotorSpeed0 =pd.DataFrame()
        #     print('\n\n**************主轴超速异常诊断错误****************************')
        # try:
        #     TempBearA,results2 = self.fault_TempBearA()
        #     TempBearA0 = self.Reli_day(TempBearA,results2)
        # except Exception:
        #     TempBearA0 = pd.DataFrame()
        #     print('\n\n**************主轴轴承A温度异常诊断错误****************************')
        try:
            TempBearB,results3 = self.fault_TempBearB()
            TempBearB0 = self.Reli_day(TempBearB,results3)
        except Exception:
            TempBearB0 = pd.DataFrame()
            print('\n\n**************主轴轴承B温度异常诊断错误****************************')
        # data = pd.concat([RotorSpeed0,  TempBearB0]).reset_index(drop=True)
        data = TempBearB0.reset_index(drop=True)
        data['farm_code'] = data['wtid'].apply(lambda x: int(str(x)[:5]))
        return data

    def Reli_Rotor(self, data):
        data_select = data[['wtid', 'real_time', 'Reliability', 'level']].drop_duplicates()
        mean = data_select.groupby(['wtid', 'real_time'], as_index=False, group_keys=False).Reliability.mean()

        data_level = data_select[['wtid', 'real_time', 'level']]
        select_level = pd.crosstab([data_level['wtid'], data_level['real_time']], data_level['level']).reset_index()

        select_level['bad'] = select_level[list(set(select_level.columns.tolist()) - {'wtid', 'real_time', '良好'})].sum(axis=1).astype(np.int64)
        select_level['level'] = select_level.apply(lambda x: '良好' if x['bad'] == 0 else ('一般' if x['bad'] == 1 else ('较差' if x['bad'] == 2 else '差')), axis=1)
        select_level = select_level[['wtid', 'real_time', 'level']]
        score_day = pd.merge(mean, select_level, on=['wtid', 'real_time'], how='left')
        score_day = score_day.rename({'Reliability': 'rotor_reliability', 'level': 'rotor_level'}, axis=1)
        return score_day











