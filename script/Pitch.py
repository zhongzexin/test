"""
--coding: utf-8
--WT
--10/22/2021
"""

# Pitch fault in this script includes(105004,105009,105011,105016,105029,105031,105036,105049,105051,105056)

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
#         sql_0 = "select * from (select distinct wtid,real_time,grPitchAngle,grPitchAngle_10min,grPitchSpeedBlade1," \
#                 "grPitchSpeedBlade2,grPitchSpeedBlade3,grPitchAngleBlade1,grPitchAngleBlade2,grPitchAngleBlade3," \
#                 "grTempPitchTransformer_10min,grBlade1TempMotor_1sec,grBlade2TempMotor_1sec,grBlade3TempMotor_1sec," \
#                 "grBlade1TempBattBox_1sec,grBlade2TempBattBox_1sec,grBlade3TempBattBox_1sec,grTempCntr_1sec," \
#                 "grBlade1TempInvBox_1sec,grBlade2TempInvBox_1sec,grBlade3TempInvBox_1sec,grGenPowerForProcess_10min," \
#                 "grWindSpeed_10min,giFaultInformation," \
#                 "cast(DATE_FORMAT( real_time , '%H' ) as signed) as hou," \
#                 "cast(DATE_FORMAT( real_time ,'%i' ) as signed) as min," \
#                 "cast(DATE_FORMAT( real_time , '%Y-%m-%d ' ) as date) as dat from {} " \
#                 "where real_time <= '2020-01-01 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 1) = 0 group by b.min, b.hou, b.dat".format(self.table)
#         sql_1 = "select * from (select distinct wtid,real_time,grPitchAngle,grPitchAngle_10min,grPitchSpeedBlade1," \
#                 "grPitchSpeedBlade2,grPitchSpeedBlade3,grPitchAngleBlade1,grPitchAngleBlade2,grPitchAngleBlade3," \
#                 "grTempPitchTransformer_10min,grBlade1TempMotor_1sec,grBlade2TempMotor_1sec,grBlade3TempMotor_1sec," \
#                 "grBlade1TempBattBox_1sec,grBlade2TempBattBox_1sec,grBlade3TempBattBox_1sec,grTempCntr_1sec," \
#                 "grBlade1TempInvBox_1sec,grBlade2TempInvBox_1sec,grBlade3TempInvBox_1sec,grGenPowerForProcess_10min," \
#                 "grWindSpeed_10min,giFaultInformation," \
#                 "cast(DATE_FORMAT( real_time , '%H' ) as signed) as hou," \
#                 "cast(DATE_FORMAT( real_time ,'%i' ) as signed) as min," \
#                 "cast(DATE_FORMAT( real_time , '%Y-%m-%d ' ) as date) as dat from {} " \
#                 "where real_time >= '2021-01-01 00:00:00' and  real_time <= '2021-08-01 00:00:00' order by real_time) b " \
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
#                "AND HAPPEN_TIME >= '2020-01-01 00:00:00' and HAPPEN_TIME <= '2021-02-01 00:00:00' and fault_code in" \
#                "(105009,105011,105029,105031,105049,105051,103008,103019,103030,103015,103026,103037)" \
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
#         label_data.to_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Pitch_fault_data\\edition3.xlsx")
#         print('\n\n**************标签数据结束****************************')
#         return label_data
#

#
# # step2: put more fault data to do DAE algorithm
def merge_data():
    d1 = pd.read_excel(r"../data/Pitch_MY.xlsx", index_col=0)

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

    # def fault_PitchAngleBlade1(self):  # 105009 pitch angle of blade1
    #     PitchAngleBlade1,results1 = self.threshold(self.score_AngleBlade1,'grPitchAngleBlade1')
    #     PitchAngleBlade1['component'] = '变桨系统'
    #     PitchAngleBlade1['Parameter'] = '桨叶1角度'
    #     PitchAngleBlade1['fault'] = '桨叶1角度异常'
    #     print('\n\n**************桨叶1角度异常---异常诊断结束****************************')
    #     return PitchAngleBlade1,results1
    #
    # def fault_PitchAngleBlade2(self):  # 105029 pitch angle of blade2
    #     PitchAngleBlade2 ,results2= self.threshold(self.score_AngleBlade2,'grPitchAngleBlade2')
    #     PitchAngleBlade2['component'] = '变桨系统'
    #     PitchAngleBlade2['Parameter'] = '桨叶2角度'
    #     PitchAngleBlade2['fault'] = '桨叶2角度异常'
    #     print('\n\n**************桨叶2角度异常---异常诊断结束****************************')
    #     return PitchAngleBlade2,results2
    #
    # def fault_PitchAngleBlade3(self):  # 105049&105051 pitch angle of blade3
    #     PitchAngleBlade3 ,results3= self.threshold(self.score_AngleBlade3,'grPitchAngleBlade3')
    #     PitchAngleBlade3['component'] = '变桨系统'
    #     PitchAngleBlade3['Parameter'] = '桨叶3角度'
    #     PitchAngleBlade3['fault'] = '桨叶3角度异常'
    #     print('\n\n**************桨叶3角度异常---异常诊断结束****************************')
    #     return PitchAngleBlade3,results3

    def fault_Blade1TempMotor(self):  # 105013 Protection fan of motor 1 may happen error
        Blade1TempMotor ,results4= self.threshold('grBlade1TempMotor_1sec')
        Blade1TempMotor['component'] = '变桨系统'
        Blade1TempMotor['Parameter'] = '桨叶1电机风扇温度'
        Blade1TempMotor['fault'] = '桨叶1电机风扇温度异常'
        print('\n\n**************桨叶1电机风扇温度异常---异常诊断结束****************************')
        return Blade1TempMotor,results4

    def fault_Blade2TempMotor(self):  # 105033 Protection fan of motor 2 may happen error
        Blade2TempMotor ,results5= self.threshold('grBlade2TempMotor_1sec')
        Blade2TempMotor['component'] = '变桨系统'
        Blade2TempMotor['Parameter'] = '桨叶2电机风扇温度'
        Blade2TempMotor['fault'] = '桨叶2电机风扇温度异常'
        print('\n\n**************桨叶2电机风扇温度异常---异常诊断结束****************************')
        return Blade2TempMotor,results5

    def fault_Blade3TempMotor(self):  # 105033 Protection fan of motor 3 may happen error
        Blade3TempMotor ,results6= self.threshold('grBlade3TempMotor_1sec')
        Blade3TempMotor['component'] = '变桨系统'
        Blade3TempMotor['Parameter'] = '桨叶3电机风扇温度'
        Blade3TempMotor['fault'] = '桨叶3电机风扇温度异常'
        print('\n\n**************桨叶3电机风扇温度异常---异常诊断结束****************************')
        return Blade3TempMotor,results6

    # def fault_Blade1TempInvBox(self):  # Tempurature of inv box 1 has problem
    #     Blade1TempInvBox ,results7= self.threshold('grBlade1TempInvBox_1sec')
    #     Blade1TempInvBox['component'] = '变桨系统'
    #     Blade1TempInvBox['Parameter'] = '桨叶1轴控箱温度'
    #     Blade1TempInvBox['fault'] = '桨叶1轴控箱温度异常'
    #     print('\n\n**************桨叶1轴控箱温度异常---异常诊断结束****************************')
    #     return Blade1TempInvBox,results7
    #
    # def fault_Blade2TempInvBox(self):  # Tempurature of inv box 2 has problem
    #     Blade2TempInvBox ,results8= self.threshold('grBlade2TempInvBox_1sec')
    #     Blade2TempInvBox['component'] = '变桨系统'
    #     Blade2TempInvBox['Parameter'] = '桨叶2轴控箱温度'
    #     Blade2TempInvBox['fault'] = '桨叶2轴控箱温度异常'
    #     print('\n\n**************桨叶2轴控箱温度异常---异常诊断结束****************************')
    #     return Blade2TempInvBox,results8
    #
    # def fault_Blade3TempInvBox(self):  # Tempurature of inv box 3 has problem
    #     Blade3TempInvBox ,results9= self.threshold('grBlade3TempInvBox_1sec')
    #     Blade3TempInvBox['component'] = '变桨系统'
    #     Blade3TempInvBox['Parameter'] = '桨叶3轴控箱温度'
    #     Blade3TempInvBox['fault'] = '桨叶3轴控箱温度异常'
    #     print('\n\n**************桨叶3轴控箱温度异常---异常诊断结束****************************')
    #     return Blade3TempInvBox,results9

    # def fault_Blade1TempBattBox(self):  # Tempurature of battery box 1 has problem
    #     Blade1TempBattBox ,results10= self.threshold('grBlade1TempBattBox_1sec')
    #     Blade1TempBattBox['component'] = '变桨系统'
    #     Blade1TempBattBox['Parameter'] = '桨叶1电池箱温度'
    #     Blade1TempBattBox['fault'] = '桨叶1电池箱温度异常'
    #     print('\n\n**************桨叶1电池箱温度异常---异常诊断结束****************************')
    #     return Blade1TempBattBox,results10
    #
    # def fault_Blade2TempBattBox(self):  # Tempurature of battery box 2 has problem
    #     Blade2TempBattBox ,results11= self.threshold( 'grBlade2TempBattBox_1sec')
    #     Blade2TempBattBox['component'] = '变桨系统'
    #     Blade2TempBattBox['Parameter'] = '桨叶2电池箱温度'
    #     Blade2TempBattBox['fault'] = '桨叶2电池箱温度异常'
    #     print('\n\n**************桨叶2电池箱温度异常---异常诊断结束****************************')
    #     return Blade2TempBattBox,results11
    #
    # def fault_Blade3TempBattBox(self):  # Tempurature of battery box 3 has problem
    #     Blade3TempBattBox ,results12= self.threshold('grBlade3TempBattBox_1sec')
    #     Blade3TempBattBox['component'] = '变桨系统'
    #     Blade3TempBattBox['Parameter'] = '桨叶3电池箱温度'
    #     Blade3TempBattBox['fault'] = '桨叶3电池箱温度异常'
    #     print('\n\n**************桨叶3电池箱温度异常---异常诊断结束****************************')
    #     return Blade3TempBattBox,results12

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
        #     PitchAngleBlade1,results1 = self.fault_PitchAngleBlade1()
        #     PitchAngleBlade1_0 = self.Reli_day(PitchAngleBlade1, results1)
        # except Exception:
        #     PitchAngleBlade1_0 = pd.DataFrame()
        #     print('\n\n**************桨叶1角度异常诊断错误****************************')
        # try:
        #     PitchAngleBlade2,results2 = self.fault_PitchAngleBlade2()
        #     PitchAngleBlade2_0 = self.Reli_day(PitchAngleBlade2, results2)
        # except Exception:
        #     PitchAngleBlade2_0 = pd.DataFrame()
        #     print('\n\n**************桨叶2角度异常诊断错误****************************')
        # try:
        #     PitchAngleBlade3,results3 = self.fault_PitchAngleBlade3()
        #     PitchAngleBlade3_0 = self.Reli_day(PitchAngleBlade3, results3)
        # except Exception:
        #     PitchAngleBlade3_0 = pd.DataFrame()
        #     print('\n\n**************桨叶3角度异常诊断错误****************************')
        try:
            Blade1TempMotor,results4= self.fault_Blade1TempMotor()
            Blade1TempMotor_0 = self.Reli_day(Blade1TempMotor,results4)
        except Exception:
            Blade1TempMotor_0 = pd.DataFrame()
            print('\n\n**************桨叶1电机风扇温度异常诊断错误****************************')
        try:
            Blade2TempMotor,results5 = self.fault_Blade2TempMotor()
            Blade2TempMotor_0 = self.Reli_day(Blade2TempMotor, results5)
        except Exception:
            Blade2TempMotor_0 = pd.DataFrame()
            print('\n\n**************桨叶2电机风扇温度异常诊断错误****************************')
        try:
            Blade3TempMotor,results6 = self.fault_Blade3TempMotor()
            Blade3TempMotor_0 = self.Reli_day(Blade3TempMotor, results6)
        except Exception:
            Blade3TempMotor_0 = pd.DataFrame()
            print('\n\n**************桨叶3电机风扇温度异常诊断错误****************************')
        # try:
        #     Blade1TempInvBox,results7 = self.fault_Blade1TempInvBox()
        #     Blade1TempInvBox_0 = self.Reli_day(Blade1TempInvBox,results7)
        # except Exception:
        #     Blade1TempInvBox_0 = pd.DataFrame()
        #     print('\n\n**************桨叶1轴控箱温度异常****************************')
        # try:
        #     Blade2TempInvBox,results8 = self.fault_Blade2TempInvBox()
        #     Blade2TempInvBox_0 = self.Reli_day(Blade2TempInvBox,results8)
        # except Exception:
        #     Blade2TempInvBox_0 = pd.DataFrame()
        #     print('\n\n**************桨叶2轴控箱温度异常****************************')
        # try:
        #     Blade3TempInvBox,results9 = self.fault_Blade3TempInvBox()
        #     Blade3TempInvBox_0 = self.Reli_day(Blade3TempInvBox,results9)
        # except Exception:
        #     Blade3TempInvBox_0 = pd.DataFrame()
        #     print('\n\n**************桨叶3轴控箱温度异常****************************')
        # try:
        #     Blade1TempBattBox,results10 = self.fault_Blade1TempBattBox()
        #     Blade1TempBattBox_0 = self.Reli_day( Blade1TempBattBox,results10)
        # except Exception:
        #     Blade1TempBattBox_0 = pd.DataFrame()
        #     print('\n\n**************桨叶1电池箱温度异常****************************')
        # try:
        #     Blade2TempBattBox,results11 = self.fault_Blade2TempBattBox()
        #     Blade2TempBattBox_0 = self.Reli_day( Blade2TempBattBox,results11)
        # except Exception:
        #     Blade2TempBattBox_0 = pd.DataFrame()
        #     print('\n\n**************桨叶2电池箱温度异常****************************')
        # try:
        #     Blade3TempBattBox,results12 = self.fault_Blade3TempBattBox()
        #     Blade3TempBattBox_0 = self.Reli_day( Blade3TempBattBox,results12)
        # except Exception:
        #     Blade3TempBattBox_0 = pd.DataFrame()
        #     print('\n\n**************桨叶3电池箱温度异常****************************')
        data = pd.concat([Blade1TempMotor_0,Blade2TempMotor_0,Blade3TempMotor_0]).reset_index(drop=True)
        data['farm_code'] = data['wtid'].apply(lambda x: int(str(x)[:5]))
        return data

    def Reli_Pitch(self, data):
        data_select = data[['wtid', 'real_time', 'Reliability', 'level']].drop_duplicates()
        mean = data_select.groupby(['wtid', 'real_time'], as_index=False, group_keys=False).Reliability.mean()

        data_level = data_select[['wtid', 'real_time', 'level']]
        select_level = pd.crosstab([data_level['wtid'], data_level['real_time']], data_level['level']).reset_index()

        select_level['bad'] = select_level[list(set(select_level.columns.tolist()) - {'wtid', 'real_time', '良好'})].sum(axis=1).astype(np.int64)
        select_level['level'] = select_level.apply(lambda x: '良好' if x['bad'] == 0 else ('一般' if x['bad'] == 1 else ('较差' if x['bad'] == 2 else '差')), axis=1)
        select_level = select_level[['wtid', 'real_time', 'level']]
        score_day = pd.merge(mean, select_level, on=['wtid', 'real_time'], how='left')
        score_day = score_day.rename({'Reliability': 'pitch_reliability', 'level': 'pitch_level'}, axis=1)
        return score_day






