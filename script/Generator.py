"""
--coding: utf-8
--WT
--10/22/2021
"""

# Generator fault in this script includes(205017,205018,205027,205030,205031,205032,205035,205036,205037)

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


# # Part 1: collect component fault feature from SCADA
# class data_extract:
#     def __init__(self, table):
#         self.table = table
#
#     # extract 2020 and 2021 year data with interval 10min
#     def run_sql(self):
#         conn_0 = sc.conn_0()
#         conn_1 = sc.conn_1()
#         sql_0 = "select * from (select distinct wtid,real_time, grGenPowerForProcess,grGenPowerForProcess_10min," \
#                 "grGenSpeedForProcess,grGenSpeedForProcess_10min,grTempGenBearDE_1sec,grTempGenBearDE_10min," \
#                 "grTempGenBearNDE_1sec,grTempGenBearNDE_10min, grTempGenStatorU_1sec,grTempGenStatorU_10min, " \
#                 "grTempGenStatorV_1sec,grTempGenStatorV_10min,grTempGenStatorW_1sec, grTempGenStatorW_10min," \
#                 "grTempGenCoolingAir_1sec,grTempGenCoolingAir_10min, grTempGenOutWind_1sec, grTempGenOutWind_10min, " \
#                 "giTurbineOperationMode,grPitchAngleBlade1,grPitchAngleBlade2,grPitchAngleBlade3,grPitchAngle," \
#                 "grPitchAngle_10min,grNacellePositionTotal,grBlade1TempBattBox_10min,grBlade2TempBattBox_10min," \
#                 "grBlade3TempBattBox_10min,grIL3_690V_KL3403,grTempRotorBearA_10min,grTempRotorBearB_10min," \
#                 "giFaultInformation,grBlade1TempMotor_10min,grBlade2TempMotor_10min,grBlade3TempMotor_10min," \
#                 "grTempOutdoor_1sec,grTempNacelle_1sec,grUL1_690V_KL3403,grUL2_690V_KL3403,grUL3_690V_KL3403," \
#                 "grCAN_ReactivePower,grRotorSpeedPDM,cast(DATE_FORMAT( real_time , '%H' ) as signed) as hou," \
#                 "cast(DATE_FORMAT( real_time ,'%i' ) as signed) as min," \
#                 "cast(DATE_FORMAT( real_time , '%Y-%m-%d ' ) as date) as dat from {} " \
#                 "where real_time <= '2020-01-01 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 1) = 0 group by b.min, b.hou, b.dat".format(self.table)
#         sql_1 = "select * from (select distinct wtid,real_time, grGenPowerForProcess,grGenPowerForProcess_10min," \
#                 "grGenSpeedForProcess,grGenSpeedForProcess_10min,grTempGenBearDE_1sec,grTempGenBearDE_10min," \
#                 "grTempGenBearNDE_1sec,grTempGenBearNDE_10min, grTempGenStatorU_1sec,grTempGenStatorU_10min, " \
#                 "grTempGenStatorV_1sec,grTempGenStatorV_10min,grTempGenStatorW_1sec, grTempGenStatorW_10min," \
#                 "grTempGenCoolingAir_1sec,grTempGenCoolingAir_10min, grTempGenOutWind_1sec, grTempGenOutWind_10min, " \
#                 "giTurbineOperationMode,grPitchAngleBlade1,grPitchAngleBlade2,grPitchAngleBlade3,grPitchAngle," \
#                 "grPitchAngle_10min,grNacellePositionTotal,grBlade1TempBattBox_10min,grBlade2TempBattBox_10min," \
#                 "grBlade3TempBattBox_10min,grIL3_690V_KL3403,grTempRotorBearA_10min,grTempRotorBearB_10min," \
#                 "giFaultInformation,grBlade1TempMotor_10min,grBlade2TempMotor_10min,grBlade3TempMotor_10min," \
#                 "grTempOutdoor_1sec,grTempNacelle_1sec,grUL1_690V_KL3403,grUL2_690V_KL3403,grUL3_690V_KL3403," \
#                 "grCAN_ReactivePower,grRotorSpeedPDM,cast(DATE_FORMAT( real_time , '%H' ) as signed) as hou," \
#                 "cast(DATE_FORMAT( real_time ,'%i' ) as signed) as min," \
#                 "cast(DATE_FORMAT( real_time , '%Y-%m-%d ' ) as date) as dat from {} " \
#                 "where real_time >= '2021-04-06 00:00:00' and real_time <= '2021-04-09 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 1) = 0 group by b.min, b.hou, b.dat".format(self.table)
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
#         # sql1 = "select WTGS_CODE,FAULT_CODE,HAPPEN_TIME,END_TIME from iot_wind.tb_wind_fault where FARM_CODE = {} " \
#         #        "AND HAPPEN_TIME >= '2020-06-21 00:00:00' and HAPPEN_TIME <= '2021-06-22 00:00:00' and fault_code in" \
#         #        "(208001,208002,208003,208004,208005,208006,208007,208008,208011,208012,208013,208014,208015,208016," \
#         #        "208017,208018,208019,208020,208021,208022,208023,208024,208025,208026,208027,208028,208029,208030," \
#         #        "208031,208032,208033,208034,208035,208036,208037,208038,208039,208040,208041,208042,208043,208044," \
#         #        "208045,208046,208047,208048,208049,208050,208052,208053,208054,208055,208056,208057,208058,208059," \
#         #        "208060,208061,208062)" \
#         #        " order by WTGS_CODE, HAPPEN_TIME".format(str)
#         sql1 = "select WTGS_CODE,FAULT_CODE,HAPPEN_TIME,END_TIME from iot_wind.tb_wind_fault where FARM_CODE = {} " \
#                "AND HAPPEN_TIME >= '2021-04-07 00:00:00' and HAPPEN_TIME <= '2021-04-09 00:00:00' and fault_code in" \
#                "(208020)" \
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
#         label_data.to_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208020\\turbine10-7(Apr).xlsx")
#         print('\n\n**************标签数据结束****************************')
#         return label_data
#
#

#
#
# # step2: put more fault data to do DAE algorithm
def merge_data():
    d1 = pd.read_excel(r"../data/Generator_MY.xlsx", index_col=0)

    test = d1.copy()
    print('\n\n**************读取数据完毕****************************')
    multi_data = test.reset_index(drop=True)
    multi_data['FAULT_CODE'] = multi_data['FAULT_CODE'].apply(str)
    fault_data = multi_data.loc[multi_data['label']==1]
    # print(f'data_shape = {multi_data.shape}\nNP_ratio = {Counter(multi_data["label"])}')
    fea_data = (multi_data.drop(columns=['wtid', 'FAULT_CODE'])).reset_index(drop=True)
    return fea_data,multi_data,fault_data

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


    # def fault_GenPower(self):
    #     GenPower, results1 = self.threshold('grGenPowerForProcess')
    #     GenPower['component'] = '发电机'
    #     GenPower['Parameter'] = '发电机功率'
    #     GenPower['fault'] = '发电机功率异常'
    #     print('\n\n**************发电机功率异常---异常诊断结束****************************')
    #     return GenPower, results1

    # def fault_GenSpeed(self):
    #     GenSpeed, results2 = self.threshold('grGenSpeedForProcess')
    #     GenSpeed['component'] = '发电机'
    #     GenSpeed['Parameter'] = '发电机转速'
    #     GenSpeed['fault'] = '发电机转速异常'
    #     print('\n\n**************发电机转速异常---异常诊断结束****************************')
    #     return GenSpeed, results2

    def fault_TempGenBearDE(self):
        TempGenBearDE, results3 = self.threshold('grTempGenBearDE_1sec')
        TempGenBearDE['component'] = '发电机'
        TempGenBearDE['Parameter'] = '发电机驱动端轴承温度'
        TempGenBearDE['fault'] = '发电机驱动端轴承温度异常'
        print('\n\n**************发电机驱动端轴承温度异常---异常诊断结束****************************')
        return TempGenBearDE, results3

    def fault_TempGenBearNDE(self):
        TempGenBearNDE, results4 = self.threshold('grTempGenBearNDE_1sec')
        TempGenBearNDE['component'] = '发电机'
        TempGenBearNDE['Parameter'] = '发电机非驱动端轴承温度'
        TempGenBearNDE['fault'] = '发电机非驱动端轴承温度异常'
        print('\n\n**************发电机非驱动端轴承温度异常---异常诊断结束****************************')
        return TempGenBearNDE, results4

    def fault_TempStatorU(self):
        TempStatorU, results5 = self.threshold('grTempGenStatorU_1sec')
        TempStatorU['component'] = '发电机'
        TempStatorU['Parameter'] = '发电机定子U绕组温度'
        TempStatorU['fault'] = '发电机定子U绕组温度异常'
        print('\n\n**************发电机定子U绕组温度异常---异常诊断结束****************************')
        return TempStatorU, results5

    def fault_TempStatorV(self):
        TempStatorV, results6 = self.threshold('grTempGenStatorV_1sec')
        TempStatorV['component'] = '发电机'
        TempStatorV['Parameter'] = '发电机定子V绕组温度'
        TempStatorV['fault'] = '发电机定子V绕组温度异常'
        print('\n\n**************发电机定子V绕组温度异常---异常诊断结束****************************')
        return TempStatorV, results6

    def fault_TempStatorW(self):
        TempStatorW, results7 = self.threshold('grTempGenStatorW_1sec')
        TempStatorW['component'] = '发电机'
        TempStatorW['Parameter'] = '发电机定子W绕组温度'
        TempStatorW['fault'] = '发电机定子W绕组温度异常'
        print('\n\n**************发电机定子W绕组温度异常---异常诊断结束****************************')
        return TempStatorW, results7

    def fault_TempGenCoolingAir(self):
        TempGenCoolingAir, results8 = self.threshold('grTempGenCoolingAir_1sec')
        TempGenCoolingAir['component'] = '发电机'
        TempGenCoolingAir['Parameter'] = '发电机冷风温度'
        TempGenCoolingAir['fault'] = '发电机冷风温度异常'
        print('\n\n**************发电机冷风温度异常---异常诊断结束****************************')
        return TempGenCoolingAir, results8

    # def fault_TempGenOutWind(self):  # 208035&208038  Tempurature of generator out wind may too high or too low
    #     TempGenOutWind, results9 = self.threshold('grTempGenOutWind_1sec')
    #     TempGenOutWind['component'] = '发电机'
    #     TempGenOutWind['Parameter'] = '发电机冷风温度'
    #     TempGenOutWind['fault'] = '发电机出风口温度异常'
    #     print('\n\n**************发电机出风口温度异常---异常诊断结束****************************')
    #     return TempGenOutWind, results9

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
        #     GenPower, results1 = self.fault_GenPower()
        #     GenPower0 = self.Reli_day(GenPower, results1)
        # except Exception:
        #     GenPower0 = pd.DataFrame()
        #     print('\n\n**************发电机功率异常诊断错误****************************')
        # try:
        #     GenSpeed, results2 = self.fault_GenSpeed()
        #     GenSpeed0 = self.Reli_day(GenSpeed, results2)
        #     GenSpeed0 = pd.DataFrame()
        # except Exception:
        #     print('\n\n**************发电机转速异常诊断错误****************************')
        try:
            TempGenBearDE, results3 = self.fault_TempGenBearDE()
            TempGenBearDE0 = self.Reli_day( TempGenBearDE, results3)
        except Exception:
            TempGenBearDE0 = pd.DataFrame()
            print('\n\n**************发电机驱动端轴承温度异常诊断错误****************************')
        try:
            TempGenBearNDE, results4 = self.fault_TempGenBearNDE()
            TempGenBearNDE0 = self.Reli_day(TempGenBearNDE, results4)
        except Exception:
            TempGenBearNDE0 = pd.DataFrame()
            print('\n\n**************发电机非驱动端轴承温度异常诊断错误****************************')
        try:
            TempStatorU, results5= self.fault_TempStatorU()
            TempStatorU0 = self.Reli_day(TempStatorU, results5)
        except Exception:
            TempStatorU0 = pd.DataFrame()
            print('\n\n**************发电机定子U绕组温度异常诊断错误****************************')
        try:
            TempStatorV, results6 = self.fault_TempStatorV()
            TempStatorV0 = self.Reli_day( TempStatorV, results6)
        except Exception:
            TempStatorV0 = pd.DataFrame()
            print('\n\n**************发电机定子V绕组温度异常诊断错误****************************')
        try:
            TempStatorW, results7 = self.fault_TempStatorW()
            TempStatorW0 = self.Reli_day(TempStatorW, results7)
        except Exception:
            TempStatorW0 = pd.DataFrame()
            print('\n\n**************发电机定子W绕组温度异常诊断错误****************************')
        try:
            TempGenCoolingAir, results8 = self.fault_TempGenCoolingAir()
            TempGenCoolingAir0 = self.Reli_day( TempGenCoolingAir, results8)
        except Exception:
            TempGenCoolingAir0 = pd.DataFrame()
            print('\n\n**************发电机冷风温度异常诊断错误****************************')
        # try:
        #     TempGenOutWind, results9 = self.fault_TempGenOutWind()
        #     TempGenOutWind0 = self.Reli_day(TempGenOutWind, results9)
        # except Exception:
        #     TempGenOutWind0 = pd.DataFrame()
        #     print('\n\n**************发电机出风口温度异常诊断错误****************************')
        data = pd.concat([TempGenBearDE0,TempGenBearNDE0,TempStatorU0,TempStatorV0,TempStatorW0,
                          TempGenCoolingAir0]).reset_index(drop=True)
        data['farm_code'] = data['wtid'].apply(lambda x: int(str(x)[:5]))
        return data

    def Reli_Generator(self, data):
        data_select = data[['wtid', 'real_time', 'Reliability', 'level']].drop_duplicates()
        mean = data_select.groupby(['wtid', 'real_time'], as_index=False, group_keys=False).Reliability.mean()

        data_level = data_select[['wtid', 'real_time', 'level']]
        select_level = pd.crosstab([data_level['wtid'], data_level['real_time']], data_level['level']).reset_index()

        select_level['bad'] = select_level[list(set(select_level.columns.tolist()) - {'wtid', 'real_time', '良好'})].sum(axis=1).astype(np.int64)
        select_level['level'] = select_level.apply(lambda x: '良好' if x['bad'] == 0 else ('一般' if x['bad'] == 1 else ('较差' if x['bad'] == 2 else '差')), axis=1)
        select_level = select_level[['wtid', 'real_time', 'level']]
        score_day = pd.merge(mean, select_level, on=['wtid', 'real_time'], how='left')
        score_day = score_day.rename({'Reliability': 'generator_reliability', 'level': 'generator_level'}, axis=1)
        return score_day





