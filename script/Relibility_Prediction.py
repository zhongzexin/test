"""
--coding: utf-8
--WT
--10/22/2021
"""

import pandas as pd
import numpy as np
import math

# import seaborn as sns
# from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
# from sklearn.model_selection import train_test_split
# from collections import Counter
# from xgboost import XGBClassifier
# from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from keras import layers


# # Part 1: collect generator fault feature from SCADA
# class data_extract:
#     def __int__(self, table):
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
#                 "where real_time >= '2020-12-01 00:00:00' and real_time <= '2020-12-30 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 10) = 0 group by b.min, b.hou, b.dat".format(self.table)
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
#                 "where real_time <= '2021-01-01 00:00:00' order by real_time) b " \
#                 "where MOD(b.min, 10) = 0 group by b.min, b.hou, b.dat".format(self.table)
#         data_0 = pd.read_sql(sql=sql_0, con=conn_0)
#         data_1 = pd.read_sql(sql=sql_1, con=conn_1)
#         return data_0, data_1
#
#     # combine 2020 year data with 2021 year data
#     def get_data(self):
#         table_list = sc.get_tables()
#         data = pd.DataFrame()
#         for table in table_list:
#             subdata_0, subdata_1 = self.run_sql(table)
#             data = pd.concat([data, subdata_0, subdata_1])
#         data = data.drop(columns=['hou', 'min', 'dat'])
#         data = data.dropna().reset_index(drop=True)
#         return data
#
#     # get fault data
#     def fault_data(self):
#         fconn = sc.fault_conn()
#         table_list = sc.get_tables()
#         total_fa = pd.DataFrame()
#         for table in table_list:
#             str = table[1:6]
#             sql1 = "select WTGS_CODE,FAULT_CODE,HAPPEN_TIME,END_TIME from iot_wind.tb_wind_fault where FARM_CODE = {} " \
#                    "AND HAPPEN_TIME >= '2020-12-01 00:00:00' and HAPPEN_TIME <= '2020-12-30 00:00:00' and fault_code in" \
#                    " (208001,208002,208003,208004,208005,208006,208007,208008,208011,208012,208013,208014,208015,208016,208017," \
#                    "208018,208019,208020,208021,208022,208023,208024,208025,208026,208027,208028,208029,208030,208031,208032," \
#                    "208033,208034,208035,208036,208037,208038,208039,208040,208041,208042,208043,208044,208045,208046,208047," \
#                    "208048,208049,208050,208052,208053,208054,208055,208056,208057,208058,208059,208060,208061,208062)" \
#                    " order by WTGS_CODE, HAPPEN_TIME".format(str)
#             fault = pd.read_sql(sql1, con=fconn)
#             total_fa = pd.concat([total_fa, fault])
#             fconn.close()
#         return total_fa
#
#     # label data and give fault code, 0:normal, 1:fault.
#     def label(self):
#         merge = pd.merge(self.get_data(), self.fault_data(), how='left', left_on='wtid', right_on='WTGS_CODE')
#         merge_pur = merge.loc[(merge['real_time'] > merge['HAPPEN_TIME']) & (merge['real_time'] < merge['END_TIME'])]
#         merge_code = merge_pur[['wtid', 'real_time', 'FAULT_CODE']]
#
#         label_data = pd.merge(self.get_data(), merge_code, on=['wtid', 'real_time'], how='left')
#         label_data['FAULT_CODE'] = label_data['FAULT_CODE'].apply(str)
#         label_data['label'] = label_data.FAULT_CODE.apply(lambda x: 0 if math.isnan(x) == True else 1)
#         print('\n\n**************标签数据结束****************************')
#         return label_data
#
#
# # part 2: 1)get label data; 2)collect fault data with as many types as possible into one excel form;
# #  3) Use confusion matrix to detect recall & precise
#
# def feature_check():
#     # For convenient, part 1 label_data are deposited into excel
#     d1 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208052\\turbine89-7(Nov).xlsx")
#     d2 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208052\\turbine83-7(Jan).xlsx")
#     d3 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208015\\turbine56-6(May).xlsx")
#     test = pd.concat([d1, d2, d3])
#     test = test.drop(['Unnamed: 0'], axis=1)
#
#     # collect happened fault data into one file
#     other = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\SC_feature_test.xlsx")
#     other = other.drop(['wtid', 'real_time'], axis=1)
#     test = test.drop(['wtid', 'real_time', 'FAULT_CODE'], axis=1)
#     all_data = pd.concat([other, test])  # merge label data with as many types fault data as possible
#
#     # use xgboost classification
#     column_name = all_data.columns.values.tolist()
#     column_name.remove('label')
#     X = all_data[column_name]
#     y = all_data['label']
#
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.3, random_state=30)
#     model = XGBClassifier()
#     model.fit(X_train, y_train, eval_metric='rmse')
#     pred = model.predict(X_test)
#     print('\n\n**************故障特征检测****************************')
#     print(f'data_shape = {all_data.shape}\nNP_ratio = {Counter(all_data["label"])}')
#     print(f'Accuracy = {accuracy_score(y_test, pred):.2f}\nRecall = {recall_score(y_test, pred):.2f}\n' \
#           f'precise = {precision_score(y_test, pred):.2f}\n')
#     cm = confusion_matrix(y_test, pred)
#     plt.figure(figsize=(8, 6))
#     plt.title('Confusion Matrix', size=16)
#     sns.heatmap(cm, annot=True, cmap='Greens')
#     plt.xlabel('Predicted')
#     plt.ylabel('Actual')
#     return ()
#
#
# def merge_data():
#     # For convenient, part 1 label_data are deposited into excel
#     # 208052
#     d1 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208052\\turbine89-7(Nov).xlsx")
#     d2 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208052\\turbine83-7(Jan).xlsx")
#     d3 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208052\\turbine73-8(Jan).xlsx")
#     d4 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208052\\turbine94-7(Dec).xlsx")
#     d5 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208052\\turbine102-8(Dec).xlsx")
#     d6 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208052\\turbine110-8(Dec).xlsx")
#
#     # 208015
#     d7 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208015\\turbine56-6(May).xlsx")
#     d8 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208015\\turbine125-8(May).xlsx")
#     d9 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208015\\turbine18-7(May).xlsx")
#     d10 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208015\\turbine43-7(Fen).xlsx")
#     d11 = pd.read_excel("C:\\Users\\lx\\Desktop\\瓜州项目\\瓜州双馈代码\\Generator_fault_data\\208015\\turbine52-6(Jan).xlsx")
#
#
#
#     test = pd.concat([d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11])
#     print('\n\n**************读取数据完毕****************************')
#     multi_data = test.drop(['Unnamed: 0'], axis=1)
#     multi_data['FAULT_CODE'] = multi_data['FAULT_CODE'].apply(str)
#     print(f'data_shape = {multi_data.shape}\nNP_ratio = {Counter(multi_data["label"])}')
#     fea_data = (multi_data.drop(columns=['wtid', 'FAULT_CODE'])).reset_index(drop=True)
#     return fea_data, multi_data


# Part 3: 1)collect fault data as much as possible； 2）using DAE to calculate reliability
class DAE_NN:
    def __init__(self, clean_data):
        self.clean_data = clean_data

    def multi(self):
        # divide data into normal and fault, 0:X_train, 1:X_test
        X_test = self.clean_data.loc[(self.clean_data['label'] == 1)]
        test_index = pd.Series(X_test.index)
        X_train = self.clean_data.loc[(self.clean_data['label'] == 0)]
        train_index = pd.Series(X_train.index)  # keep training and testing data index

        X_test = X_test.drop(['label', 'real_time'], axis=1)
        X_train = X_train.drop(['label', 'real_time'], axis=1)  # keep only fault feature
        # X_train = pd.DataFrame(MinMaxScaler().fit_transform(X_train)).set_index([train_index])
        # # X_train = pd.DataFrame(X_train).sample(frac=1)  # shuffle 100% data
        # X_train = pd.DataFrame(X_train) # shuffle 100% data
        # X_test = pd.DataFrame(MinMaxScaler().fit_transform(X_test)).set_index([test_index])
        max = X_train.max()
        min = X_train.min()
        X_train = (X_train - min) / (max - min)
        X_test = (X_test - min) / (max - min)
        return X_train, X_test

    def DAE_model(self):
        X_train, X_test = self.multi()
        # building autoEncoder model############################################################
        tf.random.set_seed(10)
        act_func = 'relu'
        # Input layer:
        model = keras.Sequential()
        # First hidden layer, connected to input vector X.
        model.add(layers.Dense(10, activation=act_func,
                               kernel_initializer='glorot_uniform',
                               kernel_regularizer=keras.regularizers.l2(0.0),
                               input_shape=(X_train.shape[1],)
                               )
                  )
        model.add(layers.Dense(3, activation=act_func,
                               kernel_initializer='glorot_uniform'))
        model.add(layers.Dense(10, activation=act_func,
                               kernel_initializer='glorot_uniform'))
        model.add(layers.Dense(X_train.shape[1],
                               kernel_initializer='glorot_uniform'))
        model.compile(loss='mse', optimizer='adam')
        # print(model.summary())

        # Train model for 100 epochs, batch size of 10: ########################################
        NUM_EPOCHS = 3
        BATCH_SIZE = 50

        history = model.fit(np.array(X_train), np.array(X_train),
                            batch_size=BATCH_SIZE,
                            epochs=NUM_EPOCHS,
                            validation_split=0.05,
                            verbose=1)

        # plt.plot(history.history['loss'],
        #          'b',
        #          label='Training loss')
        # plt.plot(history.history['val_loss'],
        #          'r',
        #          label='Validation loss')
        # plt.legend(loc='upper right')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss, [mse]')
        # # plt.ylim([0, .1])
        # plt.show()

        # fit testing data, anomaly diagnosis##################################################

        X_pred = model.predict(np.array(X_test))
        X_pred = pd.DataFrame(X_pred,
                              columns=X_test.columns)
        X_pred.index = X_test.index
        scored = pd.DataFrame()
        scored['Relibility'] = 1 - np.mean(np.abs(X_pred - X_test), axis=1)
        Reli_var = 1-np.abs(X_pred - np.abs(X_test))

        # fit training data, then loss_mae = mean(abs(predict - train)########################
        X_pred_train = model.predict(np.array(X_train))
        X_pred_train = pd.DataFrame(X_pred_train,
                                    columns=X_train.columns).set_index([pd.Series(X_train.index)])

        scored_train = pd.DataFrame()
        scored_train['Relibility'] = 1 - np.mean(np.abs(X_pred_train - X_train), axis=1)
        scored_train = scored_train.sort_index()
        Reli_var_train = 1-np.abs(X_pred_train - X_train)
        Reli_var_train = Reli_var_train.sort_index()

        sco = pd.concat([scored_train, scored], sort=False).sort_index()
        # sco.plot(logy=True, figsize=(10, 6), ylim=[5e-1, 1e0], color=['green'])
        sco_var = pd.concat([Reli_var_train,  Reli_var], sort=False).sort_index()

        column_name = self.clean_data.columns.values.tolist()
        column_name.remove('real_time')
        column_name.remove('label')
        sco_var.columns = column_name

        return sco, sco_var


