"""
--coding: utf-8
--WT
--11/19/2021
"""

import pandas as pd

from Relibility_Prediction import DAE_NN
import Gearbox
import Rotor
import Converter
import Pitch
import Generator
from functools import reduce
# import pymysql
# import uuid
# from datetime import date
# import datetime
# from datetime import datetime, timedelta
import os


class main_day:
    def __init__(self,testdata,length):
        self.testdata = testdata
        self.length = length

    def extract_data(self,clean_data,testdata):
        column_name = clean_data.columns.values.tolist()
        column_name.remove('label')
        test_data = testdata[column_name]
        test_data= test_data.copy()
        test_data['label'] = 1
        new_data = pd.concat([clean_data, test_data]).reset_index(drop=True)
        return new_data

    def Reli(self, component, testdata):
        clean_data, multi_data, fault_data = component.merge_data()
        column_name = multi_data.columns.values.tolist()
        column_name.remove('label')
        test = testdata[column_name]
        test = test.copy()
        test['label'] = 1
        multi_data = pd.concat([multi_data, test]).reset_index(drop=True)
        return multi_data

    def Sub_Rotor_day(self):
        multi_data = self.Reli(Rotor, self.testdata)
        threshold = Rotor.anomaly(multi_data, self.length)
        Sub_Rotor = threshold.run()  # 子部件结果
        Reli_Rotor = threshold.Reli_Rotor(Sub_Rotor) # 部件结果
        return Sub_Rotor, Reli_Rotor

    def Sub_Gearbox_day(self):
        multi_data = self.Reli(Gearbox, self.testdata)
        threshold =Gearbox.anomaly(multi_data,self.length)
        Sub_Gearbox = threshold.run()  # 子部件结果
        Reli_Gearbox = threshold.Reli_Gearbox(Sub_Gearbox) # 部件结果
        return Sub_Gearbox, Reli_Gearbox

    def Sub_Generator_day(self):
        multi_data = self.Reli(Generator, self.testdata)
        threshold =Generator.anomaly(multi_data,self.length)
        Sub_Generator = threshold.run()  # 子部件结果
        Reli_Generator = threshold.Reli_Generator(Sub_Generator) # 部件结果
        return Sub_Generator, Reli_Generator

    def Sub_Converter_day(self):
        multi_data = self.Reli(Converter, self.testdata)
        threshold = Converter.anomaly(multi_data, self.length)
        Sub_Converter = threshold.run()  # 子部件结果
        Reli_Converter = threshold.Reli_Converter(Sub_Converter)  # 部件结果
        return Sub_Converter, Reli_Converter

    def Sub_Pitch_day(self):
        multi_data = self.Reli(Pitch, self.testdata)
        threshold = Pitch.anomaly(multi_data, self.length)
        Sub_Pitch= threshold.run()  # 子部件结果
        Reli_Pitch = threshold.Reli_Pitch(Sub_Pitch)  # 部件结果
        return Sub_Pitch, Reli_Pitch

    def level(self,dataframe):
        def turbine_condition(data):
            if data >= 0.8:
                return '良好'
            elif data >= 0.5:
                return '一般'
            elif data >= 0.3:
                return '较差'
            elif data >= 0:
                return '差'
        dataframe['turbine_level'] = dataframe.turbine_reliability.apply(turbine_condition)
        dataframe['Mechanical_level'] = dataframe.mechanical_energy.apply(turbine_condition)
        dataframe['Electrical_level'] = dataframe.electrical_energy.apply(turbine_condition)
        return dataframe

    def change_farm_code(self,data):
        data['turbine_code'] = data.turbine_code.apply(
            lambda x: str(x).replace('10006', '99930') if '10006' in str(x) else
            (str(x).replace('10007', '99940') if '10007' in str(x) else str(x).replace('10008', '99950')))
        return data

    def Component_day(self):
        Sub_Rotor, Reli_Rotor = self.Sub_Rotor_day()
        Sub_Gearbox, Reli_Gearbox = self.Sub_Gearbox_day()
        Sub_Generator, Reli_Generator = self.Sub_Generator_day()
        Sub_Converter, Reli_Converter = self.Sub_Converter_day()
        Sub_Pitch, Reli_Pitch = self.Sub_Pitch_day()
        sub_component = pd.concat([Sub_Rotor, Sub_Gearbox, Sub_Generator, Sub_Converter, Sub_Pitch])
        data_frames = [Reli_Rotor, Reli_Gearbox, Reli_Generator, Reli_Converter, Reli_Pitch]

        # not packed code, enter sub_component, data_frames
        turbine_day = reduce(lambda left, right: pd.merge(left, right, on=['wtid', 'real_time'],
                                                          how='left'), data_frames)
        turbine_day['farm_code'] = turbine_day.wtid.apply(lambda x: str(x)[:5])

        turbine_day['turbine_reliability'] = turbine_day['rotor_reliability'] * turbine_day['gearbox_reliability'] * \
                                             turbine_day['pitch_reliability'] * turbine_day['converter_reliability'] * \
                                             turbine_day[
                                                 'generator_reliability']

        turbine_day['mechanical_energy'] = turbine_day['rotor_reliability'] * turbine_day['gearbox_reliability'] * \
                                           turbine_day[
                                               'generator_reliability']
        turbine_day['electrical_energy'] = turbine_day['converter_reliability'] * turbine_day['generator_reliability']
        turbine_day = turbine_day.rename({'wtid': 'turbine_code'}, axis=1)

        def level(dataframe):
            def turbine_condition(data, a, b, c, d):
                if data >= a:
                    return '良好'
                elif data >= b:
                    return '一般'
                elif data >= c:
                    return '较差'
                elif data >= d:
                    return '差'
            if len(dataframe.real_time[0]) == 10:
                dataframe['turbine_level'] = dataframe.turbine_reliability.apply(turbine_condition, args=(0.95, 0.90, 0.85, 0))
                dataframe['Mechanical_level'] = dataframe.mechanical_energy.apply(turbine_condition, args=(0.97, 0.95, 0.93, 0))
                dataframe['Electrical_level'] = dataframe.electrical_energy.apply(turbine_condition, args=(0.97, 0.95, 0.93, 0))
            elif len(dataframe.real_time[0]) == 7:
                dataframe['turbine_level'] = dataframe.turbine_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
                dataframe['Mechanical_level'] = dataframe.mechanical_energy.apply(turbine_condition, args=(0.95, 0.90, 0.85, 0))
                dataframe['Electrical_level'] = dataframe.electrical_energy.apply(turbine_condition, args=(0.95, 0.90, 0.85, 0))
                dataframe['rotor_level'] = dataframe.rotor_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
                dataframe['pitch_level'] = dataframe.pitch_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
                dataframe['gearbox_level'] = dataframe.gearbox_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
                dataframe['generator_level'] = dataframe.generator_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
                dataframe['converter_level'] = dataframe.converter_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
            elif len(dataframe.real_time[0]) == 4:
                dataframe['turbine_level'] = dataframe.turbine_reliability.apply(turbine_condition, args=(0.85, 0.75, 0.65, 0))
                dataframe['Mechanical_level'] = dataframe.mechanical_energy.apply(turbine_condition, args=(0.90, 0.80, 0.70, 0))
                dataframe['Electrical_level'] = dataframe.electrical_energy.apply(turbine_condition, args=(0.93, 0.85, 0.75, 0))
                dataframe['rotor_level'] = dataframe.rotor_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
                dataframe['pitch_level'] = dataframe.pitch_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
                dataframe['gearbox_level'] = dataframe.gearbox_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
                dataframe['generator_level'] = dataframe.generator_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
                dataframe['converter_level'] = dataframe.converter_reliability.apply(turbine_condition, args=(0.9, 0.8, 0.7, 0))
            return dataframe

        turbine_day = level(turbine_day)
        # turbine_month = level(turbine_month)
        # turbine_year = level(turbine_year)

        turbine_info = [['10006', '中电甘肃瓜州安北第二A区'], ['10007', '中电甘肃瓜州安北第二B区'],
                        ['10008', '中电甘肃瓜州安北第六C区']]

        turbine_info = pd.DataFrame(turbine_info, columns=['new_farm_code', 'farm_name'])
        turbine_info['model'] = '双馈'

        turbine_day = pd.merge(turbine_day, turbine_info, left_on='farm_code', right_on='new_farm_code', how='left')
        # turbine_day = self.change_farm_code(turbine_day)
        sub_component = sub_component.rename({'wtid': 'turbine_code'}, axis=1)
        sub_component['farm_code'] = sub_component.farm_code.apply(lambda x: str(x))
        sub_component = pd.merge(sub_component, turbine_info, left_on='farm_code', right_on='new_farm_code', how='left')
        # sub_component = self.change_farm_code(sub_component)
        self.sub_component = sub_component
        return sub_component, turbine_day
        # return sub_component, turbine_day,turbine_month,turbine_year

    def generate_order(self,order_file) :
        select_order = self.sub_component.loc[((self.sub_component['level'] == '差') | (self.sub_component['level'] == '较差')) & ((self.sub_component['Reliability'] > 0) & (self.sub_component['Reliability'] < 0.9))]
        order = select_order.groupby(['turbine_code','fault', 'component', 'level'], as_index=False,
                                     group_keys=False).real_time.min()
        max = select_order.groupby(['turbine_code', 'fault'], as_index=False, group_keys=False).real_time.max()
        max = max.rename({'real_time': 'endingdate'}, axis=1)

        order = pd.merge(order, max, on=['turbine_code', 'fault'], how='left')
        order = order.rename({'real_time': 'startingdate'}, axis=1)

        import uuid
        from datetime import date
        import datetime
        order['turbine_code'] = order.turbine_code.apply(lambda x: str(x))
        order['orderID'] = order.apply(lambda x: str(uuid.uuid1())[:20], axis=1)
        order['order_style'] = '能效工单'
        # order['create_time'] = date.today()
        # order['deadline'] = date.today() + datetime.timedelta(days=7)
        order['create_time'] = datetime.datetime.now()
        order['deadline'] = datetime.datetime.now() + datetime.timedelta(days=7)
        order['order_status'] = order.deadline.apply(lambda X: '待处理' if date.today() < X else '已结束')
        order['accuracy'] = '--'

        fault_info = [['机舱齿轮箱油温异常', '齿轮箱油实际温度接近或超过故障触发温度', '1、检查齿轮散热系统（滤芯、温控阀、散热器、油泵电机、散热风扇）；2、检查齿轮箱齿轮和轴承；'],
                      ['齿轮箱驱动端轴承温度异常', '齿轮箱驱动端轴承实际温度接近或超过故障触发温度',
                       '1、检查齿轮散热系统（滤芯、温控阀、散热器、油泵电机、散热风扇）；2、检查齿轮箱驱动端轴承润滑和部件；3、检查齿轮箱齿轮；4、PT100损坏；5、接线虚接或信号传输中受干扰失真；'],
                      ['齿轮箱非驱动端轴承温度异常', '齿轮箱非驱动端轴承实际温度接近或超过故障触发温度',
                       '1、检查齿轮散热系统（滤芯、温控阀、散热器、油泵电机、散热风扇）；2、检查齿轮箱非驱动端轴承润滑和部件；3、检查齿轮箱齿轮；4、PT100损坏；5、接线虚接或信号传输中受干扰失真；'],
                      ['发电机驱动端轴承温度异常', '发电机驱动端轴承实际温度接近或超过故障触发温度',
                       '1、检查轴承润滑情况；2、检查排油是否通畅；3、轴承部件检查（滚道、滚子、保持架）；4、手动强制加热器启动，待温度升至-5℃以上起机运行'],
                      ['发电机非驱动端轴承温度异常', '发电机非驱动端轴承实际温度接近或超过故障触发温度',
                       '1、检查轴承润滑情况；2、检查排油是否通畅；3、轴承部件检查（滚道、滚子、保持架）；4、手动强制加热器启动，待温度升至-5℃以上起机运行'],
                      ['发电机定子U绕组温度异常', '发电机定子U绕组实际温度接近或超过故障触发温度',
                       '1、机舱发电机定子U绕组温度检测PT100损坏；2、机舱发电机定子U绕组温度检测回路接线松动；3、机舱发电机定子U绕组温度检测回路；4、机舱发电机定子U绕组温度检测模块330A1（丹控IOM5.1模块）故障；'],
                      ['发电机定子V绕组温度异常', '发电机定子V绕组实际温度接近或超过故障触发温度',
                       '1、机舱发电机定子V绕组温度检测PT100损坏；2、机舱发电机定子V绕组温度检测回路接线松动；3、机舱发电机定子V绕组温度检测回路；4、机舱发电机定子V绕组温度检测模块330A1（丹控IOM5.1模块）故障；'],
                      ['发电机定子W绕组温度异常', '发电机定子W绕组实际温度接近或超过故障触发温度',
                       '1、机舱发电机定子W绕组温度检测PT100损坏；2、机舱发电机定子W绕组温度检测回路接线松动；3、机舱发电机定子W绕组温度检测回路；4、机舱发电机定子W绕组温度检测模块330A1（丹控IOM5.1模块）故障；'],
                      ['发电机冷风温度异常', '齿轮箱油实际温度接近或超过故障触发温度', '1、检查齿轮散热系统（滤芯、温控阀、散热器、油泵电机、散热风扇）；2、检查齿轮箱齿轮和轴承；'],
                      ['桨叶1电机风扇温度异常', '桨叶1电机风扇实际温度接近或超过故障触发温度',
                       '1.登塔检查，紧固桨叶1温度传感器接线,2.检查线路，清理风扇,3.登机检查变桨电机风扇，发现风扇损坏则更换变桨电机风扇；'],
                      ['桨叶2电机风扇温度异常', '桨叶2电机风扇实际温度接近或超过故障触发温度',
                       '1.登塔检查，紧固桨叶2温度传感器接线,2.检查线路，清理风扇,3.登机检查变桨电机风扇，发现风扇损坏则更换变桨电机风扇；'],
                      ['桨叶3电机风扇温度异常', '桨叶3电机风扇实际温度接近或超过故障触发温度',
                       '1.登塔检查，紧固桨叶3温度传感器接线,2.检查线路，清理风扇,3.登机检查变桨电机风扇，发现风扇损坏则更换变桨电机风扇；'],
                      ['主轴轴承B温度异常', '齿轮箱油实际温度接近或超过故障触发温度', '1、轴承润滑不良，更换油脂；2、轴承油脂过多；3、轴承问题; 4、主机空转或停机，等待温度上升;'],
                      ['电网L1电压异常', '当主控检测到的电网电压超过规定上限时，故障报出，机组停机', '1、查看故障时机组数据，实地检查排查问题；'],
                      ['电网L2电压异常', '当主控检测到的电网电压超过规定上限时，故障报出，机组停机', '1、查看故障时机组数据，实地检查排查问题；'],
                      ['电网L3电压异常', '当主控检测到的电网电压超过规定上限时，故障报出，机组停机', '1、查看故障时机组数据，实地检查排查问题；'],
                      ['电网L1电流异常', '当主控检测到的电网L1电流高于规定上限时，故障报出，机组停机', '1、查看故障时机组数据，实地检查排查问题；'],
                      ['电网L2电流异常', '当主控检测到的电网L2电流高于规定上限时，故障报出，机组停机', '1、查看故障时机组数据，实地检查排查问题；'],
                      ['电网L3电流异常', '当主控检测到的电网L3电流高于规定上限时，故障报出，机组停机', '1、查看故障时机组数据，实地检查排查问题；']]

        fault_info = pd.DataFrame(fault_info, columns=['fault', 'principle', 'faultshooting'])
        order = pd.merge(order, fault_info, on='fault', how='left')
        if os.path.isfile(order_file):
            pre_order = pd.read_csv(order_file)
            final_order = order[(~order.turbine_code.isin(pre_order.turbine_code)) | (
                ~order.fault.isin(pre_order.fault))].reset_index(drop=True)
        else:
            final_order = order

        final_order['reliability_ID'] = final_order.index + 1
        final_order['farm_code'] = final_order.turbine_code.apply(lambda x: x[:5])
        return final_order

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def read_file(self, file_name, farm_name):
        sub_file_path = os.path.join(self.data_path, file_name)
        sub_file_name = os.listdir(sub_file_path)
        wd_id = pd.DataFrame()

        for id in farm_name:
            id = str(id)
            wd_filter_name = list(filter(lambda x: id in x, sub_file_name))
            temp = pd.DataFrame(wd_filter_name, columns=['wd_file_name'])
            temp['label'] = int(id)
            wd_id = wd_id.append(temp)

        wd_id = wd_id.reset_index(drop=True)
        return wd_id

    def drop_col(self,df, cutoff=0.1):
        n = len(df)
        cnt = df.count()
        cnt = cnt / n
        return df.loc[:, cnt[cnt >= cutoff].index]

    def drop_col_uniq(self,data):
        ori_cols = data.columns
        drop_col_name = []
        for col in ori_cols:
            col_unique_num = data[col].dropna().nunique()
            if col_unique_num == 1:
                drop_col_name.append(col)
        data.drop(columns=drop_col_name, inplace=True)

        data.dropna(subset=['s'], inplace=True)
        return data

    def getdata_time(self,dataframe, name):
        group_data = dataframe.groupby(pd.Grouper(key=name, freq='10min', sort=True))
        choose_data = pd.DataFrame()
        for key, value in group_data:
            if len(value) > 0:
                choose_data = choose_data.append(value.iloc[0, :])
        choose_data = choose_data.sort_values(["s"]).reset_index(drop=True)
        return choose_data

    def data_process(self, data, day_date):
        if data.iloc[0, 0] == 'deviceCode':
            data = data.rename(columns=data.iloc[0])
        else:
            data = data.rename(columns=data.iloc[1])
            data = data.drop(0)
        data = self.drop_col_uniq(data)
        data = self.drop_col(data, cutoff=0.1)
        data.drop_duplicates(inplace=True)
        data = data.sort_values(["s"]).reset_index(drop=True)
        data = data[data['s'].str.contains(day_date)]
        data['s'] = pd.to_datetime(data.s)
        data = self.getdata_time(data, 's')
        data.rename(columns={"deviceCode": "wtid", "s": "real_time", "ActiveStatuscode_1": "FAULT_CODE"}, inplace=True)
        data = data.drop(columns=['tenantCode'], axis=1)
        time_series = data['real_time']
        data = data.drop(columns=['real_time'], axis=1)
        data = data.astype(float)
        data['real_time'] = time_series
        return data

