"""
--coding: utf-8
--WT
--01/12/2022
"""

from datetime import datetime, timedelta
from time import sleep

import numpy as np
import pandas as pd
import path_file_config as path_cog
import generate_order as order
from generate_order import DataProcessor
import warnings
warnings.filterwarnings('ignore')
import sys
import os
import windfunction as wf

def run():
    try:
        print(datetime.now())
        # Y = str(datetime.now().year)

        data_path, result_path,farm = path_cog.read_file()
        # farm = [10007,10008]
        data_file = os.listdir(data_path)

        # data_file = data_file[-1:]
        for day_date in data_file:
            file_date = str(datetime.now().year) +"-"+ day_date
            component_reliab_file = os.path.join(result_path, 'data_reliability_component_day%s.csv'%file_date)
            day_reliab_file = os.path.join(result_path, 'data_reliability_day.csv')
            order_file = os.path.join(result_path, 'data_reliability_order.csv')
            data_processor = DataProcessor(data_path)
            wd_name = data_processor.read_file(day_date, farm)  # 获取不同风场下风机的数据名
            for farm_code in farm:
                # farm_code = 10008
                tables = wd_name.loc[wd_name['label']==farm_code,'wd_file_name']
                # tables = tables[-17:]
                for table in tables:
                    # table ='10007033_2023-04-05.csv'
                    starttime = datetime.now()
                    if os.path.isfile(day_reliab_file):
                        already_precess_data = pd.read_csv(day_reliab_file)
                        already_precess_data['real_time'] = pd.to_datetime(already_precess_data['real_time'])

                        ori_data = pd.read_csv(os.path.join(data_path,day_date,table), header=None)
                        test_data = data_processor.data_process(ori_data, day_date)
                        test_data_timestamp = test_data['real_time'].max().date()

                        max_timestamp = already_precess_data.loc[
                            already_precess_data['turbine_code'] == test_data.loc[
                                0, 'wtid'], 'real_time'].max().date()

                        if test_data_timestamp<=max_timestamp:
                            length = 0
                        else:
                            length = test_data.shape[0]
                    else:
                        ori_data = pd.read_csv(os.path.join(data_path, day_date, table), header=None)
                        test_data = data_processor.data_process(ori_data, day_date)
                        test_data_timestamp = test_data['real_time'].max().date()
                        length = test_data.shape[0]

                    try:
                        if length > 0:
                            algo = order.main_day(test_data, length)  # 取出明阳所有机组一天数据
                            # sub_component, turbine_day, turbine_month, turbine_year = algo.Component_day()  #得到子部件状态，天、月、年周期能效值
                            sub_component, turbine_day = algo.Component_day()
                            # 不确定是否每次生成的 turbine_day 为两行， 此处采用第二行的数据作为存储结果
                            # turbine_day = turbine_day.loc[[1]]
                            final_order = algo.generate_order(order_file) # 生成工单
                            turbine_day = wf.wind_power_effcient(turbine_day,test_data)

                            if os.path.isfile(component_reliab_file):
                                sub_component.to_csv(component_reliab_file, mode='a+', header=False, index=None)
                            else:
                                sub_component.to_csv(component_reliab_file, mode='a+', header=True,index=None)

                            if os.path.isfile(day_reliab_file):
                                turbine_day.to_csv(day_reliab_file, mode='a+', header=False, index=None)
                            else:
                                turbine_day.to_csv(day_reliab_file, mode='a+', header=True, index=None)
                            if os.path.isfile(order_file):
                                final_order.to_csv(order_file, mode='a+', header=False,index=None)
                            else:
                                final_order.to_csv(order_file, mode='a+', header=True, index=None)
                            test = 1
                        else:
                            print('{}:{}已完成分析'.format(table[0:9], str(test_data_timestamp)))
                            # final_order = algo.generate_order(day_date)  # 生成工单
                    except Exception as err:
                        print(err)


    except Exception as err:
        print(err)







