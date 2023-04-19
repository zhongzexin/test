import os
import pandas as pd
import numpy as np

def wind_power_effcient(turbine_day,test_data):
    coeff = pd.read_csv('../data/coeff.csv')
    wind_power = test_data[['grWindSpeed', 'grGenPowerForProcess']]

    wind_power = wind_power.sort_values(['grWindSpeed']).reset_index(drop=True)

    def compute_theory_power(wind_power, coeff, i):
        first_stage = wind_power.loc[(wind_power['grWindSpeed'] >= coeff.loc[i, 'min_speed']) & (
                    wind_power['grWindSpeed'] < coeff.loc[i, 'max_speed']), :]
        # 将多项式系数转换为可调用函数 poly_func
        poly_func = np.poly1d(coeff.iloc[i, 1:6])
        # 对 grWindSpeed 数据应用 poly_func 函数进行计算，得到 power 数组
        power = poly_func(first_stage['grWindSpeed'])
        # 将计算结果添加到 DataFrame 中
        first_stage['theory_power'] = power
        return first_stage

    data_power = pd.DataFrame()
    for i in range(coeff.shape[0]):
        stage_data = compute_theory_power(wind_power, coeff, i)
        data_power = data_power.append(stage_data)
    cp = data_power['grGenPowerForProcess'].mean() / abs(data_power['theory_power']).mean()

    energy_efficiency_level = round(cp / 0.4, 4)  # 计算风能能效值

    if energy_efficiency_level > 1:  # 定义风能能效等级
        energy_efficiency_level = 0.9999

        level = '良好'

    elif 0.80 < energy_efficiency_level < 1:
        level = '良好'

    elif 0.70 < energy_efficiency_level < 0.80:
        level = '一般'

    elif 0.60 < energy_efficiency_level < 0.70:
        level = '较差'

    elif energy_efficiency_level < 0 or energy_efficiency_level == 0:
        energy_efficiency_level = 0.1000
        level = '差'
    else:
        level = '差'

    deviation = (1 - round(cp / 0.598, 4))
    if deviation < 0:
        deviation = 0.01
    try:
        turbine_reliability_new_val = turbine_day['turbine_reliability'] * energy_efficiency_level
        turbine_level = str(turbine_day['turbine_level'].values)
    except:
        turbine_reliability = 0.1
        turbine_level = '差'
    if level == '较差' and turbine_level == '一般' or turbine_level == '较差' or turbine_level == '差':
        turbine_reliability_level = '较差'
    elif level == '差' and turbine_level == '一般' or turbine_level == '较差' or turbine_level == '差':
        turbine_reliability_level = '差'
    elif level == '一般' and turbine_level == '一般' or turbine_level == '较差' or turbine_level == '差':
        turbine_reliability_level = '较差'
    elif level == '良好' and turbine_level == '一般' or turbine_level == '较差' or turbine_level == '差':
        turbine_reliability_level = '一般'
    elif level == '较差' or level == '差' and turbine_level == '良好':
        turbine_reliability_level = '一般'
    else:
        turbine_reliability_level = '良好'
    turbine_day['turbine_reliability'] = turbine_reliability_new_val
    turbine_day['turbine_level'] = turbine_reliability_level
    turbine_day['deviation'] = deviation
    turbine_day['energy_efficiency_level'] = energy_efficiency_level
    turbine_day['level'] = level
    return turbine_day