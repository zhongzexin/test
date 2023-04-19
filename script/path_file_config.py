import configparser

# config = configparser.ConfigParser()

# config.read('file_path.ini')

# data_path = config.get('data','data_path')
# result_path = config.get('result','result_path')
# print(config)


# [sql1]
# ip = 10.168.1.34
# user = develop
# password = szly2022
# port = 3306
# db = guazhou_wind
# charset = utf8
# ;
# ;
# [sql2]
# ip = 10.168.1.246
# user = develop
# password = szly2022
# port =5029
# db =db
# charset = utf8



from pandas.core.frame import DataFrame

def read_file():
    conf = configparser.ConfigParser()
    conf.read(filenames=r'../config/file_path.ini', encoding='utf-8')

    sections = conf.sections()
    data_path = conf.get('data','data_path')
    result_path = conf.get('result','result_path')
    # 将farm = '[10007,10008]' 外围的字符串去除
    farm = conf.get('data', 'farm_code')
    farm = farm[1:-1]  # 去除最外层的方括号
    farm = [int(x) for x in farm.split(',')]  # 使用逗号分隔，并将字符串转换为整数类型，# 输出结果应该是 [10007, 10008]

    return data_path,result_path,farm


#
# def conn_gz():
#     conn = pymysql.connect(host=host1, port=port1, user=user1, password=password1, db=db1, charset=charset1)
#     return conn
#
# def conn_1(db):
#     conn = pymysql.connect(host=host2, port=port2, user=user2, password=password2, db=db, charset=charset2)
#     return conn
#
# def get_tables(db):
#     try:
#         con_table = conn_1(db)
#         sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = '{}' ".format(db)
#         cur = con_table.cursor()
#         cur.execute(sql)
#         data = DataFrame(list(cur.fetchall()))
#         con_table.close()
#         cur.close()
#         tables = list(data[0])
#     except Exception:
#         tables = []
#     return tables
#
#
# ########################################################################################
# def conn_10006():
#     conn_10006 = pymysql.connect(host='192.168.0.36', user='tangwei', password='123456', db='db10006_2021', port=5029,
#                            charset='utf8')
#     return conn_10006
#
# def get_table_10006():
#     table_10006 = []
#     for i in range(41, 107, 1):
#         if len(str(i)) == 2:
#             table = 't100060' + str(i) + '_all'
#         else:
#             table = 't10006' + str(i) + '_all'
#         table_10006.append(table)
#     return table_10006
#
# ######################################################################################
# def conn_10007():
#     conn_10007 = pymysql.connect(host='192.168.0.36', user='tangwei', password='123456', db='db10007_2021', port=5029,
#                            charset='utf8')
#     return conn_10007
#
# def get_table_10007():
#     table_10007 = []
#     for i in range(1, 135, 1):
#         if len(str(i)) == 1:
#             table = 't1000700' + str(i) + '_all'
#         elif len(str(i)) == 2:
#             table = 't100070' + str(i) + '_all'
#         else:
#             table = 't10007' + str(i) + '_all'
#         table_10007.append(table)
#     return table_10007
#
# ##########################################################################################
# def conn_10008():
#     conn_10008 = pymysql.connect(host='192.168.0.36', user='tangwei', password='123456', db='db10008_2021', port=5029,
#                            charset='utf8')
#     return conn_10008
#
# def get_table_10008():
#     table_10008 = []
#     for i in range(71, 135, 1):
#         if len(str(i)) == 2:
#             table = 't100080' + str(i) + '_all'
#         else:
#             table = 't10008' + str(i) + '_all'
#         table_10008.append(table)
#     return table_10008
#
