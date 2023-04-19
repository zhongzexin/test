"""
    putpose: 利用tornado获取请求参数
    time: 2023/02/09
    author: lzz
"""

import tornado.ioloop
import tornado.web
import json
from configparser import ConfigParser
import os
# import torch
import main_30


# hdfs URL
# gpu_rate = eval(conf.get("gpuLimit", "r"))   # hdfs URL
# logger.info("显存使用率：{}".format(gpu_rate))
# torch.cuda.set_per_process_memory_fraction(gpu_rate, 0)      # 限制0号设备的显存使用率


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('能效分析启动')
        main_30.run()
        self.write('能效分析结束')

class HealthyHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('ok')


if __name__ == "__main__":
    application = tornado.web.Application([(r"/", MainHandler), (r"/healthy", HealthyHandler)])
    application.listen(10419)
    # server = tornado.web.HTTPServer(application)
    # server.bind(8000)
    # server.start(2)
    tornado.ioloop.IOLoop.current().start()
