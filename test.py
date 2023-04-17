import pandas as pd
import numpy as np
import re
class Expossion():
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def func1(self):
        print("self.param1=%s"%(self.name))
    def func2(self):
        print("self.param2=%s"%(self.age))
if __name__ == '__main__':

    exp = Expossion("1","2")
    exp.func1()
    exp.func2()


