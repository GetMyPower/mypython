#!/usr/bin/env python
#coding:utf8

class car(object):        #新建一个类
    def __init__(self):
        self.num=0
        self.name=0
    def prt(self):
        print("这车num=",self.num,",name=",self.name)

carpark=[]       #一个list，用于盛放类的实体
for i in range(10):
    a=car()             #一个类的实体
    a.num=i
    a.name=i*i
    carpark.append(a)

for k in carpark:
    k.prt()