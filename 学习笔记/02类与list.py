#!/usr/bin/env python
#coding:utf8

class car(object):        #�½�һ����
    def __init__(self):
        self.num=0
        self.name=0
    def prt(self):
        print("�⳵num=",self.num,",name=",self.name)

carpark=[]       #һ��list������ʢ�����ʵ��
for i in range(10):
    a=car()             #һ�����ʵ��
    a.num=i
    a.name=i*i
    carpark.append(a)

for k in carpark:
    k.prt()