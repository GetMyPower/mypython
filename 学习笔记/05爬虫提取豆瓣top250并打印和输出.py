#coding:utf-8
#������ȡ����top250����ӡ�����
import requests
from lxml import html

K1=[]
for Q in range(10):
    url='https://movie.douban.com/top250?start={}&filter='.format(Q*25)    #ͨ��������ַ���ɣ��ó�1-10ҳ����ַ
    con=requests.get(url).content
    sel=html.fromstring(con)

    for i in sel.xpath('//div[@class="info"]'):
        title=i.xpath('div[@class="hd"]/a/span[@class="title"]/text()')[0]
        people=i.xpath('div[@class="bd"]/p/text()')[0].replace(" ","").replace("\xa0","").replace("\xe5","").replace("\xfb","").replace("\xee","").replace("\u0161","").replace("\xf4","").replace("\xf6","").replace("\n","")   #�������ڱ�������ı���
        yr_ctry_tp=i.xpath('div[@class="bd"]/p/text()')[1].replace(" ","").replace("\xa0","").replace("\n","").split('/')
        year=yr_ctry_tp[0]
        country=yr_ctry_tp[1]
        type =yr_ctry_tp[2]
        score= i.xpath('div[@class="bd"]/div[@class="star"]/span[@class="rating_num"]/text()')[0]   #����1
        score=i.xpath('div[@class="bd"]/div[@class="star"]/span[2]/text()')[0]              #����2
        cntnum=i.xpath('div[@class="bd"]/div[@class="star"]/span[4]/text()')[0]

        #print(Q*25+sel.xpath('//div[@class="info"]').index(i)+1,title,people,year,country,type,score,cntnum)

        #��list���б���
        K2 = []
        K2.append(str(Q*25+sel.xpath('//div[@class="info"]').index(i)+1))
        K2.append(title)
        K2.append(people)
        K2.append(year)
        K2.append(country)
        K2.append(type)
        K2.append(score)
        K2.append(cntnum)
        K1.append(K2)


for ooo in K1:
    print(ooo)

haha=open('TOP250_list.csv','w')
haha.writelines("���"+",����"+",��������"+",���"+",����"+",����"+",����"+",��������\n")
for ooo in K1:
    for ppp in ooo:
        haha.writelines(ppp+",")
    haha.writelines("\n")
haha.close()