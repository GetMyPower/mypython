#coding:utf-8
#爬虫提取豆瓣top250并打印和输出
import requests
from lxml import html

K1=[]
for Q in range(10):
    url='https://movie.douban.com/top250?start={}&filter='.format(Q*25)    #通过分析网址规律，得出1-10页的网址
    con=requests.get(url).content
    sel=html.fromstring(con)

    for i in sel.xpath('//div[@class="info"]'):
        title=i.xpath('div[@class="hd"]/a/span[@class="title"]/text()')[0]
        people=i.xpath('div[@class="bd"]/p/text()')[0].replace(" ","").replace("\xa0","").replace("\xe5","").replace("\xfb","").replace("\xee","").replace("\u0161","").replace("\xf4","").replace("\xf6","").replace("\n","")   #消除由于编码引起的报错
        yr_ctry_tp=i.xpath('div[@class="bd"]/p/text()')[1].replace(" ","").replace("\xa0","").replace("\n","").split('/')
        year=yr_ctry_tp[0]
        country=yr_ctry_tp[1]
        type =yr_ctry_tp[2]
        score= i.xpath('div[@class="bd"]/div[@class="star"]/span[@class="rating_num"]/text()')[0]   #方法1
        score=i.xpath('div[@class="bd"]/div[@class="star"]/span[2]/text()')[0]              #方法2
        cntnum=i.xpath('div[@class="bd"]/div[@class="star"]/span[4]/text()')[0]

        #print(Q*25+sel.xpath('//div[@class="info"]').index(i)+1,title,people,year,country,type,score,cntnum)

        #用list进行保存
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
haha.writelines("序号"+",名称"+",导演主演"+",年份"+",国家"+",类型"+",分数"+",评论人数\n")
for ooo in K1:
    for ppp in ooo:
        haha.writelines(ppp+",")
    haha.writelines("\n")
haha.close()