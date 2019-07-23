import numpy
import matplotlib
import matplotlib.pyplot as plt
import  math

N=100
x=[0]*N
y=[0]*N
for i in range(N):
    x[i]=i/N*10
    y[i]=math.sin(x[i])


plt.plot(y)
plt.show()


def cal_f_E0(E0):
    mu=3.7
    sigma=0.92
    a1=1/(100*(1-E0)*sigma*math.sqrt(2*3.1415))
    a2=-(math.log(100*(1-E0),math.e)-mu)*(math.log(100*(1-E0),math.e)-mu)/(2*sigma*sigma)
    a3=a1*math.exp(a2)
    return a3

xx=[]
yy=[]
for i in range(N):
    tmp=random.uniform(0.2,1)
    xx.append(tmp)
    yy.append(cal_f_E0(tmp))
plt.scatter(xx,yy)
plt.show()
