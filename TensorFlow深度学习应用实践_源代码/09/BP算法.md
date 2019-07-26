参考资料https://www.zhihu.com/question/27239198?rf=24827633  
http://galaxy.agh.edu.pl/~vlsi/AI/backp_t_en/backprop.html  
## 1 关于多层神经网络

![多层神经网络表示](https://pic4.zhimg.com/80/808254232cd4983cac374c5cc2a1fc87_hd.jpg)多层神经网络表示  
其对应的表达式如下  
![其对应的表达式如下](https://pic4.zhimg.com/80/e62889afe359c859e9a6a1ad2a432ebb_hd.jpg)  
wmn(k)可以理解为：(k,k+1)层之间，右侧第m个神经元对第n个输入变量xn的权重


## 2 关于BP（BackPropagation算法）

同样是利用链式法则，BP算法则机智地避开了这种冗余，它对于每一个路径只访问一次就能求顶点对所有下层节点的偏导值。正如反向传播(BP)算法的名字说的那样，BP算法是反向(自上往下)来寻找路径的。从最上层的节点e开始，初始值为1，以层为单位进行处理。对于e的下一层的所有子节点，将1乘以e到某个节点路径上的偏导值，并将结果“堆放”在该子节点中。等e所在的层按照这样传播完毕后，第二层的每一个节点都"堆放"些值，然后我们针对每个节点，把它里面所有“堆放”的值求和，就得到了顶点e对该节点的偏导。然后将这些第二层的节点各自作为起始顶点，初始值设为顶点e对它们的偏导值，以"层"为单位重复上述传播过程，即可求出顶点e对每一层节点的偏导数。
![pic1](https://pic2.zhimg.com/80/ee59254c9432b47cfcc3b11eab3e5984_hd.jpg)  
![pic2](https://pic4.zhimg.com/80/986aacfebb87f4e9573fa2fe87f439d1_hd.jpg)  
以上图为例，节点c接受e发送的1\*2并堆放起来，节点d接受e发送的1\*3并堆放起来，至此第二层完毕，求出各节点总堆放量并继续向下一层发送。节点c向a发送2\*1并对堆放起来，节点c向b发送2\*1并堆放起来，节点d向b发送3\*1并堆放起来，至此第三层完毕，节点a堆放起来的量为2，节点b堆放起来的量为2\*1+3\*1=5, 即顶点e对b的偏导数为5.  
举个不太恰当的例子，如果把上图中的箭头表示欠钱的关系，即c→e表示e欠c的钱。以a, b为例，直接计算e对它们俩的偏导相当于a, b各自去讨薪。a向c讨薪，c说e欠我钱，你向他要。于是a又跨过c去找e。b先向c讨薪，同样又转向e，b又向d讨薪，再次转向e。可以看到，追款之路，充满艰辛，而且还有重复，即a, b 都从c转向e。  
而BP算法就是主动还款。e把所欠之钱还给c，d。c，d收到钱，乐呵地把钱转发给了a，b，皆大欢喜。  

## 3 BP公式
详见https://github.com/GetMyPower/mypython/blob/master/TensorFlow%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%BA%94%E7%94%A8%E5%AE%9E%E8%B7%B5_%E6%BA%90%E4%BB%A3%E7%A0%81/09/Principles%20of%20training%20multi-layer%20neural%20network%20using%20backpropagation.docx
