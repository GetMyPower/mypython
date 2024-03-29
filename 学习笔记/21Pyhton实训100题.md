## Python 面试实训 100 题，哪道难住了你？

1.	将元组 (1,2,3) 和集合 {4,5,6} 合并成一个列表。
```python
list((1,2,3)) + list({4,5,6})
```

2.	在列表 [1,2,3,4,5,6] 首尾分别添加整型元素 7 和 0。
```python
a = [1,2,3,4,5,6]
a.insert(0,7)
a.insert(-1,0)
```
3.	反转列表 [0,1,2,3,4,5,6,7] 。
```python
a = [0,1,2,3,4,5,6,7]
a.reverse()
```

4.	反转列表 [0,1,2,3,4,5,6,7] 后给出中元素 5 的索引号。
```python
a = [0,1,2,3,4,5,6,7]
a.reverse()
a.index(5)
```

5.	分别统计列表 [True,False,0,1,2] 中 True,False,0,1,2的元素个数，发现了什么？
```python
from collections import Counter
a = [True, False, 0, 1, 2]
result = Counter(a)
```

6.	从列表 [True, 1, 0, 'x', None, 'x', False, 2, True] 中删除元素'x'。
```python
a = [True, 1, 0, 'x', None, 'x', False, 2, True]
a = [x for x in a if x != 'x']
```

7.	从列表 [True, 1, 0, 'x', None, 'x', False, 2, True] 中删除索引号为4的元素。
```python
a = [True, 1, 0, 'x', None, 'x', False, 2, True]
a.pop(4)
```

8.	删除列表中索引号为奇数（或偶数）的元素。
```python
a = [0, 1, 2, 3, 4, 5, 6, 7, 8]
print(a[::2])  # 此为索引号为偶数的元素，即删除索引号为奇数的元素
print(a[1::2])  # 此为索引号为奇数的元素，即删除索引号为偶数的元素
```

9.	清空列表中的所有元素。
```python
a.clear()
```

10.	对列表 [3,0,8,5,7] 分别做升序和降序排列。
```python
a = [3, 0, 8, 5, 7]
a_ascending = sorted(a)  # 升序
a_descending = sorted(a, reverse=True)  # 降序
```

11.	将列表 [3,0,8,5,7] 中大于 5 元素置为1，其余元素置为0。
```python
import numpy as np
a = [3, 0, 8, 5, 7]
a = np.array(a)
a[a <= 5] = 0
a[a > 5] = 1
```

12.	遍历列表 ['x','y','z']，打印每一个元素及其对应的索引号。
```python
z = ['x', 'y', 'z']
for x in z:
    print(x, z.index(x))
```

13.	将列表 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 拆分为奇数组和偶数组两个列表。

14.	分别根据每一行的首元素和尾元素大小对二维列表 [[6, 5], [3, 7], [2, 8]] 排序。

15.	从列表 [1,4,7,2,5,8] 索引为3的位置开始，依次插入列表 ['x','y','z'] 的所有元素。

16.	快速生成由 [5,50) 区间内的整数组成的列表。

17.	若 a = [1,2,3]，令 b = a，执行 b[0] = 9， a[0]亦被改变。为何？如何避免？

18.	将列表 ['x','y','z'] 和 [1,2,3] 转成 [('x',1),('y',2),('z',3)] 的形式。

19.	以列表形式返回字典 {'Alice': 20, 'Beth': 18, 'Cecil': 21} 中所有的键。

20.	以列表形式返回字典 {'Alice': 20, 'Beth': 18, 'Cecil': 21} 中所有的值。

21.	以列表形式返回字典 {'Alice': 20, 'Beth': 18, 'Cecil': 21} 中所有键值对组成的元组。

22.	向字典 {'Alice': 20, 'Beth': 18, 'Cecil': 21} 中追加 'David':19 键值对，更新Cecil的值为17。

23.	删除字典 {'Alice': 20, 'Beth': 18, 'Cecil': 21} 中的Beth键后，清空该字典。

24.	判断 David 和 Alice 是否在字典 {'Alice': 20, 'Beth': 18, 'Cecil': 21} 中。

25.	遍历字典 {'Alice': 20, 'Beth': 18, 'Cecil': 21}，打印键值对。

26.	若 a = dict()，令 b = a，执行 b.update({'x':1})， a亦被改变。为何？如何避免？

27.	以列表 ['A','B','C','D','E','F','G','H'] 中的每一个元素为键，默认值都是0，创建一个字典。

28.	将二维结构 [['a',1],['b',2]] 和 (('x',3),('y',4)) 转成字典。

29.	将元组 (1,2) 和 (3,4) 合并成一个元组。

30.	将空间坐标元组 (1,2,3) 的三个元素解包对应到变量 x,y,z。

31.	返回元组 ('Alice','Beth','Cecil') 中 'Cecil' 元素的索引号。

32.	返回元组 (2,5,3,2,4) 中元素 2 的个数。

33.	判断 'Cecil' 是否在元组 ('Alice','Beth','Cecil') 中。

34.	返回在元组 (2,5,3,7) 索引号为2的位置插入元素 9 之后的新元组。

35.	创建一个空集合，增加 {'x','y','z'} 三个元素。

36.	删除集合 {'x','y','z'} 中的 'z' 元素，增j加元素 'w'，然后清空整个集合。

37.	返回集合 {'A','D','B'} 中未出现在集合 {'D','E','C'} 中的元素（差集）。

38.	返回两个集合 {'A','D','B'} 和 {'D','E','C'} 的并集。

39.	返回两个集合 {'A','D','B'} 和 {'D','E','C'} 的交集。

40.	返回两个集合 {'A','D','B'} 和 {'D','E','C'} 未重复的元素的集合。

41.	判断两个集合 {'A','D','B'} 和 {'D','E','C'} 是否有重复元素。

42.	判断集合 {'A','C'} 是否是集合 {'D','C','E','A'} 的子集。

43.	去除数组 [1,2,5,2,3,4,5,'x',4,'x'] 中的重复元素。

44.	返回字符串 'abCdEfg' 的全部大写、全部小写和大下写互换形式。

45.	判断字符串 'abCdEfg' 是否首字母大写，字母是否全部小写，字母是否全部大写。

46.	返回字符串 'this is python' 首字母大写以及字符串内每个单词首字母大写形式。

47.	判断字符串 'this is python' 是否以 'this' 开头，又是否以 'python' 结尾。

48.	返回字符串 'this is python' 中 'is' 的出现次数。

49.	返回字符串 'this is python' 中 'is' 首次出现和最后一次出现的位置。

50.	将字符串 'this is python' 切片成3个单词。

51.	返回字符串 'blog.csdn.net/xufive/article/details/102946961' 按路径分隔符切片的结果。

52.	将字符串 '2.72, 5, 7, 3.14' 以半角逗号切片后，再将各个元素转成浮点型或整形。

53.	判断字符串 'adS12K56' 是否完全为字母数字，是否全为数字，是否全为字母，是否全为ASCII码。

54.	将字符串 'there is python' 中的 'is' 替换为 'are'。

55.	清除字符串 '\t python \n' 左侧、右侧，以及左右两侧的空白字符。

56.	将三个全英文字符串（比如，'ok', 'hello', 'thank you'）分行打印，实现左对齐、右对齐和居中对齐效果。

57.	将三个字符串（比如，'Hello, 我是David', 'OK, 好', '很高兴认识你'）分行打印，实现左对齐、右对齐和居中效果。

58.	将三个字符串 '15', '127', '65535' 左侧补0成同样长度。

59.	提取 url 字符串 'https://blog.csdn.net/xufive' 中的协议名。

60.	将列表 ['a','b','c'] 中各个元素用'|'连接成一个字符串。

61.	将字符串 'abc' 相邻的两个字母之间加上半角逗号，生成新的字符串。

62.	从键盘输入手机号码，输出形如 'Mobile: 186 6677 7788' 的字符串。

63.	从键盘输入年月日时分秒，输出形如 '2019-05-01 12:00:00' 的字符串。

64.	给定两个浮点数 3.1415926 和 2.7182818，格式化输出字符串 'pi = 3.1416, e = 2.7183'。

65.	将 0.00774592 和 356800000 格式化输出为科学计数法字符串。

66.	将十进制整数 240 格式化为八进制和十六进制的字符串。

67.	将十进制整数 240 转为二进制、八进制、十六进制的字符串。

68.	将字符串 '10100' 按照二进制、八进制、十进制、十六进制转为整数。

69.	求二进制整数1010、八进制整数65、十进制整数52、十六进制整数b4的和。

70.	将列表 [0,1,2,3.14,'x',None,'',list(),{5}] 中各个元素转为布尔型。

71.	返回字符 'a' 和 'A' 的ASCII编码值。

72.	返回ASCII编码值为 57 和 122 的字符。

73.	将二维列表 [[0.468,0.975,0.446],[0.718,0.826,0.359]] 写成名为 csv_data 的 csv 格式的文件，并尝试用 excel 打开它。

74.	从 csv_data.csv 文件中读出二维列表。

75.	向 csv_data.csv 文件追加二维列表 [[1.468,1.975,1.446],[1.718,1.826,1.359]]，然后读出所有数据。

76.	交换变量 x 和 y 的值。

77.	判断给定的参数 x 是否是整形。

78.	判断给定的参数 x 是否为列表或元组。

79.	判断 'https://blog.csdn.net' 是否以 'http://' 或 'https://' 开头。若是，则返回 'http' 或 'https'；否则，返回None。

80.	判断 'https://blog.csdn.net' 是否以 '.com' 或 '.net' 结束。若是，则返回 'com' 或 'net'；否则，返回None。

81.	将列表 [3,'a',5.2,4,{},9,[]] 中 大于3的整数或浮点数置为1，其余置为0。

82.	a,b 是两个数字，返回其中较小者或最大者。

83.	找到列表 [8,5,2,4,3,6,5,5,1,4,5] 中出现最频繁的数字以及出现的次数。

84.	将二维列表 [[1], ['a','b'], [2.3, 4.5, 6.7]] 转为 一维列表。

85.	将等长的键列表和值列表转为字典。

86.	使用链状比较操作符重写逻辑表达式 a > 10 and a < 20。

87.	写一个函数，以0.1秒的间隔不换行打印30次由函数参数传入的字符，实现类似打字机的效果。

88.	数字列表求和。

89.	返回数字列表中的最大值和最小值。

90.	计算 5 的 3.5 方和 3 的立方根。

91.	对 3.1415926 四舍五入，保留小数点后5位。

92.	判断两个对象是在内存中是否是同一个。

93.	返回给定对象的属性和方法。

94.	计算字符串表达式 '(2+3)*5' 的值。

95.	实现字符串 'x={"name":"David", "age":18}' 包含的代码功能。

96.	使用 map 函数求列表 [2,3,4,5] 中每个元素的立方根。

97.	使用 sys.stdin.readline() 写一个和 input() 函数功能完全相同的函数。

98.	使用二维列表描述9x9围棋局面，'w'表示白色棋子，'b'表示黑色棋子，'-'表示无子，打印成下图左所示的文本棋盘。

99.	对于9x9围棋盘，用a-i标识各行，用1-9标识各列，设计函数go()，输入位置和颜色，即输出文本棋盘，模拟围棋对弈的过程。

100.	下图中是国际跳棋的初始局面，10x10的棋盘上只有50个深色格子可以落子，'w'表示白色棋子，'b'表示黑色棋子，'-'表示无子，字符串 phase = 'b'*20 + '-'*10 + 'w'*20 表示下图中的局面，请将 phase 打印成下图右所示的样子。


