# deepcopy的使用，将被复制对象完全再复制一遍作为独立的新个体单独存在
from copy import deepcopy

# region (1)采用b=a幅值
print("(1)采用b=a幅值")
a = [1, 2, 3]
b = a
print("a=", a)
print("b=", b)
print("(a,b)=", (a, b))
a[0] = [100]  # 改变a中的一个元素，而不是改变整个a
print("a[0]=", a[0])
print("(a,b)=", (a, b))
# endregion

print()

# region (2)采用b=deepcopy(a)幅值
print("(2)采用b=deepcopy(a)幅值")
a = [1, 2, 3]
b = deepcopy(a)
print("a=", a)
print("b=", b)
print("(a,b)=", (a, b))
a[0] = [100]  # 改变a中的一个元素，而不是改变整个a
print("a[0]=", a[0])
print("(a,b)=", (a, b))
# endregion
