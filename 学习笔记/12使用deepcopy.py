# deepcopy��ʹ�ã��������ƶ�����ȫ�ٸ���һ����Ϊ�������¸��嵥������
from copy import deepcopy

# region (1)����b=a��ֵ
print("(1)����b=a��ֵ")
a = [1, 2, 3]
b = a
print("a=", a)
print("b=", b)
print("(a,b)=", (a, b))
a[0] = [100]  # �ı�a�е�һ��Ԫ�أ������Ǹı�����a
print("a[0]=", a[0])
print("(a,b)=", (a, b))
# endregion

print()

# region (2)����b=deepcopy(a)��ֵ
print("(2)����b=deepcopy(a)��ֵ")
a = [1, 2, 3]
b = deepcopy(a)
print("a=", a)
print("b=", b)
print("(a,b)=", (a, b))
a[0] = [100]  # �ı�a�е�һ��Ԫ�أ������Ǹı�����a
print("a[0]=", a[0])
print("(a,b)=", (a, b))
# endregion
