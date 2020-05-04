import numpy as np

def unique(list1):
    x = np.array(list1)
    print(np.unique(x))

a = np.array([1, 2, 3, 1])
b = np.array([4, 5, 6, 4])
list = []
output = set()
for i in range(len(a)):
    output.add((a[i], b[i]))
print(output)
for i in range(len(a)):
    list.append((a[i], b[i]))
c = np.concatenate((a, b), axis=0).T
d = np.unique(c, axis=0)
e = np.count_nonzero(c == d[1], axis=0)
x = [3, 0, 1, 2, 2, 0, 1, 3, 3, 3, 4, 1, 4, 3, 0]
y = [1, 0, 4, 3, 2, 1, 4, 0, 3, 0, 4, 2, 3, 3, 1]
x = np.array(x)
y = np.array(y)
hist, xbins, ybins = np.histogram2d(y,x, bins=range(6))
z = np.sum(hist / len(x))
print(hist)
print(a.shape)
print(c)
print(d)
print(d[0])
print(e)
print(list)
print(z)
